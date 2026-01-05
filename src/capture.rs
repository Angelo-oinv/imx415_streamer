//! V4L2 capture module for IMX415 sensor
//!
//! Supports both grayscale (byte-4 method) and color (10-bit Bayer demosaic) modes

use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use image::codecs::jpeg::JpegEncoder;
use std::process::{Command, Stdio};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

const WIDTH: usize = 3840;
const HEIGHT: usize = 2160;
const STRIDE: usize = 4864;
const GROUPS_PER_ROW: usize = 960;

/// Capture mode
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CaptureMode {
    /// 4K Grayscale using byte-4 + row averaging (artifact-free)
    Grayscale,
    /// 4K Color using 10-bit Bayer demosaicing
    Color,
}

/// Frame capture configuration
pub struct CaptureConfig {
    pub device_path: String,
    pub sensor_subdev: String,
    pub mode: CaptureMode,
    pub link_frequency: u32,
    pub jpeg_quality: u8,
    pub gamma: f32,
    pub enable_white_balance: bool,
    pub temp_dir: PathBuf,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video9".to_string(),
            sensor_subdev: "/dev/v4l-subdev3".to_string(),
            mode: CaptureMode::Color,
            link_frequency: 0,
            jpeg_quality: 90,
            gamma: 2.2,
            enable_white_balance: true,
            temp_dir: PathBuf::from("/tmp/imx415_capture"),
        }
    }
}

/// Frame capture instance
pub struct FrameCapture {
    config: CaptureConfig,
    // 10-bit Bayer buffer (for color mode)
    bayer10: Vec<u16>,
    // RGB output buffer (for color mode)
    rgb_buffer: Vec<u8>,
    // Grayscale buffers
    gray_native: Vec<u8>,   // 960x1080
    gray_output: Vec<u8>,   // 3840x2160
    // JPEG output
    jpeg_buffer: Vec<u8>,
    // Gamma LUT
    gamma_lut: [u8; 1024],  // 10-bit input -> 8-bit output
}

impl FrameCapture {
    pub fn new() -> Result<Self> {
        Self::with_config(CaptureConfig::default())
    }

    pub fn with_config(config: CaptureConfig) -> Result<Self> {
        fs::create_dir_all(&config.temp_dir)?;
        
        // Build gamma LUT (10-bit to 8-bit with gamma)
        let mut gamma_lut = [0u8; 1024];
        let inv_gamma = 1.0 / config.gamma;
        for i in 0..1024 {
            gamma_lut[i] = ((i as f32 / 1023.0).powf(inv_gamma) * 255.0) as u8;
        }
        
        Ok(Self {
            config,
            bayer10: vec![0u16; WIDTH * HEIGHT],
            rgb_buffer: vec![0u8; WIDTH * HEIGHT * 3],
            gray_native: vec![0u8; GROUPS_PER_ROW * (HEIGHT / 2)],
            gray_output: vec![0u8; WIDTH * HEIGHT],
            jpeg_buffer: Vec::with_capacity(3 * 1024 * 1024),
            gamma_lut,
        })
    }

    pub fn setup_sensor(&self) -> Result<()> {
        let output = Command::new("v4l2-ctl")
            .args([
                "-d", &self.config.sensor_subdev,
                "--set-ctrl",
                &format!("link_frequency={}", self.config.link_frequency),
            ])
            .output()
            .context("Failed to run v4l2-ctl")?;
        
        if !output.status.success() {
            tracing::warn!("Could not set link frequency: {:?}", 
                          String::from_utf8_lossy(&output.stderr));
        }

        let _ = Command::new("v4l2-ctl")
            .args(["-d", &self.config.sensor_subdev, "--set-ctrl", "analogue_gain=0"])
            .output();

        tracing::info!("Sensor configured");
        Ok(())
    }

    pub fn start_streaming(&mut self) -> Result<()> {
        tracing::info!("Capture ready: {}x{} {:?}", WIDTH, HEIGHT, self.config.mode);
        Ok(())
    }
    
    pub fn set_mode(&mut self, mode: CaptureMode) {
        self.config.mode = mode;
        tracing::info!("Mode changed to {:?}", mode);
    }
    
    pub fn mode(&self) -> CaptureMode {
        self.config.mode
    }

    fn capture_raw_frame(&self) -> Result<Vec<u8>> {
        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let raw_path = self.config.temp_dir.join(format!("frame_{}.raw", frame_num % 4));
        
        let output = Command::new("v4l2-ctl")
            .args([
                "-d", &self.config.device_path,
                "--stream-mmap=4",
                "--stream-skip=1",
                "--stream-count=1",
                &format!("--stream-to={}", raw_path.display()),
            ])
            .stderr(Stdio::null())
            .output()
            .context("Failed to run v4l2-ctl capture")?;
        
        if !output.status.success() {
            anyhow::bail!("v4l2-ctl capture failed");
        }
        
        let raw_data = fs::read(&raw_path).context("Failed to read raw frame")?;
        let _ = fs::remove_file(&raw_path);
        
        Ok(raw_data)
    }

    /// Unpack CSI-2 packed SGBRG10 (5 bytes → 4 pixels at 10-bit)
    fn unpack_bayer10(&mut self, raw: &[u8]) {
        for y in 0..HEIGHT {
            let raw_row = y * STRIDE;
            let out_row = y * WIDTH;

            for x in 0..(WIDTH / 4) {
                let i = raw_row + x * 5;
                if i + 4 >= raw.len() {
                    break;
                }

                let b0 = raw[i + 0] as u16;
                let b1 = raw[i + 1] as u16;
                let b2 = raw[i + 2] as u16;
                let b3 = raw[i + 3] as u16;
                let b4 = raw[i + 4] as u16;

                self.bayer10[out_row + x * 4 + 0] = (b0 << 2) | ((b4 >> 0) & 0x3);
                self.bayer10[out_row + x * 4 + 1] = (b1 << 2) | ((b4 >> 2) & 0x3);
                self.bayer10[out_row + x * 4 + 2] = (b2 << 2) | ((b4 >> 4) & 0x3);
                self.bayer10[out_row + x * 4 + 3] = (b3 << 2) | ((b4 >> 6) & 0x3);
            }
        }
    }

    /// Safe Bayer access with clamping
    #[inline]
    fn bayer(&self, x: isize, y: isize) -> u16 {
        let x = x.clamp(0, (WIDTH - 1) as isize) as usize;
        let y = y.clamp(0, (HEIGHT - 1) as isize) as usize;
        self.bayer10[y * WIDTH + x]
    }

    /// GBRG bilinear demosaic (10-bit precision)
    fn demosaic_bayer(&mut self) {
        for y in 0..HEIGHT as isize {
            for x in 0..WIDTH as isize {
                let idx = (y as usize * WIDTH + x as usize) * 3;

                let (r, g, b) = match ((y & 1), (x & 1)) {
                    // G (row 0, col 0) - Green in GB row
                    (0, 0) => (
                        (self.bayer(x, y - 1) + self.bayer(x, y + 1)) / 2,
                        self.bayer(x, y),
                        (self.bayer(x - 1, y) + self.bayer(x + 1, y)) / 2,
                    ),
                    // B (row 0, col 1) - Blue in GB row
                    (0, 1) => (
                        (self.bayer(x - 1, y - 1)
                            + self.bayer(x + 1, y - 1)
                            + self.bayer(x - 1, y + 1)
                            + self.bayer(x + 1, y + 1)) / 4,
                        (self.bayer(x - 1, y)
                            + self.bayer(x + 1, y)
                            + self.bayer(x, y - 1)
                            + self.bayer(x, y + 1)) / 4,
                        self.bayer(x, y),
                    ),
                    // R (row 1, col 0) - Red in RG row
                    (1, 0) => (
                        self.bayer(x, y),
                        (self.bayer(x - 1, y)
                            + self.bayer(x + 1, y)
                            + self.bayer(x, y - 1)
                            + self.bayer(x, y + 1)) / 4,
                        (self.bayer(x - 1, y - 1)
                            + self.bayer(x + 1, y - 1)
                            + self.bayer(x - 1, y + 1)
                            + self.bayer(x + 1, y + 1)) / 4,
                    ),
                    // G (row 1, col 1) - Green in RG row
                    _ => (
                        (self.bayer(x - 1, y) + self.bayer(x + 1, y)) / 2,
                        self.bayer(x, y),
                        (self.bayer(x, y - 1) + self.bayer(x, y + 1)) / 2,
                    ),
                };

                // Store as 10-bit values (will apply gamma later)
                self.rgb_buffer[idx + 0] = (r.min(1023) >> 2) as u8;
                self.rgb_buffer[idx + 1] = (g.min(1023) >> 2) as u8;
                self.rgb_buffer[idx + 2] = (b.min(1023) >> 2) as u8;
            }
        }
    }

    /// Apply gray-world white balance
    fn apply_white_balance(&mut self) {
        if !self.config.enable_white_balance {
            return;
        }
        
        let pixels = WIDTH * HEIGHT;
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        
        for i in 0..pixels {
            r_sum += self.rgb_buffer[i * 3 + 0] as u64;
            g_sum += self.rgb_buffer[i * 3 + 1] as u64;
            b_sum += self.rgb_buffer[i * 3 + 2] as u64;
        }
        
        let r_avg = r_sum as f32 / pixels as f32;
        let g_avg = g_sum as f32 / pixels as f32;
        let b_avg = b_sum as f32 / pixels as f32;
        let avg = (r_avg + g_avg + b_avg) / 3.0;
        
        // Limit gains to prevent extreme correction
        let r_gain = (avg / r_avg).clamp(0.5, 2.0);
        let g_gain = (avg / g_avg).clamp(0.5, 2.0);
        let b_gain = (avg / b_avg).clamp(0.5, 2.0);
        
        for i in 0..pixels {
            self.rgb_buffer[i * 3 + 0] = (self.rgb_buffer[i * 3 + 0] as f32 * r_gain).min(255.0) as u8;
            self.rgb_buffer[i * 3 + 1] = (self.rgb_buffer[i * 3 + 1] as f32 * g_gain).min(255.0) as u8;
            self.rgb_buffer[i * 3 + 2] = (self.rgb_buffer[i * 3 + 2] as f32 * b_gain).min(255.0) as u8;
        }
    }

    /// Apply gamma correction
    fn apply_gamma(&mut self) {
        let inv_gamma = 1.0 / self.config.gamma;
        for byte in self.rgb_buffer.iter_mut() {
            let normalized = *byte as f32 / 255.0;
            *byte = (normalized.powf(inv_gamma) * 255.0) as u8;
        }
    }

    // ==================== GRAYSCALE MODE ====================

    /// Extract byte-4 with row averaging → 960x1080
    fn extract_grayscale(&mut self, raw: &[u8]) {
        for out_y in 0..(HEIGHT / 2) {
            let row0 = out_y * 2;
            let row1 = row0 + 1;
            let row0_start = row0 * STRIDE;
            let row1_start = row1 * STRIDE;
            let out_row_start = out_y * GROUPS_PER_ROW;
            
            for g in 0..GROUPS_PER_ROW {
                let idx0 = row0_start + g * 5 + 4;
                let idx1 = row1_start + g * 5 + 4;
                
                let v0 = raw.get(idx0).copied().unwrap_or(0) as u16;
                let v1 = raw.get(idx1).copied().unwrap_or(0) as u16;
                
                self.gray_native[out_row_start + g] = ((v0 + v1) / 2) as u8;
            }
        }
    }

    /// Upscale 960x1080 → 3840x2160 using bilinear interpolation
    fn upscale_grayscale(&mut self) {
        let src_w = GROUPS_PER_ROW;
        let src_h = HEIGHT / 2;
        let dst_w = WIDTH;
        let dst_h = HEIGHT;
        
        let x_ratio = ((src_w - 1) << 16) / (dst_w - 1);
        let y_ratio = ((src_h - 1) << 16) / (dst_h - 1);
        
        for dst_y in 0..dst_h {
            let src_y_fp = dst_y * y_ratio;
            let src_y0 = src_y_fp >> 16;
            let src_y1 = (src_y0 + 1).min(src_h - 1);
            let y_frac = (src_y_fp & 0xFFFF) as u32;
            
            let dst_row = dst_y * dst_w;
            
            for dst_x in 0..dst_w {
                let src_x_fp = dst_x * x_ratio;
                let src_x0 = src_x_fp >> 16;
                let src_x1 = (src_x0 + 1).min(src_w - 1);
                let x_frac = (src_x_fp & 0xFFFF) as u32;
                
                let p00 = self.gray_native[src_y0 * src_w + src_x0] as u32;
                let p01 = self.gray_native[src_y0 * src_w + src_x1] as u32;
                let p10 = self.gray_native[src_y1 * src_w + src_x0] as u32;
                let p11 = self.gray_native[src_y1 * src_w + src_x1] as u32;
                
                let x_inv = 0x10000 - x_frac;
                let y_inv = 0x10000 - y_frac;
                
                let top = (p00 * x_inv + p01 * x_frac) >> 16;
                let bot = (p10 * x_inv + p11 * x_frac) >> 16;
                let val = (top * y_inv + bot * y_frac) >> 16;
                
                self.gray_output[dst_row + dst_x] = val as u8;
            }
        }
    }

    // ==================== JPEG ENCODING ====================

    fn encode_jpeg(&mut self) -> Result<Vec<u8>> {
        self.jpeg_buffer.clear();
        
        match self.config.mode {
            CaptureMode::Color => {
                let image = RgbImage::from_raw(
                    WIDTH as u32,
                    HEIGHT as u32,
                    self.rgb_buffer.clone(),
                ).context("Failed to create RGB image")?;
                
                let mut encoder = JpegEncoder::new_with_quality(&mut self.jpeg_buffer, self.config.jpeg_quality);
                encoder.encode(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgb8,
                ).context("Failed to encode JPEG")?;
            }
            CaptureMode::Grayscale => {
                let image = GrayImage::from_raw(
                    WIDTH as u32,
                    HEIGHT as u32,
                    self.gray_output.clone(),
                ).context("Failed to create grayscale image")?;
                
                let mut encoder = JpegEncoder::new_with_quality(&mut self.jpeg_buffer, self.config.jpeg_quality);
                encoder.encode(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::L8,
                ).context("Failed to encode JPEG")?;
            }
        }
        
        Ok(self.jpeg_buffer.clone())
    }

    /// Capture and return JPEG-encoded frame
    pub fn capture_jpeg_frame(&mut self) -> Result<Vec<u8>> {
        let raw_data = self.capture_raw_frame()?;
        
        match self.config.mode {
            CaptureMode::Color => {
                self.unpack_bayer10(&raw_data);
                self.demosaic_bayer();
                self.apply_white_balance();
                self.apply_gamma();
            }
            CaptureMode::Grayscale => {
                self.extract_grayscale(&raw_data);
                self.upscale_grayscale();
            }
        }
        
        self.encode_jpeg()
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }
}

impl Drop for FrameCapture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.config.temp_dir);
        tracing::info!("Capture stopped");
    }
}
