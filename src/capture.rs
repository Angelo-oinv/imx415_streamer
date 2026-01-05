//! V4L2 capture module for IMX415 sensor
//!
//! Uses byte-4 extraction with row averaging, upscaled to 4K

use anyhow::{Context, Result};
use image::GrayImage;
use image::codecs::jpeg::JpegEncoder;
use std::process::{Command, Stdio};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Frame capture configuration
pub struct CaptureConfig {
    pub device_path: String,
    pub sensor_subdev: String,
    pub output_width: u32,   // Final output: 3840
    pub output_height: u32,  // Final output: 2160
    pub stride: usize,       // Raw stride: 4864
    pub groups_per_row: usize, // 960
    pub link_frequency: u32,
    pub jpeg_quality: u8,
    pub temp_dir: PathBuf,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video9".to_string(),
            sensor_subdev: "/dev/v4l-subdev3".to_string(),
            output_width: 3840,
            output_height: 2160,
            stride: 4864,
            groups_per_row: 960,
            link_frequency: 0,
            jpeg_quality: 85,
            temp_dir: PathBuf::from("/tmp/imx415_capture"),
        }
    }
}

/// Frame capture instance
pub struct FrameCapture {
    config: CaptureConfig,
    // Native buffer: 960x1080 (byte-4 + row averaging)
    native_buffer: Vec<u8>,
    // Output buffer: 3840x2160 (4K upscaled)
    output_buffer: Vec<u8>,
    jpeg_buffer: Vec<u8>,
}

impl FrameCapture {
    pub fn new() -> Result<Self> {
        Self::with_config(CaptureConfig::default())
    }

    pub fn with_config(config: CaptureConfig) -> Result<Self> {
        fs::create_dir_all(&config.temp_dir)?;
        
        // Native: 960x1080
        let native_w = config.groups_per_row;
        let native_h = 1080;
        let native_size = native_w * native_h;
        
        // Output: 3840x2160
        let output_size = config.output_width as usize * config.output_height as usize;
        
        Ok(Self {
            config,
            native_buffer: vec![0u8; native_size],
            output_buffer: vec![0u8; output_size],
            jpeg_buffer: Vec::with_capacity(2 * 1024 * 1024),
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
        tracing::info!("Capture ready: {}x{} (4K upscaled grayscale)", 
                      self.config.output_width, self.config.output_height);
        Ok(())
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

    /// Extract byte-4 with row averaging → 960x1080
    fn extract_byte4_averaged(&mut self, raw: &[u8]) {
        let stride = self.config.stride;
        let groups = self.config.groups_per_row;
        let out_height = 1080;
        
        for out_y in 0..out_height {
            let row0 = out_y * 2;
            let row1 = row0 + 1;
            let row0_start = row0 * stride;
            let row1_start = row1 * stride;
            let out_row_start = out_y * groups;
            
            for g in 0..groups {
                let idx0 = row0_start + g * 5 + 4;
                let idx1 = row1_start + g * 5 + 4;
                
                let v0 = raw.get(idx0).copied().unwrap_or(0) as u16;
                let v1 = raw.get(idx1).copied().unwrap_or(0) as u16;
                
                self.native_buffer[out_row_start + g] = ((v0 + v1) / 2) as u8;
            }
        }
    }

    /// Upscale 960x1080 → 3840x2160 using bilinear interpolation
    fn upscale_to_4k(&mut self) {
        let src_w = self.config.groups_per_row;  // 960
        let src_h = 1080;
        let dst_w = self.config.output_width as usize;   // 3840
        let dst_h = self.config.output_height as usize;  // 2160
        
        // Scale factors (fixed point 16-bit fraction)
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
                
                // Get 4 source pixels
                let p00 = self.native_buffer[src_y0 * src_w + src_x0] as u32;
                let p01 = self.native_buffer[src_y0 * src_w + src_x1] as u32;
                let p10 = self.native_buffer[src_y1 * src_w + src_x0] as u32;
                let p11 = self.native_buffer[src_y1 * src_w + src_x1] as u32;
                
                // Bilinear interpolation
                let x_inv = 0x10000 - x_frac;
                let y_inv = 0x10000 - y_frac;
                
                let top = (p00 * x_inv + p01 * x_frac) >> 16;
                let bot = (p10 * x_inv + p11 * x_frac) >> 16;
                let val = (top * y_inv + bot * y_frac) >> 16;
                
                self.output_buffer[dst_row + dst_x] = val as u8;
            }
        }
    }

    fn encode_jpeg(&mut self) -> Result<Vec<u8>> {
        self.jpeg_buffer.clear();
        
        let image = GrayImage::from_raw(
            self.config.output_width,
            self.config.output_height,
            self.output_buffer.clone(),
        ).context("Failed to create grayscale image")?;
        
        let mut encoder = JpegEncoder::new_with_quality(
            &mut self.jpeg_buffer, 
            self.config.jpeg_quality
        );
        encoder.encode(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::L8,
        ).context("Failed to encode JPEG")?;
        
        Ok(self.jpeg_buffer.clone())
    }

    /// Capture and return 4K grayscale JPEG
    pub fn capture_jpeg_frame(&mut self) -> Result<Vec<u8>> {
        let raw_data = self.capture_raw_frame()?;
        self.extract_byte4_averaged(&raw_data);
        self.upscale_to_4k();
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
