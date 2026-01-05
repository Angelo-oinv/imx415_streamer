//! V4L2 capture module for IMX415 sensor
//!
//! Handles raw frame capture from /dev/video9 (CIF bypass path)
//! and extracts the clean grayscale data from byte 4 of each 5-byte group.

use anyhow::{Context, Result};
use image::GrayImage;
use image::codecs::jpeg::JpegEncoder;
use std::process::{Command, Stdio};
use std::io::Read;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Frame capture configuration
pub struct CaptureConfig {
    pub device_path: String,
    pub sensor_subdev: String,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub groups_per_row: usize,
    pub link_frequency: u32,
    pub jpeg_quality: u8,
    pub temp_dir: PathBuf,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video9".to_string(),
            sensor_subdev: "/dev/v4l-subdev3".to_string(),
            width: 3840,
            height: 2160,
            stride: 4864,
            groups_per_row: 960,
            link_frequency: 0, // 297MHz for stability
            jpeg_quality: 85,
            temp_dir: PathBuf::from("/tmp/imx415_capture"),
        }
    }
}

/// Frame capture instance
pub struct FrameCapture {
    config: CaptureConfig,
    // Reusable buffers
    grayscale_buffer: Vec<u8>,
    upscaled_buffer: Vec<u8>,
    jpeg_buffer: Vec<u8>,
}

impl FrameCapture {
    /// Create new capture instance
    pub fn new() -> Result<Self> {
        Self::with_config(CaptureConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: CaptureConfig) -> Result<Self> {
        // Create temp directory
        fs::create_dir_all(&config.temp_dir)?;
        
        // Pre-allocate buffers
        let native_size = config.groups_per_row * config.height as usize;
        let upscaled_size = config.width as usize * config.height as usize;
        
        Ok(Self {
            config,
            grayscale_buffer: vec![0u8; native_size],
            upscaled_buffer: vec![0u8; upscaled_size],
            jpeg_buffer: Vec::with_capacity(2 * 1024 * 1024), // 2MB initial
        })
    }

    /// Setup sensor parameters
    pub fn setup_sensor(&self) -> Result<()> {
        // Set link frequency for stable MIPI transfer
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

        // Reset gain
        let _ = Command::new("v4l2-ctl")
            .args(["-d", &self.config.sensor_subdev, "--set-ctrl", "analogue_gain=0"])
            .output();

        tracing::info!("Sensor configured: link_freq={}", self.config.link_frequency);
        Ok(())
    }

    /// Start video streaming (no-op for v4l2-ctl based capture)
    pub fn start_streaming(&mut self) -> Result<()> {
        tracing::info!("Capture ready (using v4l2-ctl)");
        Ok(())
    }

    /// Capture a single raw frame using v4l2-ctl
    fn capture_raw_frame(&self) -> Result<Vec<u8>> {
        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let raw_path = self.config.temp_dir.join(format!("frame_{}.raw", frame_num % 4));
        
        // Capture using v4l2-ctl
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
        
        // Read the raw file
        let raw_data = fs::read(&raw_path)
            .context("Failed to read raw frame")?;
        
        // Clean up (optional, keep for debugging)
        let _ = fs::remove_file(&raw_path);
        
        Ok(raw_data)
    }

    /// Extract grayscale from raw data
    /// 
    /// The raw data is 10-bit packed (5 bytes = 4 pixels), but only
    /// byte 4 of each 5-byte group contains clean image data.
    fn extract_grayscale(&mut self, raw_data: &[u8]) {
        let height = self.config.height as usize;
        let stride = self.config.stride;
        let groups = self.config.groups_per_row;
        
        // Extract byte 4 from each 5-byte group
        for row in 0..height {
            let row_start = row * stride;
            let output_row_start = row * groups;
            
            for g in 0..groups {
                let idx = row_start + g * 5 + 4;
                if idx < raw_data.len() {
                    self.grayscale_buffer[output_row_start + g] = raw_data[idx];
                }
            }
        }
    }

    /// Upscale grayscale to full resolution using bilinear interpolation
    fn upscale_grayscale(&mut self) {
        let src_width = self.config.groups_per_row;
        let src_height = self.config.height as usize;
        let dst_width = self.config.width as usize;
        let dst_height = self.config.height as usize;
        
        // Use integer arithmetic for speed
        let x_scale = (src_width << 16) / dst_width;
        let y_scale = (src_height << 16) / dst_height;
        
        for dst_y in 0..dst_height {
            let src_y = ((dst_y * y_scale) >> 16).min(src_height - 1);
            let dst_row = dst_y * dst_width;
            let src_row = src_y * src_width;
            
            for dst_x in 0..dst_width {
                let src_x = ((dst_x * x_scale) >> 16).min(src_width - 1);
                self.upscaled_buffer[dst_row + dst_x] = self.grayscale_buffer[src_row + src_x];
            }
        }
    }

    /// Encode image to JPEG
    fn encode_jpeg(&mut self) -> Result<Vec<u8>> {
        self.jpeg_buffer.clear();
        
        let image = GrayImage::from_raw(
            self.config.width,
            self.config.height,
            self.upscaled_buffer.clone(),
        ).context("Failed to create image")?;
        
        let mut encoder = JpegEncoder::new_with_quality(&mut self.jpeg_buffer, self.config.jpeg_quality);
        encoder.encode(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::L8,
        ).context("Failed to encode JPEG")?;
        
        Ok(self.jpeg_buffer.clone())
    }

    /// Capture and return JPEG-encoded frame
    pub fn capture_jpeg_frame(&mut self) -> Result<Vec<u8>> {
        // Capture raw
        let raw_data = self.capture_raw_frame()?;
        
        // Extract grayscale
        self.extract_grayscale(&raw_data);
        
        // Upscale
        self.upscale_grayscale();
        
        // Encode to JPEG
        self.encode_jpeg()
    }

    /// Get current configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }
}

impl Drop for FrameCapture {
    fn drop(&mut self) {
        // Clean up temp directory
        let _ = fs::remove_dir_all(&self.config.temp_dir);
        tracing::info!("Capture stopped");
    }
}
