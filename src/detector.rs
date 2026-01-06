//! YOLO Object Detection Module
//!
//! Communicates with Python RKNN-Lite subprocess for NPU inference

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// Bounding box coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// Single detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class: String,
    pub confidence: f32,
    pub bbox: BBox,
}

/// Detection result for a frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub detections: Vec<Detection>,
    #[serde(default)]
    pub error: Option<String>,
}

impl Default for DetectionResult {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            detections: Vec::new(),
            error: None,
        }
    }
}

/// Request to detector thread
enum DetectorRequest {
    Detect(Vec<u8>),
    Shutdown,
}

/// YOLO Detector interface (thread-safe)
pub struct YoloDetector {
    request_tx: Sender<DetectorRequest>,
    last_result: Arc<Mutex<DetectionResult>>,
    _handle: thread::JoinHandle<()>,
}

// Implement Send + Sync for YoloDetector
unsafe impl Send for YoloDetector {}
unsafe impl Sync for YoloDetector {}

impl YoloDetector {
    /// Create and start the detector
    pub fn new() -> Result<Self> {
        let (request_tx, request_rx) = mpsc::channel::<DetectorRequest>();
        let last_result = Arc::new(Mutex::new(DetectionResult::default()));
        let result_clone = last_result.clone();

        // Spawn detector thread
        let handle = thread::spawn(move || {
            if let Err(e) = detector_thread(request_rx, result_clone) {
                tracing::error!("Detector thread error: {}", e);
            }
        });

        Ok(Self {
            request_tx,
            last_result,
            _handle: handle,
        })
    }

    /// Submit frame for detection (non-blocking)
    pub fn detect(&self, jpeg_data: Vec<u8>) -> Result<()> {
        self.request_tx
            .send(DetectorRequest::Detect(jpeg_data))
            .context("Failed to send detection request")?;
        Ok(())
    }

    /// Get latest detection result (non-blocking)
    pub fn get_last_result(&self) -> DetectionResult {
        self.last_result.lock().unwrap().clone()
    }
}

impl Drop for YoloDetector {
    fn drop(&mut self) {
        let _ = self.request_tx.send(DetectorRequest::Shutdown);
    }
}

/// Detector thread - manages Python subprocess
fn detector_thread(
    request_rx: Receiver<DetectorRequest>,
    last_result: Arc<Mutex<DetectionResult>>,
) -> Result<()> {
    tracing::info!("Starting YOLO detector subprocess...");

    // Spawn Python process
    let mut child = Command::new("python3")
        .arg("/home/angelo/imx415_streamer/yolo_detector.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to spawn YOLO detector")?;

    let mut stdin = child.stdin.take().context("No stdin")?;
    let stdout = child.stdout.take().context("No stdout")?;
    let mut reader = BufReader::new(stdout);

    // Wait for READY signal
    let mut ready_line = String::new();
    reader.read_line(&mut ready_line)?;
    if !ready_line.trim().eq("READY") {
        anyhow::bail!("Detector did not signal READY: {}", ready_line);
    }
    tracing::info!("YOLO detector ready!");

    // Process requests
    for request in request_rx {
        match request {
            DetectorRequest::Detect(jpeg_data) => {
                // Send length prefix + data
                let len = jpeg_data.len() as u32;
                if stdin.write_all(&len.to_le_bytes()).is_err() {
                    tracing::error!("Failed to write length to detector");
                    break;
                }
                if stdin.write_all(&jpeg_data).is_err() {
                    tracing::error!("Failed to write data to detector");
                    break;
                }
                if stdin.flush().is_err() {
                    tracing::error!("Failed to flush detector stdin");
                    break;
                }

                // Read JSON response
                let mut response_line = String::new();
                if reader.read_line(&mut response_line).is_err() {
                    tracing::error!("Failed to read detector response");
                    break;
                }

                // Parse JSON and update shared result
                match serde_json::from_str::<DetectionResult>(&response_line) {
                    Ok(result) => {
                        if let Ok(mut guard) = last_result.lock() {
                            *guard = result;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse detection result: {}", e);
                        if let Ok(mut guard) = last_result.lock() {
                            *guard = DetectionResult {
                                error: Some(format!("Parse error: {}", e)),
                                ..Default::default()
                            };
                        }
                    }
                }
            }
            DetectorRequest::Shutdown => {
                tracing::info!("Detector shutdown requested");
                break;
            }
        }
    }

    // Cleanup
    let _ = child.kill();
    let _ = child.wait();
    tracing::info!("YOLO detector stopped");

    Ok(())
}

/// Draw detection boxes on an image (modifies JPEG in-place would require re-encoding)
/// Returns a new JPEG with boxes drawn
pub fn draw_detections(jpeg_data: &[u8], detections: &[Detection]) -> Result<Vec<u8>> {
    use image::codecs::jpeg::{JpegDecoder, JpegEncoder};
    use image::{DynamicImage, Rgb};
    use std::io::Cursor;

    if detections.is_empty() {
        return Ok(jpeg_data.to_vec());
    }

    // Decode JPEG
    let decoder = JpegDecoder::new(Cursor::new(jpeg_data))?;
    let img = DynamicImage::from_decoder(decoder)?;
    let mut rgb_img = img.to_rgb8();

    // Colors
    let box_color = Rgb([255u8, 50u8, 50u8]); // Red box
    let label_bg = Rgb([255u8, 255u8, 255u8]); // White background
    let label_text = Rgb([200u8, 0u8, 0u8]); // Dark red text
    let thickness = 4;

    for det in detections {
        let x1 = det.bbox.x1.max(0) as u32;
        let y1 = det.bbox.y1.max(0) as u32;
        let x2 = (det.bbox.x2 as u32).min(rgb_img.width().saturating_sub(1));
        let y2 = (det.bbox.y2 as u32).min(rgb_img.height().saturating_sub(1));

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        // Draw rectangle (top, bottom, left, right lines)
        for t in 0..thickness {
            // Top line
            if y1 + t < rgb_img.height() {
                for x in x1..=x2 {
                    if x < rgb_img.width() {
                        rgb_img.put_pixel(x, y1 + t, box_color);
                    }
                }
            }
            // Bottom line
            if y2 >= t {
                let by = y2 - t;
                if by < rgb_img.height() {
                    for x in x1..=x2 {
                        if x < rgb_img.width() {
                            rgb_img.put_pixel(x, by, box_color);
                        }
                    }
                }
            }
            // Left line
            if x1 + t < rgb_img.width() {
                for y in y1..=y2 {
                    if y < rgb_img.height() {
                        rgb_img.put_pixel(x1 + t, y, box_color);
                    }
                }
            }
            // Right line
            if x2 >= t {
                let rx = x2 - t;
                if rx < rgb_img.width() {
                    for y in y1..=y2 {
                        if y < rgb_img.height() {
                            rgb_img.put_pixel(rx, y, box_color);
                        }
                    }
                }
            }
        }

        // Draw label with white background above the box
        let label = format!("{} {:.0}%", det.class, det.confidence * 100.0);
        let char_width = 12u32;  // Bigger characters
        let char_height = 24u32;
        let label_width = (label.len() as u32 * char_width).min(rgb_img.width() - x1);
        let label_height = char_height + 8; // Padding
        let label_y = if y1 >= label_height { y1 - label_height } else { y2 + 4 };

        // Draw white background for label
        for ly in 0..label_height {
            for lx in 0..label_width {
                let px = x1 + lx;
                let py = label_y + ly;
                if px < rgb_img.width() && py < rgb_img.height() {
                    rgb_img.put_pixel(px, py, label_bg);
                }
            }
        }

        // Draw simple block letters for the label
        let text_y = label_y + 4;
        let text_x = x1 + 4;
        draw_text(&mut rgb_img, &label, text_x, text_y, label_text, 2);
    }

    // Re-encode to JPEG
    let mut output = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut output, 85);
    encoder.encode(
        rgb_img.as_raw(),
        rgb_img.width(),
        rgb_img.height(),
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(output)
}

/// Draw simple blocky text on image
fn draw_text(img: &mut image::RgbImage, text: &str, x: u32, y: u32, color: image::Rgb<u8>, scale: u32) {
    // Simple 5x7 font bitmaps for common characters
    let font: std::collections::HashMap<char, [[u8; 5]; 7]> = [
        ('0', [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,1,1],[1,0,1,0,1],[1,1,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        ('1', [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]]),
        ('2', [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]]),
        ('3', [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        ('4', [[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]]),
        ('5', [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        ('6', [[0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        ('7', [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0]]),
        ('8', [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        ('9', [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,1,1,1,0]]),
        ('a', [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,1],[0,1,1,1,1],[1,0,0,0,1],[0,1,1,1,1]]),
        ('b', [[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0]]),
        ('c', [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,1,1,0]]),
        ('d', [[0,0,0,0,1],[0,0,0,0,1],[0,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1]]),
        ('e', [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,0],[0,1,1,1,0]]),
        ('f', [[0,0,1,1,0],[0,1,0,0,0],[0,1,0,0,0],[1,1,1,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0]]),
        ('g', [[0,0,0,0,0],[0,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,1,1,1,0]]),
        ('h', [[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]]),
        ('i', [[0,0,1,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]]),
        ('k', [[1,0,0,0,0],[1,0,0,0,0],[1,0,0,1,0],[1,0,1,0,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0]]),
        ('l', [[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]]),
        ('m', [[0,0,0,0,0],[0,0,0,0,0],[1,1,0,1,0],[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1]]),
        ('n', [[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]]),
        ('o', [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        ('p', [[0,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0]]),
        ('r', [[0,0,0,0,0],[0,0,0,0,0],[1,0,1,1,0],[1,1,0,0,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]]),
        ('s', [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,1],[1,0,0,0,0],[0,1,1,1,0],[0,0,0,0,1],[1,1,1,1,0]]),
        ('t', [[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0]]),
        ('u', [[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1]]),
        ('v', [[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]]),
        ('w', [[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1],[0,1,0,1,0]]),
        ('y', [[0,0,0,0,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]]),
        (' ', [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]),
        ('%', [[1,1,0,0,1],[1,1,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,1,1],[1,0,0,1,1]]),
        ('.', [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0]]),
        ('-', [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]),
        ('_', [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]]),
    ].iter().cloned().collect();

    let mut cursor_x = x;
    for ch in text.chars().flat_map(|c| c.to_lowercase()) {
        if let Some(bitmap) = font.get(&ch) {
            for (row_idx, row) in bitmap.iter().enumerate() {
                for (col_idx, &pixel) in row.iter().enumerate() {
                    if pixel == 1 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                let px = cursor_x + (col_idx as u32) * scale + sx;
                                let py = y + (row_idx as u32) * scale + sy;
                                if px < img.width() && py < img.height() {
                                    img.put_pixel(px, py, color);
                                }
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 6 * scale; // Character width + spacing
    }
}
