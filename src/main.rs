//! IMX415 Camera Streamer for Rock5C
//!
//! Captures frames from IMX415 sensor (bypassing ISP) and
//! streams them to web browsers via MJPEG or single frame endpoints.
//! Supports both grayscale (artifact-free) and color (experimental) modes.
//! Optional YOLO object detection via Rock5C NPU (RKNN-Lite).

mod capture;
mod detector;

use anyhow::Result;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use bytes::Bytes;
use capture::{CaptureMode, FrameCapture};
use detector::{DetectionResult, YoloDetector};
use parking_lot::RwLock;
use std::{sync::Arc, time::Duration};
use tokio::time::interval;
use tokio_stream::wrappers::IntervalStream;
use futures::StreamExt;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

/// Shared application state
struct AppState {
    current_frame: RwLock<Option<Bytes>>,
    capture: RwLock<Option<FrameCapture>>,
    frame_count: RwLock<u64>,
    current_mode: RwLock<CaptureMode>,
    // Detection state
    detector: RwLock<Option<YoloDetector>>,
    detection_enabled: RwLock<bool>,
    last_detections: RwLock<DetectionResult>,
}

impl AppState {
    fn new() -> Self {
        Self {
            current_frame: RwLock::new(None),
            capture: RwLock::new(None),
            frame_count: RwLock::new(0),
            current_mode: RwLock::new(CaptureMode::Grayscale), // Start with grayscale (stable)
            detector: RwLock::new(None),
            detection_enabled: RwLock::new(false),
            last_detections: RwLock::new(DetectionResult::default()),
        }
    }
}

type SharedState = Arc<AppState>;

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("IMX415 Streamer starting...");

    let mut capture = FrameCapture::new()?;
    capture.setup_sensor()?;
    capture.set_mode(CaptureMode::Grayscale); // Start with grayscale
    capture.start_streaming()?;
    
    info!("Camera initialized");

    let state = Arc::new(AppState::new());
    *state.capture.write() = Some(capture);

    // Try to initialize YOLO detector (optional - will work without it)
    match YoloDetector::new() {
        Ok(detector) => {
            info!("YOLO detector initialized (NPU)");
            *state.detector.write() = Some(detector);
        }
        Err(e) => {
            info!("YOLO detector not available: {} (detection disabled)", e);
        }
    }

    let capture_state = state.clone();
    tokio::spawn(async move {
        capture_loop(capture_state).await;
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/frame.jpg", get(frame_handler))
        .route("/stream", get(mjpeg_stream_handler))
        .route("/status", get(status_handler))
        .route("/mode/:mode", get(set_mode_handler))
        .route("/detect/:enabled", get(set_detection_handler))
        .route("/detections", get(detections_handler))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    info!("Starting web server on http://{}", addr);
    info!("  - Live view: http://<ip>:8080/");
    info!("  - Single frame: http://<ip>:8080/frame.jpg");
    info!("  - MJPEG stream: http://<ip>:8080/stream");
    info!("  - Set mode: http://<ip>:8080/mode/grayscale or /mode/color");
    info!("  - Toggle detection: http://<ip>:8080/detect/on or /detect/off");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn capture_loop(state: SharedState) {
    let mut interval = interval(Duration::from_millis(33));
    let mut detection_frame_counter = 0u32;
    
    loop {
        interval.tick().await;
        
        let frame_result = {
            let mut capture_guard = state.capture.write();
            if let Some(ref mut capture) = *capture_guard {
                capture.capture_jpeg_frame()
            } else {
                continue;
            }
        };
        
        match frame_result {
            Ok(mut jpeg_data) => {
                let detection_enabled = *state.detection_enabled.read();
                
                // Run detection every 3rd frame to maintain framerate
                if detection_enabled {
                    detection_frame_counter += 1;
                    
                    if detection_frame_counter % 3 == 0 {
                        // Send frame to detector
                        if let Some(ref detector) = *state.detector.read() {
                            if detection_frame_counter % 30 == 0 {
                                tracing::info!("Sending frame {} to detector ({} bytes)", detection_frame_counter, jpeg_data.len());
                            }
                            let _ = detector.detect(jpeg_data.clone());
                        }
                    }
                    
                    // Get latest detection results
                    if let Some(ref detector) = *state.detector.read() {
                        let result = detector.get_last_result();
                        *state.last_detections.write() = result;
                    }
                    
                    // Draw detection boxes on frame
                    let detections = state.last_detections.read();
                    if !detections.detections.is_empty() {
                        match detector::draw_detections(&jpeg_data, &detections.detections) {
                            Ok(annotated) => jpeg_data = annotated,
                            Err(e) => tracing::warn!("Failed to draw detections: {}", e),
                        }
                    }
                }
                
                *state.current_frame.write() = Some(Bytes::from(jpeg_data));
                *state.frame_count.write() += 1;
            }
            Err(e) => {
                error!("Capture error: {}", e);
            }
        }
    }
}

/// Set capture mode endpoint
async fn set_mode_handler(
    State(state): State<SharedState>,
    Path(mode): Path<String>,
) -> impl IntoResponse {
    let new_mode = match mode.to_lowercase().as_str() {
        "grayscale" | "gray" | "g" => CaptureMode::Grayscale,
        "color" | "c" => CaptureMode::Color,
        _ => {
            return axum::Json(serde_json::json!({
                "error": "Invalid mode. Use 'grayscale' or 'color'"
            }));
        }
    };
    
    {
        let mut capture_guard = state.capture.write();
        if let Some(ref mut capture) = *capture_guard {
            capture.set_mode(new_mode);
        }
    }
    *state.current_mode.write() = new_mode;
    
    axum::Json(serde_json::json!({
        "mode": format!("{:?}", new_mode),
        "success": true
    }))
}

/// Toggle detection endpoint
async fn set_detection_handler(
    State(state): State<SharedState>,
    Path(enabled): Path<String>,
) -> axum::Json<serde_json::Value> {
    let enable = match enabled.to_lowercase().as_str() {
        "on" | "true" | "1" | "enable" | "enabled" => true,
        "off" | "false" | "0" | "disable" | "disabled" => false,
        _ => {
            return axum::Json(serde_json::json!({
                "error": "Invalid value. Use 'on' or 'off'"
            }));
        }
    };
    
    // Check if detector is available
    let detector_available = state.detector.read().is_some();
    if enable && !detector_available {
        return axum::Json(serde_json::json!({
            "error": "YOLO detector not available (RKNN runtime not installed)",
            "detection_enabled": false
        }));
    }
    
    *state.detection_enabled.write() = enable;
    tracing::info!("Detection {}!", if enable { "ENABLED" } else { "DISABLED" });
    
    // Clear detections when disabling
    if !enable {
        *state.last_detections.write() = DetectionResult::default();
    }
    
    axum::Json(serde_json::json!({
        "detection_enabled": enable,
        "success": true
    }))
}

/// Get current detections endpoint
async fn detections_handler(State(state): State<SharedState>) -> axum::Json<serde_json::Value> {
    let detections = state.last_detections.read().clone();
    let enabled = *state.detection_enabled.read();
    
    axum::Json(serde_json::json!({
        "enabled": enabled,
        "detections": detections.detections,
        "count": detections.detections.len()
    }))
}

async fn index_handler(State(state): State<SharedState>) -> Html<String> {
    let current_mode = *state.current_mode.read();
    let detection_enabled = *state.detection_enabled.read();
    let detector_available = state.detector.read().is_some();
    
    let mode_str = match current_mode {
        CaptureMode::Grayscale => "grayscale",
        CaptureMode::Color => "color",
    };
    
    let (detect_checked, detect_status, detect_status_class) = if !detector_available {
        ("disabled", "unavailable", "")
    } else if detection_enabled {
        ("checked", "active", "active")
    } else {
        ("", "off", "")
    };
    
    let html = format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMX415 Live View</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            color: #e0e0e0;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 20px;
        }}
        .mode-tabs {{
            display: flex;
            gap: 0;
            margin-bottom: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .mode-tab {{
            padding: 12px 32px;
            border: none;
            background: transparent;
            color: #888;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1rem;
            font-weight: 500;
        }}
        .mode-tab:hover {{
            color: #ccc;
        }}
        .mode-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        .mode-tab.grayscale.active {{
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            box-shadow: 0 4px 15px rgba(74, 85, 104, 0.4);
        }}
        .mode-tab.color.active {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        }}
        .mode-info {{
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 15px;
            text-align: center;
        }}
        .mode-info.grayscale {{ color: #718096; }}
        .mode-info.color {{ color: #f687b3; }}
        .video-container {{
            position: relative;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                        0 0 40px rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        #stream {{
            display: block;
            max-width: 100%;
            max-height: 75vh;
        }}
        .controls {{
            display: flex;
            gap: 15px;
            margin-top: 25px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        button, .link-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 12px 28px;
            font-size: 1rem;
            border-radius: 30px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }}
        button:hover, .link-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }}
        .stats {{
            margin-top: 25px;
            font-size: 0.85rem;
            color: #666;
            display: flex;
            gap: 30px;
        }}
        .stat {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .stat-value {{
            color: #00d4ff;
            font-weight: 600;
        }}
        .stream-selector {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            background: rgba(255, 255, 255, 0.05);
            padding: 5px;
            border-radius: 25px;
        }}
        .stream-btn {{
            padding: 8px 20px;
            border: none;
            background: transparent;
            color: #888;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9rem;
        }}
        .stream-btn.active {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 1.2rem;
        }}
        .detect-toggle {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            padding: 8px 16px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .detect-toggle label {{
            color: #888;
            font-size: 0.9rem;
            cursor: pointer;
        }}
        .detect-toggle input[type="checkbox"] {{
            width: 40px;
            height: 20px;
            appearance: none;
            background: #333;
            border-radius: 10px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }}
        .detect-toggle input[type="checkbox"]:checked {{
            background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        }}
        .detect-toggle input[type="checkbox"]::before {{
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }}
        .detect-toggle input[type="checkbox"]:checked::before {{
            transform: translateX(20px);
        }}
        .detect-toggle .detect-status {{
            font-size: 0.75rem;
            color: #666;
        }}
        .detect-toggle .detect-status.active {{
            color: #00d4ff;
        }}
        .detection-info {{
            display: none;
            margin-top: 15px;
            padding: 12px 20px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 10px;
            font-size: 0.85rem;
        }}
        .detection-info.visible {{
            display: block;
        }}
        .detection-list {{
            max-height: 150px;
            overflow-y: auto;
            margin-top: 8px;
        }}
        .detection-item {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .detection-class {{
            color: #00d4ff;
            font-weight: 500;
        }}
        .detection-conf {{
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>IMX415 LIVE</h1>
    <p class="subtitle">Rock5C • 4K • 3840×2160</p>
    
    <div class="mode-tabs">
        <button class="mode-tab grayscale {grayscale_active}" onclick="setImageMode('grayscale')">
            ⬛ Grayscale
        </button>
        <button class="mode-tab color {color_active}" onclick="setImageMode('color')">
            🌈 Color
        </button>
    </div>
    
    <p class="mode-info {mode_str}" id="modeInfo">{mode_info}</p>
    
    <div class="stream-selector">
        <button class="stream-btn active" onclick="setStreamMode('mjpeg')">MJPEG</button>
        <button class="stream-btn" onclick="setStreamMode('polling')">Polling</button>
    </div>
    
    <div class="detect-toggle">
        <label for="detectToggle">🎯 YOLO Detection</label>
        <input type="checkbox" id="detectToggle" onchange="toggleDetection(this.checked)" {detect_checked}>
        <span class="detect-status {detect_status_class}" id="detectStatus">{detect_status}</span>
    </div>
    
    <div class="video-container">
        <img id="stream" src="/stream" alt="Live Stream">
    </div>
    
    <div class="controls">
        <button onclick="snapshot()">📷 Snapshot</button>
        <a href="/frame.jpg" target="_blank" class="link-btn">🖼️ Full Frame</a>
        <button onclick="toggleFullscreen()">⛶ Fullscreen</button>
    </div>
    
    <div class="stats">
        <div class="stat">
            <span>Mode:</span>
            <span class="stat-value" id="currentMode">{mode_str}</span>
        </div>
        <div class="stat">
            <span>Frames:</span>
            <span class="stat-value" id="frameCount">0</span>
        </div>
        <div class="stat">
            <span>FPS:</span>
            <span class="stat-value" id="fps">--</span>
        </div>
        <div class="stat">
            <span>Objects:</span>
            <span class="stat-value" id="objectCount">0</span>
        </div>
    </div>
    
    <div class="detection-info" id="detectionInfo">
        <strong>🎯 Detected Objects:</strong>
        <div class="detection-list" id="detectionList"></div>
    </div>
    
    <script>
        let streamMode = 'mjpeg';
        let pollInterval = null;
        let lastCount = 0;
        
        async function setImageMode(mode) {{
            // Update UI immediately
            document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
            document.querySelector('.mode-tab.' + mode).classList.add('active');
            
            const modeInfo = document.getElementById('modeInfo');
            modeInfo.className = 'mode-info ' + mode;
            if (mode === 'grayscale') {{
                modeInfo.textContent = '✓ Artifact-free • Byte-4 extraction with row averaging';
            }} else {{
                modeInfo.textContent = '🧪 Experimental • 10-bit Bayer demosaicing';
            }}
            
            // Send request to server
            try {{
                const res = await fetch('/mode/' + mode);
                const data = await res.json();
                if (data.success) {{
                    document.getElementById('currentMode').textContent = mode;
                    // Refresh stream
                    refreshStream();
                }}
            }} catch (e) {{
                console.error('Failed to set mode:', e);
            }}
        }}
        
        function setStreamMode(mode) {{
            streamMode = mode;
            document.querySelectorAll('.stream-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            refreshStream();
        }}
        
        function refreshStream() {{
            const img = document.getElementById('stream');
            if (pollInterval) {{
                clearInterval(pollInterval);
                pollInterval = null;
            }}
            
            if (streamMode === 'mjpeg') {{
                img.src = '/stream?' + Date.now();
            }} else {{
                img.src = '/frame.jpg?' + Date.now();
                pollInterval = setInterval(() => {{
                    img.src = '/frame.jpg?' + Date.now();
                }}, 100);
            }}
        }}
        
        function snapshot() {{
            const link = document.createElement('a');
            link.href = '/frame.jpg';
            const mode = document.getElementById('currentMode').textContent;
            link.download = 'imx415_' + mode + '_' + new Date().toISOString().slice(0,19).replace(/[:]/g, '-') + '.jpg';
            link.click();
        }}
        
        function toggleFullscreen() {{
            const container = document.querySelector('.video-container');
            if (document.fullscreenElement) {{
                document.exitFullscreen();
            }} else {{
                container.requestFullscreen();
            }}
        }}
        
        async function toggleDetection(enabled) {{
            try {{
                const res = await fetch('/detect/' + (enabled ? 'on' : 'off'));
                const data = await res.json();
                
                const status = document.getElementById('detectStatus');
                const info = document.getElementById('detectionInfo');
                
                if (data.error) {{
                    alert(data.error);
                    document.getElementById('detectToggle').checked = false;
                    status.textContent = 'unavailable';
                    status.className = 'detect-status';
                    return;
                }}
                
                if (data.detection_enabled) {{
                    status.textContent = 'active';
                    status.className = 'detect-status active';
                    info.classList.add('visible');
                }} else {{
                    status.textContent = 'off';
                    status.className = 'detect-status';
                    info.classList.remove('visible');
                    document.getElementById('objectCount').textContent = '0';
                    document.getElementById('detectionList').innerHTML = '';
                }}
            }} catch (e) {{
                console.error('Failed to toggle detection:', e);
            }}
        }}
        
        async function updateDetections() {{
            if (!document.getElementById('detectToggle').checked) return;
            
            try {{
                const res = await fetch('/detections');
                const data = await res.json();
                
                document.getElementById('objectCount').textContent = data.count;
                
                const list = document.getElementById('detectionList');
                if (data.detections && data.detections.length > 0) {{
                    list.innerHTML = data.detections.map(d => 
                        `<div class="detection-item">
                            <span class="detection-class">${{d.class}}</span>
                            <span class="detection-conf">${{(d.confidence * 100).toFixed(1)}}%</span>
                        </div>`
                    ).join('');
                }} else {{
                    list.innerHTML = '<div style="color:#666">No objects detected</div>';
                }}
            }} catch (e) {{}}
        }}
        
        setInterval(async () => {{
            try {{
                const res = await fetch('/status');
                const data = await res.json();
                document.getElementById('frameCount').textContent = data.frame_count;
                document.getElementById('currentMode').textContent = data.mode;
                const fps = data.frame_count - lastCount;
                document.getElementById('fps').textContent = fps;
                lastCount = data.frame_count;
            }} catch (e) {{}}
            
            // Update detections
            updateDetections();
        }}, 1000);
    </script>
</body>
</html>"##,
        grayscale_active = if current_mode == CaptureMode::Grayscale { "active" } else { "" },
        color_active = if current_mode == CaptureMode::Color { "active" } else { "" },
        mode_str = mode_str,
        mode_info = match current_mode {
            CaptureMode::Grayscale => "✓ Artifact-free • Byte-4 extraction with row averaging",
            CaptureMode::Color => "🧪 Experimental • 10-bit Bayer demosaicing",
        },
        detect_checked = detect_checked,
        detect_status = detect_status,
        detect_status_class = detect_status_class,
    );
    
    Html(html)
}

async fn frame_handler(State(state): State<SharedState>) -> Response {
    match &*state.current_frame.read() {
        Some(frame) => {
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "image/jpeg")
                .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
                .body(Body::from(frame.clone()))
                .unwrap()
        }
        None => {
            Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("No frame available"))
                .unwrap()
        }
    }
}

async fn mjpeg_stream_handler(State(state): State<SharedState>) -> Response {
    let boundary = "frame";
    
    let stream = IntervalStream::new(interval(Duration::from_millis(33)))
        .map(move |_| {
            let frame = state.current_frame.read().clone();
            match frame {
                Some(jpeg_data) => {
                    let header = format!(
                        "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                        boundary,
                        jpeg_data.len()
                    );
                    let mut data = header.into_bytes();
                    data.extend_from_slice(&jpeg_data);
                    data.extend_from_slice(b"\r\n");
                    Ok::<_, std::convert::Infallible>(Bytes::from(data))
                }
                None => Ok(Bytes::new()),
            }
        });
    
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/x-mixed-replace; boundary={}", boundary),
        )
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

async fn status_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let frame_count = *state.frame_count.read();
    let has_frame = state.current_frame.read().is_some();
    let mode = *state.current_mode.read();
    let detection_enabled = *state.detection_enabled.read();
    let detection_count = state.last_detections.read().detections.len();
    let detector_available = state.detector.read().is_some();
    
    axum::Json(serde_json::json!({
        "frame_count": frame_count,
        "has_frame": has_frame,
        "resolution": "3840x2160",
        "mode": format!("{:?}", mode).to_lowercase(),
        "detection_enabled": detection_enabled,
        "detection_count": detection_count,
        "detector_available": detector_available
    }))
}
