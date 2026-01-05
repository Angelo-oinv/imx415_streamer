//! IMX415 Camera Streamer for Rock5C
//!
//! Captures grayscale frames from IMX415 sensor (bypassing ISP) and
//! streams them to web browsers via MJPEG or single frame endpoints.
//!
//! Usage:
//!     cargo run --release
//!     # Then open http://<rock5c-ip>:8080 in browser

mod capture;

use anyhow::Result;
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use bytes::Bytes;
use capture::FrameCapture;
use parking_lot::RwLock;
use std::{sync::Arc, time::Duration};
use tokio::time::interval;
use tokio_stream::wrappers::IntervalStream;
use futures::StreamExt;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

/// Shared application state
struct AppState {
    /// Most recent captured frame (JPEG encoded)
    current_frame: RwLock<Option<Bytes>>,
    /// Frame capture instance
    capture: RwLock<Option<FrameCapture>>,
    /// Frame counter
    frame_count: RwLock<u64>,
}

impl AppState {
    fn new() -> Self {
        Self {
            current_frame: RwLock::new(None),
            capture: RwLock::new(None),
            frame_count: RwLock::new(0),
        }
    }
}

type SharedState = Arc<AppState>;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("IMX415 Streamer starting...");

    // Initialize capture
    let mut capture = FrameCapture::new()?;
    capture.setup_sensor()?;
    capture.start_streaming()?;
    
    info!("Camera initialized");

    // Create shared state
    let state = Arc::new(AppState::new());
    *state.capture.write() = Some(capture);

    // Start capture task
    let capture_state = state.clone();
    tokio::spawn(async move {
        capture_loop(capture_state).await;
    });

    // Build router
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/frame.jpg", get(frame_handler))
        .route("/stream", get(mjpeg_stream_handler))
        .route("/status", get(status_handler))
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:8080";
    info!("Starting web server on http://{}", addr);
    info!("  - Live view: http://<ip>:8080/");
    info!("  - Single frame: http://<ip>:8080/frame.jpg");
    info!("  - MJPEG stream: http://<ip>:8080/stream");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Background capture loop
async fn capture_loop(state: SharedState) {
    let mut interval = interval(Duration::from_millis(33)); // ~30 FPS target
    
    loop {
        interval.tick().await;
        
        // Capture and encode frame
        let frame_result = {
            let mut capture_guard = state.capture.write();
            if let Some(ref mut capture) = *capture_guard {
                capture.capture_jpeg_frame()
            } else {
                continue;
            }
        };
        
        match frame_result {
            Ok(jpeg_data) => {
                *state.current_frame.write() = Some(Bytes::from(jpeg_data));
                *state.frame_count.write() += 1;
            }
            Err(e) => {
                error!("Capture error: {}", e);
            }
        }
    }
}

/// Index page with live view
async fn index_handler() -> Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMX415 Live View</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            color: #e0e0e0;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 30px;
        }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                        0 0 40px rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        #stream {
            display: block;
            max-width: 100%;
            max-height: 80vh;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 25px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button, .link-btn {
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
        }
        button:hover, .link-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .stats {
            margin-top: 25px;
            font-size: 0.85rem;
            color: #666;
            display: flex;
            gap: 30px;
        }
        .stat {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .stat-value {
            color: #00d4ff;
            font-weight: 600;
        }
        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            background: rgba(255, 255, 255, 0.05);
            padding: 5px;
            border-radius: 25px;
        }
        .mode-btn {
            padding: 8px 20px;
            border: none;
            background: transparent;
            color: #888;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .mode-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
    </style>
</head>
<body>
    <h1>IMX415 LIVE</h1>
    <p class="subtitle">Rock5C • Grayscale • 3840×2160</p>
    
    <div class="mode-selector">
        <button class="mode-btn active" onclick="setMode('mjpeg')">MJPEG Stream</button>
        <button class="mode-btn" onclick="setMode('polling')">Polling Mode</button>
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
            <span>Frames:</span>
            <span class="stat-value" id="frameCount">0</span>
        </div>
        <div class="stat">
            <span>FPS:</span>
            <span class="stat-value" id="fps">--</span>
        </div>
    </div>
    
    <script>
        let mode = 'mjpeg';
        let frameCount = 0;
        let lastCount = 0;
        let pollInterval = null;
        
        function setMode(newMode) {
            mode = newMode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
            const img = document.getElementById('stream');
            
            if (mode === 'mjpeg') {
                if (pollInterval) clearInterval(pollInterval);
                img.src = '/stream';
            } else {
                img.src = '/frame.jpg?' + Date.now();
                pollInterval = setInterval(() => {
                    img.src = '/frame.jpg?' + Date.now();
                    frameCount++;
                }, 100);
            }
        }
        
        function snapshot() {
            const link = document.createElement('a');
            link.href = '/frame.jpg';
            link.download = 'imx415_' + new Date().toISOString().slice(0,19).replace(/[:]/g, '-') + '.jpg';
            link.click();
        }
        
        function toggleFullscreen() {
            const container = document.querySelector('.video-container');
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else {
                container.requestFullscreen();
            }
        }
        
        // Update stats
        setInterval(async () => {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                document.getElementById('frameCount').textContent = data.frame_count;
                const fps = data.frame_count - lastCount;
                document.getElementById('fps').textContent = fps;
                lastCount = data.frame_count;
            } catch (e) {}
        }, 1000);
    </script>
</body>
</html>"#;
    
    Html(html.to_string())
}

/// Single frame endpoint
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

/// MJPEG stream endpoint
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

/// Status endpoint
async fn status_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let frame_count = *state.frame_count.read();
    let has_frame = state.current_frame.read().is_some();
    
    axum::Json(serde_json::json!({
        "frame_count": frame_count,
        "has_frame": has_frame,
        "resolution": "3840x2160",
        "format": "grayscale"
    }))
}
