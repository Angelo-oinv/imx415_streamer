#!/usr/bin/env python3
"""
YOLO Object Detection Service for IMX415 Streamer
Uses RKNN-Lite to run YOLOv5s on Rock 5C NPU

Runs as a subprocess, communicates via stdin/stdout:
- Input: Raw grayscale JPEG data (length prefix)
- Output: JSON detection results
"""

import sys
import struct
import json
import numpy as np
import cv2
from io import BytesIO

# RKNN-Lite imports
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("ERROR: rknnlite not installed", file=sys.stderr)
    sys.exit(1)

# Model configuration
MODEL_PATH = "/home/angelo/imx415_streamer/models/yolov5s-640-640.rknn"
LABELS_PATH = "/home/angelo/imx415_streamer/models/coco_80_labels_list.txt"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# YOLOv5 anchors for 640x640
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],      # P3/8
    [[30, 61], [62, 45], [59, 119]],     # P4/16
    [[116, 90], [156, 198], [373, 326]]  # P5/32
]
STRIDES = [8, 16, 32]


def load_labels(path):
    """Load COCO class labels"""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process_yolo_output(outputs, img_w, img_h):
    """
    Process YOLOv5 RKNN output to bounding boxes
    Output format: (1, 255, H, W) where 255 = 3 anchors * 85 (4 + 1 + 80)
    """
    boxes = []
    scores = []
    class_ids = []
    
    # Scale factors from input to original image
    scale_x = img_w / INPUT_SIZE
    scale_y = img_h / INPUT_SIZE
    
    for idx, output in enumerate(outputs):
        stride = STRIDES[idx]
        anchor = ANCHORS[idx]
        
        output = np.array(output)
        
        # Output shape: (1, 255, H, W) - NCHW format
        # 255 = 3 anchors * 85 (x, y, w, h, obj_conf, 80 class scores)
        batch, channels, grid_h, grid_w = output.shape
        num_anchors = 3
        num_outputs = channels // num_anchors  # 85
        
        # Reshape to (1, 3, 85, H, W) then transpose to (1, 3, H, W, 85)
        output = output.reshape(batch, num_anchors, num_outputs, grid_h, grid_w)
        output = output.transpose(0, 1, 3, 4, 2)  # (1, 3, H, W, 85)
        
        for a in range(num_anchors):
            for gy in range(grid_h):
                for gx in range(grid_w):
                    pred = output[0, a, gy, gx]
                    
                    # Object confidence (already sigmoid from model or need sigmoid)
                    obj_conf = pred[4]
                    if obj_conf < CONF_THRESHOLD:
                        continue
                    
                    # Class probabilities
                    class_probs = pred[5:]
                    class_id = np.argmax(class_probs)
                    class_conf = class_probs[class_id]
                    
                    # Final confidence
                    confidence = obj_conf * class_conf
                    if confidence < CONF_THRESHOLD:
                        continue
                    
                    # Bounding box - YOLOv5 outputs are already decoded in some RKNN exports
                    # Check if values are in 0-1 range (normalized) or grid-relative
                    bx = pred[0]
                    by = pred[1]
                    bw = pred[2]
                    bh = pred[3]
                    
                    # If values seem to be grid-relative (small values), apply YOLOv5 decode
                    if bx < 10 and by < 10:
                        bx = (bx * 2 - 0.5 + gx) * stride
                        by = (by * 2 - 0.5 + gy) * stride
                        bw = (bw * 2) ** 2 * anchor[a][0]
                        bh = (bh * 2) ** 2 * anchor[a][1]
                    
                    # Convert to corner format and scale to original image
                    x1 = (bx - bw / 2) * scale_x
                    y1 = (by - bh / 2) * scale_y
                    x2 = (bx + bw / 2) * scale_x
                    y2 = (by + bh / 2) * scale_y
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))
    
    return boxes, scores, class_ids


def nms(boxes, scores, class_ids, threshold):
    """Non-maximum suppression"""
    if len(boxes) == 0:
        return [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with rest
        rest = indices[1:]
        
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        
        iou = inter / (area_i + area_rest - inter + 1e-6)
        
        # Keep boxes with low IoU
        indices = rest[iou < threshold]
    
    return boxes[keep].tolist(), scores[keep].tolist(), class_ids[keep].tolist()


class YOLODetector:
    def __init__(self):
        self.rknn = RKNNLite()
        self.labels = load_labels(LABELS_PATH)
        
        # Load model
        print(f"Loading model: {MODEL_PATH}", file=sys.stderr)
        ret = self.rknn.load_rknn(MODEL_PATH)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        # Init runtime
        print("Initializing NPU runtime...", file=sys.stderr)
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime: {ret}")
        
        print("YOLO Detector ready!", file=sys.stderr)
    
    def detect(self, jpeg_data):
        """
        Detect objects in JPEG image
        Returns list of detections
        """
        # Decode JPEG
        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image", "detections": []}
        
        orig_h, orig_w = img.shape[:2]
        
        # Handle grayscale images - convert to 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize to model input size (letterbox)
        img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension (model expects 4D: [batch, height, width, channels])
        img_input = np.expand_dims(img_rgb, axis=0)
        
        # Run inference
        outputs = self.rknn.inference(inputs=[img_input])
        
        # Debug: print output shapes
        print(f"Inference outputs: {len(outputs)} tensors", file=sys.stderr)
        for i, out in enumerate(outputs):
            out_arr = np.array(out)
            print(f"  Output {i}: shape={out_arr.shape}, min={out_arr.min():.2f}, max={out_arr.max():.2f}", file=sys.stderr)
        
        # Process outputs
        try:
            boxes, scores, class_ids = process_yolo_output(outputs, orig_w, orig_h)
            print(f"Post-process: {len(boxes)} raw detections", file=sys.stderr)
        except Exception as e:
            print(f"Post-process ERROR: {e}", file=sys.stderr)
            return {"error": str(e), "detections": []}
        
        # Apply NMS
        boxes, scores, class_ids = nms(boxes, scores, class_ids, NMS_THRESHOLD)
        
        # Format detections
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append({
                "class": self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}",
                "confidence": round(score, 3),
                "bbox": {
                    "x1": int(max(0, box[0])),
                    "y1": int(max(0, box[1])),
                    "x2": int(min(orig_w, box[2])),
                    "y2": int(min(orig_h, box[3]))
                }
            })
        
        return {
            "width": orig_w,
            "height": orig_h,
            "detections": detections
        }
    
    def __del__(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()


def main():
    """
    Main loop - reads JPEG frames from stdin, outputs JSON detections
    Protocol:
    - Input: 4-byte length (little-endian) + JPEG data
    - Output: JSON line (newline terminated)
    """
    detector = YOLODetector()
    
    print("READY", flush=True)  # Signal ready to parent process
    
    while True:
        try:
            # Read length prefix
            length_bytes = sys.stdin.buffer.read(4)
            if len(length_bytes) < 4:
                break
            
            length = struct.unpack('<I', length_bytes)[0]
            
            # Read JPEG data
            jpeg_data = sys.stdin.buffer.read(length)
            if len(jpeg_data) < length:
                break
            
            # Detect
            result = detector.detect(jpeg_data)
            
            # Output JSON
            print(json.dumps(result), flush=True)
            
        except Exception as e:
            print(json.dumps({"error": str(e), "detections": []}), flush=True)


if __name__ == "__main__":
    main()
