from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import io
import base64
from PIL import Image
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load YOLOv8 model
try:
    MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt')
    model = YOLO(MODEL_PATH)
    logger.info(f"YOLOv8 model loaded successfully: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

# Configuration
CONFIDENCE_THRESHOLD = float(os.environ.get('YOLO_CONFIDENCE', '0.25'))
IOU_THRESHOLD = float(os.environ.get('YOLO_IOU', '0.45'))

@app.route('/')
def home():
    return jsonify({
        "service": "YOLOv8 Object Detection API",
        "status": "Running" if model else "Error - Model not loaded",
        "model": MODEL_PATH if model else None,
        "version": "1.0",
        "endpoints": {
            "/detect": "POST - Detect objects in image",
            "/health": "GET - Health check",
            "/info": "GET - Model information"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/info')
def model_info():
    """Get model information"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        return jsonify({
            "model_path": MODEL_PATH,
            "model_type": "YOLOv8",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "classes": model.names,
            "class_count": len(model.names),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Main detection endpoint"""
    if not model:
        return jsonify({"error": "YOLO model not loaded"}), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        height, width = image.shape[:2]
        logger.info(f"Processing image: {width}x{height}")
        
        # Run YOLO detection
        results = model(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detection = {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                    
                    detections.append(detection)
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Detected {len(detections)} objects")
        
        response = {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": {"width": width, "height": height},
            "model_info": {
                "model": MODEL_PATH,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/detect-base64', methods=['POST'])
def detect_objects_base64():
    """Detection endpoint for base64 encoded images"""
    if not model:
        return jsonify({"error": "YOLO model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image data provided"}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({"error": f"Invalid base64 data: {str(e)}"}), 400
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        height, width = image.shape[:2]
        logger.info(f"Processing base64 image: {width}x{height}")
        
        # Run YOLO detection (same logic as above)
        results = model(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detection = {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                    
                    detections.append(detection)
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        response = {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": {"width": width, "height": height},
            "model_info": {
                "model": MODEL_PATH,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    if not model:
        logger.error("Cannot start server - YOLO model failed to load")
        exit(1)
    
    port = int(os.environ.get('YOLO_PORT', 5001))
    logger.info(f"Starting YOLOv8 Detection Server on port {port}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"IoU threshold: {IOU_THRESHOLD}")
    
    app.run(host='0.0.0.0', port=port, debug=False)