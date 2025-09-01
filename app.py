from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
from fabric_defect_detector import SimpleFabricDetector
import tempfile
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global detector instance
detector = None

def initialize_detector():
    """Initialize the fabric defect detector"""
    global detector
    model_path = "./models/best.pt"
    if os.path.exists(model_path):
        try:
            detector = SimpleFabricDetector(model_path, confidence_threshold=0.4)
            print("Detector initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return False
    else:
        print(f"Model file not found: {model_path}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_defects():
    """Handle image upload and defect detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        # Read and process image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Run detection
        result_frame, detections = detector.detect(image)
        
        # Convert result to base64 for web display
        _, buffer = cv2.imencode('.jpg', result_frame)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare detection results
        detection_results = []
        defect_counts = {}
        
        for det in detections:
            detection_results.append({
                'class': det['class'],
                'confidence': float(det['confidence']),
                'bbox': det['bbox']
            })
            
            # Count defects by type
            defect_type = det['class'].lower()
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        return jsonify({
            'success': True,
            'result_image': result_base64,
            'detections': detection_results,
            'total_defects': len(detections),
            'defect_counts': defect_counts,
            'confidence_threshold': detector.confidence_threshold
        })
        
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/camera')
def camera():
    """Camera detection page"""
    return render_template('camera.html')

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    """Update confidence threshold"""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    try:
        data = request.get_json()
        new_confidence = float(data.get('confidence', 0.4))
        
        # Clamp confidence between 0.05 and 0.95
        new_confidence = max(0.05, min(0.95, new_confidence))
        
        detector.confidence_threshold = new_confidence
        detector.model.conf = new_confidence
        
        return jsonify({
            'success': True,
            'confidence': new_confidence
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to update confidence: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_initialized': detector is not None,
        'model_path': "./yolov8-fabric-defect-detection/best.pt"
    })

if __name__ == '__main__':
    # Initialize detector on startup
    if initialize_detector():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize detector. Please check your model file.")
