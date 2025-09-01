import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename

# Import our fabric detector (with fallback for Vercel)
try:
    from fabric_defect_detector import SimpleFabricDetector
    DETECTOR_AVAILABLE = True
    print("Using full AI detector")
except ImportError:
    try:
        from fabric_defect_detector_lite import EnhancedFabricDetector as SimpleFabricDetector
        DETECTOR_AVAILABLE = True
        print("Using lightweight detector for Vercel")
    except ImportError:
        DETECTOR_AVAILABLE = False
        print("Warning: No detector available, running in basic demo mode")

app = Flask(__name__,
           template_folder='../templates',
           static_folder='../static')

# Configuration for Vercel
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Global detector instance
detector = None

def initialize_detector():
    """Initialize the fabric defect detector"""
    global detector
    if not DETECTOR_AVAILABLE:
        return False

    # Try to find the model file
    possible_paths = [
        "../models/best.pt",
        "models/best.pt",
        "/tmp/best.pt"
    ]

    model_path = None
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            model_path = full_path
            break

    if model_path and os.path.exists(model_path):
        try:
            detector = SimpleFabricDetector(model_path, confidence_threshold=0.4)
            print("Detector initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return False
    else:
        print("Model file not found")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/camera')
def camera():
    """Camera detection page"""
    return render_template('camera.html')

@app.route('/detect', methods=['POST'])
def detect_defects():
    """Handle image upload and defect detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Read and process image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Run detection (with fallback for demo mode)
        if detector is not None:
            result_frame, detections = detector.detect(image)
        else:
            # Demo mode fallback
            result_frame = image.copy()
            height, width = image.shape[:2]
            detections = [
                {
                    'class': 'demo_defect',
                    'confidence': 0.75,
                    'bbox': (width//4, height//4, width//2, height//2)
                }
            ]
            # Draw demo box
            x1, y1, x2, y2 = detections[0]['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(result_frame, "DEMO: demo_defect: 0.75",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
            'confidence_threshold': detector.confidence_threshold if detector else 0.4,
            'demo_mode': detector is None
        })

    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_initialized': detector is not None,
        'demo_mode': detector is None,
        'detector_available': DETECTOR_AVAILABLE
    })

# Initialize detector on startup (if possible)
initialize_detector()