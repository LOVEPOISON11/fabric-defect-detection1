#!/usr/bin/env python3
"""
Simple Fabric Defect Detection using YOLOv8

Usage:
    python fabric_defect_detector.py --mode camera --model path/to/best.pt
    python fabric_defect_detector.py --mode image --image path/to/image.jpg --model path/to/best.pt
"""

import os
import cv2
import torch
import argparse
import numpy as np
import time # Added for FPS calculation

class SimpleFabricDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_path)

    def _load_model(self, model_path):
        print(f"Loading model from {model_path} on {self.device}")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False)
        self.model.to(self.device)
        self.model.eval()
        self.model.conf = self.confidence_threshold
        self.model.iou = 0.45
        print("Model loaded successfully")

    def detect(self, frame):
        """Run inference on a frame and return results with bounding boxes drawn"""
        try:
            with torch.no_grad():
                results = self.model(frame, size=640)
            
            detections = []
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det
                    conf = float(conf)
                    if conf < self.confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                    class_id = int(cls.item())
                    class_name = self.model.names.get(class_id, f"obj_{class_id}")
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2), 
                        'confidence': conf, 
                        'class': class_name
                    })
                    


            # Draw detections on frame
            draw_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                class_name = det['class']
                
                # Color based on confidence
                color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(draw_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return draw_frame, detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []



def camera_detection(model_path, confidence=0.4, camera_src=0):
    """Run detection on camera feed"""
    detector = SimpleFabricDetector(model_path, confidence)
    cap = cv2.VideoCapture(camera_src)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit, '+' to increase confidence, '-' to decrease confidence")
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                
            # Run detection
            result_frame, detections = detector.detect(frame)
            
            # Add FPS and defect count overlay
            cv2.putText(result_frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Current Defects: {len(detections)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Count defects by type
            defect_counts = {}
            for det in detections:
                defect_type = det['class'].lower()
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
            
            # Display individual defect counts
            y_pos = 110
            for defect_type, count in defect_counts.items():
                cv2.putText(result_frame, f"{defect_type} count: {count}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 30
            
            cv2.putText(result_frame, f"Conf: {detector.confidence_threshold:.2f}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display
            cv2.imshow("Fabric Defect Detection", result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                detector.confidence_threshold = min(0.95, detector.confidence_threshold + 0.05)
                detector.model.conf = detector.confidence_threshold
                print(f"Confidence: {detector.confidence_threshold:.2f}")
            elif key == ord('-'):
                detector.confidence_threshold = max(0.05, detector.confidence_threshold - 0.05)
                detector.model.conf = detector.confidence_threshold
                print(f"Confidence: {detector.confidence_threshold:.2f}")

                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def image_detection(image_path, model_path, confidence=0.4, output_path=None):
    """Run detection on single image and display in grid format"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
        
    detector = SimpleFabricDetector(model_path, confidence)
    result_frame, detections = detector.detect(image)
    
    print(f"Found {len(detections)} defects")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']} (confidence: {det['confidence']:.2f})")
    
    # Create a grid layout similar to the defect detection result
    # Get image dimensions
    h, w = result_frame.shape[:2]
    
    # Create a 2x2 grid (4 quadrants)
    grid_h, grid_w = h * 2, w * 2
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Fill each quadrant with the result image
    # Top-left
    grid_image[0:h, 0:w] = result_frame
    # Top-right  
    grid_image[0:h, w:2*w] = result_frame
    # Bottom-left
    grid_image[h:2*h, 0:w] = result_frame
    # Bottom-right
    grid_image[h:2*h, w:2*w] = result_frame
    
    # Add grid lines
    cv2.line(grid_image, (w, 0), (w, grid_h), (128, 128, 128), 2)
    cv2.line(grid_image, (0, h), (grid_w, h), (128, 128, 128), 2)
    
    # Add quadrant labels
    cv2.putText(grid_image, "Sample 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(grid_image, "Sample 2", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(grid_image, "Sample 3", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(grid_image, "Sample 4", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add summary information
    summary_text = f"Total Defects: {len(detections)} | Confidence: {confidence:.2f}"
    cv2.putText(grid_image, summary_text, (10, grid_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, grid_image)
        print(f"Grid result saved to: {output_path}")
    else:
        # Display the grid result
        cv2.namedWindow("Fabric Defect Detection - Grid View", cv2.WINDOW_NORMAL)
        cv2.imshow("Fabric Defect Detection - Grid View", grid_image)
        print("Press any key to close the grid view")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Simple Fabric Defect Detection")
    parser.add_argument("--mode", choices=["camera", "image"], default="camera", 
                       help="Detection mode")
    parser.add_argument("--image", help="Path to image file (for image mode)")
    parser.add_argument("--model", "-m", default="./yolov8-fabric-defect-detection/best.pt", 
                       help="Path to YOLOv8 model (.pt) - defaults to ./yolov8-fabric-defect-detection/best.pt")
    parser.add_argument("--confidence", "-c", type=float, default=0.4, 
                       help="Confidence threshold (0.0-1.0) - defaults to 0.4")
    parser.add_argument("--output", "-o", help="Output path for image mode")
    parser.add_argument("--src", type=int, default=0, help="Camera source index")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please provide a valid path to your YOLOv8 model file (.pt)")
        print("Example: python fabric_defect_detector.py --mode camera --model ./path/to/your/model.pt")
        return
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA available - using GPU")
    else:
        print("CUDA not available - using CPU")
    
    try:
        if args.mode == "camera":
            camera_detection(args.model, args.confidence, args.src)
        elif args.mode == "image":
            if not args.image:
                print("Error: --image is required for image mode")
                print("Example: python fabric_defect_detector.py --mode image --image ./test.jpg --model ./best.pt")
                return
            image_detection(args.image, args.model, args.confidence, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
