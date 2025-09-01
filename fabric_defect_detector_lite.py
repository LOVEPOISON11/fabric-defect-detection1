#!/usr/bin/env python3
"""
Lightweight Fabric Defect Detection for Vercel deployment
This version provides demo functionality without heavy ML dependencies
"""

import cv2
import numpy as np
import random
import time


class SimpleFabricDetector:
    """Lightweight fabric detector for Vercel deployment"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.demo_mode = True  # Always run in demo mode for Vercel
        
        # Demo defect types
        self.defect_types = [
            'hole', 'stain', 'thread_break', 'color_variation', 
            'texture_defect', 'weave_defect', 'dirt', 'oil_spot'
        ]
        
        print(f"Initialized lightweight detector (demo mode) with confidence threshold: {confidence_threshold}")
    
    def detect(self, frame):
        """Run demo detection on a frame and return results with bounding boxes drawn"""
        try:
            height, width = frame.shape[:2]
            
            # Generate random number of defects (0-3)
            num_defects = random.randint(0, 3)
            detections = []
            
            # Generate demo detections
            for i in range(num_defects):
                # Random position and size
                x1 = random.randint(0, width // 2)
                y1 = random.randint(0, height // 2)
                w = random.randint(width // 8, width // 4)
                h = random.randint(height // 8, height // 4)
                x2 = min(x1 + w, width - 1)
                y2 = min(y1 + h, height - 1)
                
                # Random defect type and confidence
                defect_type = random.choice(self.defect_types)
                confidence = random.uniform(self.confidence_threshold, 0.95)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class': defect_type
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
                
                # Add "DEMO" watermark
                cv2.putText(draw_frame, "DEMO MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            return draw_frame, detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []


def demo_detection_with_patterns(image):
    """Enhanced demo detection with pattern-based defects"""
    height, width = image.shape[:2]
    
    # Convert to grayscale for pattern analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple pattern-based "defect" detection
    detections = []
    
    # Look for very dark or very bright regions (simulated defects)
    dark_threshold = 50
    bright_threshold = 200
    
    # Find dark regions
    dark_mask = gray < dark_threshold
    contours_dark, _ = cv2.findContours(dark_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_dark:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 0.8,
                'class': 'dark_spot'
            })
    
    # Find bright regions
    bright_mask = gray > bright_threshold
    contours_bright, _ = cv2.findContours(bright_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_bright:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 0.75,
                'class': 'bright_spot'
            })
    
    # Limit to maximum 5 detections
    detections = detections[:5]
    
    # Draw detections
    result_frame = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class']
        
        color = (0, 255, 255) if class_name == 'dark_spot' else (255, 255, 0)
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(result_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add demo watermark
    cv2.putText(result_frame, "DEMO - Pattern Analysis", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return result_frame, detections


class EnhancedFabricDetector(SimpleFabricDetector):
    """Enhanced version with basic image processing"""
    
    def detect(self, frame):
        """Enhanced detection using basic image processing"""
        try:
            # Use pattern-based detection for more realistic demo
            return demo_detection_with_patterns(frame)
            
        except Exception as e:
            print(f"Enhanced detection error: {e}")
            # Fallback to simple demo
            return super().detect(frame)


def camera_detection_demo():
    """Demo function for camera detection (not used in Vercel)"""
    print("Camera detection demo - not available in Vercel deployment")
    return False


def image_detection_demo(image_path):
    """Demo function for image detection"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
        
    detector = EnhancedFabricDetector(confidence_threshold=0.4)
    result_frame, detections = detector.detect(image)
    
    print(f"Demo detection found {len(detections)} defects")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']} (confidence: {det['confidence']:.2f})")
    
    return result_frame, detections


if __name__ == "__main__":
    print("Lightweight Fabric Defect Detector - Demo Mode")
    print("This version is optimized for Vercel deployment")
    print("For full AI detection, use the complete version with PyTorch")
