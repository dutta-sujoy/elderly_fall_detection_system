import cv2
import torch
import numpy as np
import time
from datetime import datetime
import os
import threading
import sys

# Add path to YOLOv5 directory
YOLOV5_PATH = r"C:\Users\KIIT\Desktop\fall_detection_system\yolov5"  # Using raw string
sys.path.append(YOLOV5_PATH)

# Create output directory for fall images
if not os.path.exists('fall_events'):
    os.makedirs('fall_events')

def send_alert(image_path=None):
    """Send alert when a fall is detected"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*50)
    print(f"⚠️ FALL DETECTED at {timestamp} ⚠️")
    print(f"Image saved: {image_path if image_path else 'No image'}")
    print("="*50 + "\n")
    
    with open("fall_alerts.log", "a") as log_file:
        log_file.write(f"{timestamp} - Fall detected. Image: {image_path}\n")
    
    return True

class SimplifiedFallDetector:
    def __init__(self):
        # Load YOLOv5 model for person detection
        print("Loading YOLOv5 model from local installation...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Method 1: Using torch.hub with local source
            self.model = torch.hub.load(YOLOV5_PATH, 'custom', 
                                       path=f"{YOLOV5_PATH}/yolov5s.pt", 
                                       source='local',
                                       device=self.device)
        except Exception as e:
            print(f"Error loading model with torch.hub: {e}")
            try:
                # Method 2: Alternative direct loading
                sys.path.insert(0, YOLOV5_PATH)
                from models.experimental import attempt_load
                self.model = attempt_load(f"{YOLOV5_PATH}/yolov5s.pt", device=self.device)
            except Exception as e2:
                print(f"Error with alternative loading: {e2}")
                print("Failed to load YOLOv5 model")
                self.model = None
        
        if self.model:
            # Set model parameters
            self.model.classes = [0]  # Only detect persons (class 0 in COCO)
        
        # Fall detection parameters
        self.bbox_ratio_threshold = 0.7  # Aspect ratio threshold (width/height)
        self.velocity_threshold = 15  # Sudden movement threshold (pixels/frame)
        self.fall_frames_threshold = 5  # Number of consecutive frames to confirm fall
        self.fall_frames_counter = 0
        self.fall_cooldown = 30  # Frames to wait before detecting another fall
        self.cooldown_counter = 0
        self.fall_detected = False
        
        # Motion tracking
        self.prev_boxes = []
        self.prev_centroids = []
        self.centroid_history = []  # Store last 10 centroids for velocity calculation
        self.history_size = 10
        
        # Bounding box history for size change detection
        self.bbox_size_history = []
        
        print(f"Simplified fall detector initialized on {self.device}")
    
    def detect_persons(self, frame):
        """Detect persons in the frame using YOLOv5"""
        if self.model is None:
            return np.array([])
            
        # Convert BGR to RGB (YOLOv5 expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(img_rgb)
        
        # Extract person detections
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        
        # Filter for persons only (class 0)
        person_detections = detections[detections[:, 5] == 0]
        
        return person_detections
    
    def calculate_velocity(self, current_centroid):
        """Calculate velocity based on centroid history"""
        if not self.centroid_history or not current_centroid:
            return 0
            
        # Calculate displacement between current and oldest position in history
        if len(self.centroid_history) > 5:
            old_centroid = self.centroid_history[0]
            dx = current_centroid[0] - old_centroid[0]
            dy = current_centroid[1] - old_centroid[1]
            displacement = np.sqrt(dx**2 + dy**2)
            # Velocity is displacement over time (frames)
            velocity = displacement / len(self.centroid_history)
            return velocity
        return 0
    
    def detect_bbox_size_change(self, current_box):
        """Detect significant changes in bounding box size (potential fall)"""
        if not self.bbox_size_history or not current_box:
            return False
            
        # Calculate current box size
        width = current_box[2] - current_box[0]
        height = current_box[3] - current_box[1]
        current_size = width * height
        
        # Get oldest box size from history
        if len(self.bbox_size_history) > 5:
            old_box = self.bbox_size_history[0]
            old_width = old_box[2] - old_box[0]
            old_height = old_box[3] - old_box[1]
            old_size = old_width * old_height
            
            # Calculate size change ratio
            size_ratio = current_size / old_size if old_size > 0 else 1.0
            
            # Check if size changed significantly
            return size_ratio > 1.5 or size_ratio < 0.7
        
        return False
    
    def detect_fall(self, frame, detections):
        """Detect falls using bounding box and motion analysis"""
        current_boxes = []
        current_centroids = []
        fall_indicators = {
            "bbox_ratio": False,
            "sudden_movement": False,
            "bbox_size_change": False
        }
        
        # Process YOLO detections
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            
            # Calculate bounding box properties
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # Store bounding box and centroid
            box = [int(x1), int(y1), int(x2), int(y2)]
            centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            current_boxes.append(box)
            current_centroids.append(centroid)
            
            # Check if aspect ratio indicates a fall (wider than tall)
            if aspect_ratio > self.bbox_ratio_threshold and conf > 0.5:
                fall_indicators["bbox_ratio"] = True
                
                # Draw red box for potential fall
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(frame, f"Potential Fall: {aspect_ratio:.2f}", 
                           (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Draw green box for normal posture
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {conf:.2f}", 
                           (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Process movement for velocity calculation
        if current_centroids and current_boxes:
            main_centroid = current_centroids[0]  # Use first person detected
            main_box = current_boxes[0]
            
            # Update centroid history
            self.centroid_history.append(main_centroid)
            if len(self.centroid_history) > self.history_size:
                self.centroid_history.pop(0)
            
            # Update bbox size history
            self.bbox_size_history.append(main_box)
            if len(self.bbox_size_history) > self.history_size:
                self.bbox_size_history.pop(0)
            
            # Calculate velocity
            velocity = self.calculate_velocity(main_centroid)
            
            # Check for sudden movement (potential fall)
            if velocity > self.velocity_threshold:
                fall_indicators["sudden_movement"] = True
                
            # Check for significant bounding box size change
            if self.detect_bbox_size_change(main_box):
                fall_indicators["bbox_size_change"] = True
                
            # Display velocity
            cv2.putText(frame, f"Velocity: {velocity:.1f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Combine fall indicators with weights
        fall_score = (
            (1 if fall_indicators["bbox_ratio"] else 0) * 0.4 +
            (1 if fall_indicators["sudden_movement"] else 0) * 0.4 +
            (1 if fall_indicators["bbox_size_change"] else 0) * 0.2
        )
        
        # Display fall indicators
        cv2.putText(frame, f"Fall Score: {fall_score:.2f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Detect sitting vs. falling
        # If person is in bottom half of frame with high width/height ratio
        # but without sudden movement, likely sitting not falling
        if (current_centroids and 
            current_centroids[0][1] > frame.shape[0] * 0.5 and  # Person in bottom half
            fall_indicators["bbox_ratio"] and  # Wide aspect ratio
            not fall_indicators["sudden_movement"]):  # No sudden movement
            
            cv2.putText(frame, "Likely sitting, not falling", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            fall_score *= 0.5  # Reduce fall score for likely sitting
        
        # Update fall detection state
        fall_indicator = fall_score > 0.5  # Threshold for combined indicators
        
        if fall_indicator:
            self.fall_frames_counter += 1
        else:
            self.fall_frames_counter = 0
        
        # Check if fall is confirmed (multiple consecutive frames)
        if self.fall_frames_counter >= self.fall_frames_threshold and not self.fall_detected:
            self.fall_detected = True
            self.cooldown_counter = 0
            
            # Save image of the fall
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fall_img_path = f"fall_events/fall_{timestamp}.jpg"
            cv2.imwrite(fall_img_path, frame)
            
            # Trigger alert in a separate thread to avoid blocking
            alert_thread = threading.Thread(
                target=send_alert, 
                args=(fall_img_path,)
            )
            alert_thread.start()
            
            return True
        
        # Handle cooldown after fall detection
        if self.fall_detected:
            self.cooldown_counter += 1
            if self.cooldown_counter > self.fall_cooldown:
                self.fall_detected = False
        
        # Update previous data
        self.prev_boxes = current_boxes
        self.prev_centroids = current_centroids
        
        return False

def main():
    # Initialize fall detector
    detector = SimplifiedFallDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Simplified fall detection system started. Press 'q' to quit.")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Increment frame counter
        frame_count += 1
        
        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time
        
        # Detect persons in the frame
        detections = detector.detect_persons(frame)
        
        # Check for falls
        fall_detected = detector.detect_fall(frame, detections)
        
        # Display fall alert on screen if detected
        if detector.fall_detected:
            cv2.putText(frame, "FALL DETECTED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Draw red border around frame
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        
        # Display FPS and detection count
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {len(detections)}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Simplified Fall Detection System", frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
