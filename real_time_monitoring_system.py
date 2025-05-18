import cv2
import numpy as np
import time
import datetime
import argparse
import os
import threading
import pandas as pd
from collections import defaultdict


class OccupancyMonitor:
    def __init__(self, source=0, output_dir="output", confidence_threshold=0.5, 
                 zone_config=None, idle_threshold=60, break_threshold=300):
        """
        Initialize the occupancy monitoring system.
        
        Args:
            source: Camera index or video file path
            output_dir: Directory to save reports
            confidence_threshold: Detection confidence threshold
            zone_config: Dictionary defining monitoring zones
            idle_threshold: Seconds to consider someone idle (default: 60s)
            break_threshold: Seconds to consider someone on break (default: 5min)
        """
        self.source = source
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.idle_threshold = idle_threshold
        self.break_threshold = break_threshold
        
        # Set up zones (if not provided, use default)
        self.zones = zone_config or {
            "desk_area": [(0, 0, 0.5, 1.0), "productive"],  # Format: [x1, y1, x2, y2], zone_type
            "meeting_area": [(0.5, 0, 1.0, 0.7), "productive"],
            "break_area": [(0.5, 0.7, 1.0, 1.0), "break"]
        }
        
        # Initialize tracking data
        self.person_tracks = {}  # {id: tracking_data}
        self.zone_occupancy = {zone: 0 for zone in self.zones}
        
        # Tracking metrics
        self.productivity_data = defaultdict(lambda: {
            "productive_time": 0,
            "break_time": 0,
            "idle_time": 0,
            "current_zone": None,
            "zone_entry_time": None,
            "last_movement_time": None,
            "is_idle": None,
            "zone_history": []
        })
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load detection model
        print("Loading YOLO model...")
        self.load_model()
        
        # Dashboard update thread
        self.running = True
        self.dashboard_thread = threading.Thread(target=self.update_dashboard)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
    
    def load_model(self):
        """Load YOLOv4 model for person detection"""
        # Load YOLO model
        weights_path = "yolov4.weights"  # Path to pre-trained weights
        config_path = "yolov4.cfg"       # Path to model configuration
        
        # Check if files exist, if not, provide instructions
        if not (os.path.exists(weights_path) and os.path.exists(config_path)):
            print(f"ERROR: YOLOv4 weights or config files not found.")
            print(f"Please download YOLOv4 weights from: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
            print(f"And config from: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
            exit(1)
            
        # Load the network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Set preferred backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load COCO class names
        with open("coco.names", "r") as f:
            self.classes = f.read().strip().split('\n')
        self.person_class_id = self.classes.index("person")
        
        print("Model loaded successfully")
    
    def detect_people(self, frame):
        """
        Detect people in frame using YOLO
        Returns list of bounding boxes and confidence scores
        """
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Pass blob through network
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class with sufficient confidence
                if class_id == self.person_class_id and confidence > self.confidence_threshold:
                    # Scale bbox coordinates to original image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        # Format results
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                confidence = confidences[i]
                detections.append((box, confidence))
        
        return detections
    
    def track_people(self, frame, detections):
        """
        Simple tracking mechanism using IOU between frames
        For production use, consider using a dedicated tracker like SORT or DeepSORT
        """
        height, width = frame.shape[:2]
        current_ids = []
        
        # Match current detections with existing tracks
        for box, confidence in detections:
            x, y, w, h = box
            current_center = (x + w//2, y + h//2)
            matched = False
            best_iou = 0
            best_id = None
            
            # Try to match with existing tracks
            for person_id, track_data in self.person_tracks.items():
                if "last_box" not in track_data:
                    continue
                    
                last_box = track_data["last_box"]
                last_x, last_y, last_w, last_h = last_box
                
                # Calculate IoU between current detection and previous box
                # This is a simple matching method - more sophisticated trackers would do better
                xA = max(x, last_x)
                yA = max(y, last_y)
                xB = min(x + w, last_x + last_w)
                yB = min(y + h, last_y + last_h)
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = w * h
                boxBArea = last_w * last_h
                
                iou = interArea / float(boxAArea + boxBArea - interArea)
                
                if iou > 0.45 and iou > best_iou:  # IoU threshold for match
                    best_iou = iou
                    best_id = person_id
                    matched = True
            
            # If matched with existing track
            if matched:
                self.person_tracks[best_id]["last_box"] = box
                self.person_tracks[best_id]["last_seen"] = time.time()
                self.person_tracks[best_id]["confidence"] = confidence
                current_ids.append(best_id)
                
                # Check for movement
                last_center = self.person_tracks[best_id].get("center", current_center)
                movement = np.sqrt((current_center[0] - last_center[0])**2 + 
                                  (current_center[1] - last_center[1])**2)
                
                if movement > 15:  # Threshold for significant movement (in pixels)
                    self.productivity_data[best_id]["last_movement_time"] = time.time()
                    if self.productivity_data[best_id]["is_idle"]:
                        self.productivity_data[best_id]["is_idle"] = False
                
                self.person_tracks[best_id]["center"] = current_center
            
            # Create new track
            else:
                new_id = max(self.person_tracks.keys(), default=0) + 1
                self.person_tracks[new_id] = {
                    "last_box": box,
                    "last_seen": time.time(),
                    "first_seen": time.time(),
                    "confidence": confidence,
                    "center": current_center
                }
                current_ids.append(new_id)
                
                # Initialize productivity metrics
                self.productivity_data[new_id]["last_movement_time"] = time.time()
                self.productivity_data[new_id]["zone_entry_time"] = time.time()
        
        # Update zone information for all tracked individuals
        for person_id in current_ids:
            box = self.person_tracks[person_id]["last_box"]
            x, y, w, h = box
            frame_h, frame_w = frame.shape[:2]
            
            # Bottom center point of the bounding box (feet position)
            person_point = (x + w//2, y + h)
            
            # Normalized coordinates (0-1 range)
            norm_x = person_point[0] / frame_w
            norm_y = person_point[1] / frame_h
            
            # Determine which zone the person is in
            current_zone = None
            for zone_name, (zone_coords, zone_type) in self.zones.items():
                x1, y1, x2, y2 = zone_coords
                if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
                    current_zone = zone_name
                    break
                    
            # Handle zone transitions
            previous_zone = self.productivity_data[person_id]["current_zone"]
            if current_zone != previous_zone:
                # Record time spent in previous zone if applicable
                if previous_zone is not None:
                    zone_entry_time = self.productivity_data[person_id]["zone_entry_time"]
                    time_in_zone = time.time() - zone_entry_time
                    zone_type = self.zones[previous_zone][1]
                    
                    # Update metrics based on zone type
                    if zone_type == "productive":
                        self.productivity_data[person_id]["productive_time"] += time_in_zone
                    elif zone_type == "break":
                        self.productivity_data[person_id]["break_time"] += time_in_zone
                    
                    # Record zone history
                    self.productivity_data[person_id]["zone_history"].append({
                        "zone": previous_zone,
                        "entry_time": datetime.datetime.fromtimestamp(zone_entry_time).strftime('%H:%M:%S'),
                        "exit_time": datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S'),
                        "duration": time_in_zone
                    })
                
                # Update current zone information
                self.productivity_data[person_id]["current_zone"] = current_zone
                self.productivity_data[person_id]["zone_entry_time"] = time.time()
            
            # Check for idle state
            last_movement = self.productivity_data[person_id]["last_movement_time"]
            idle_duration = time.time() - last_movement
            
            if idle_duration > self.idle_threshold and not self.productivity_data[person_id]["is_idle"]:
                self.productivity_data[person_id]["is_idle"] = True
            
            # Accumulate idle time if applicable
            if self.productivity_data[person_id]["is_idle"]:
                # Only count as idle time if in productive zone
                if current_zone and self.zones[current_zone][1] == "productive":
                    self.productivity_data[person_id]["idle_time"] += 1  # Add time increment (1 second if called each frame)
        
        # Remove old tracks
        current_time = time.time()
        ids_to_remove = []
        for person_id, track_data in self.person_tracks.items():
            if current_time - track_data["last_seen"] > 5:  # Track expiration time (seconds)
                ids_to_remove.append(person_id)
                
                # Finalize metrics for person leaving
                if self.productivity_data[person_id]["current_zone"]:
                    zone = self.productivity_data[person_id]["current_zone"]
                    zone_entry_time = self.productivity_data[person_id]["zone_entry_time"]
                    time_in_zone = time.time() - zone_entry_time
                    zone_type = self.zones[zone][1]
                    
                    # Update metrics based on zone type
                    if zone_type == "productive":
                        self.productivity_data[person_id]["productive_time"] += time_in_zone
                    elif zone_type == "break":
                        self.productivity_data[person_id]["break_time"] += time_in_zone
                    
                    # Record final zone history
                    self.productivity_data[person_id]["zone_history"].append({
                        "zone": zone,
                        "entry_time": datetime.datetime.fromtimestamp(zone_entry_time).strftime('%H:%M:%S'),
                        "exit_time": datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S'),
                        "duration": time_in_zone
                    })
        
        for person_id in ids_to_remove:
            del self.person_tracks[person_id]
    
    def draw_visualizations(self, frame):
        """Draw bounding boxes, zones, and status information on frame"""
        height, width = frame.shape[:2]
        
        # Draw zones
        for zone_name, (coords, zone_type) in self.zones.items():
            x1, y1, x2, y2 = coords
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            if zone_type == "productive":
                color = (0, 255, 0)  # Green for productive
            elif zone_type == "break":
                color = (0, 165, 255)  # Orange for break
            else:
                color = (128, 128, 128)  # Gray for other
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{zone_name}", (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw person bounding boxes and status
        for person_id, track_data in self.person_tracks.items():
            if "last_box" not in track_data:
                continue
                
            x, y, w, h = track_data["last_box"]
            
            # Color code based on state
            if self.productivity_data[person_id]["is_idle"]:
                color = (0, 0, 255)  # Red for idle
                status = "IDLE"
            else:
                zone = self.productivity_data[person_id]["current_zone"]
                if zone and self.zones[zone][1] == "productive":
                    color = (0, 255, 0)  # Green for productive
                    status = "ACTIVE"
                else:
                    color = (0, 165, 255)  # Orange for break
                    status = "BREAK"
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID: {person_id} ({status})", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw productivity stats
            if self.productivity_data[person_id]["current_zone"]:
                zone = self.productivity_data[person_id]["current_zone"]
                productive_time = self.productivity_data[person_id]["productive_time"]
                break_time = self.productivity_data[person_id]["break_time"]
                idle_time = self.productivity_data[person_id]["idle_time"]
                
                cv2.putText(frame, f"Zone: {zone}", (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Productive: {self.format_time(productive_time)}", (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Break: {self.format_time(break_time)}", (x, y + h + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Idle: {self.format_time(idle_time)}", (x, y + h + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw overall statistics
        total_people = len(self.person_tracks)
        active_count = sum(1 for pid in self.person_tracks if not self.productivity_data[pid]["is_idle"])
        idle_count = total_people - active_count
        
        cv2.putText(frame, f"Total People: {total_people}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Active: {active_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Idle: {idle_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 230, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        return str(datetime.timedelta(seconds=int(seconds)))
    
    def update_dashboard(self):
        """Update dashboard and save reports periodically"""
        report_interval = 300  # Generate report every 5 minutes
        last_report_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Generate report every X seconds
            if current_time - last_report_time >= report_interval:
                self.generate_report()
                last_report_time = current_time
            
            time.sleep(1)
    
    def generate_report(self):
        """Generate CSV reports with productivity metrics"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Individual report
        individual_data = []
        for person_id, metrics in self.productivity_data.items():
            if person_id not in self.person_tracks:
                continue  # Skip if no longer being tracked
                
            first_seen = datetime.datetime.fromtimestamp(
                self.person_tracks[person_id]["first_seen"]).strftime('%H:%M:%S')
            
            productive_time = metrics["productive_time"]
            break_time = metrics["break_time"]
            idle_time = metrics["idle_time"]
            total_time = productive_time + break_time
            
            if total_time > 0:
                productivity_ratio = (productive_time - idle_time) / total_time * 100
            else:
                productivity_ratio = 0
                
            individual_data.append({
                "Person ID": person_id,
                "First Detected": first_seen,
                "Total Time (s)": total_time,
                "Productive Time (s)": productive_time,
                "Break Time (s)": break_time,
                "Idle Time (s)": idle_time,
                "Productivity Ratio (%)": productivity_ratio,
                "Current Zone": metrics["current_zone"] or "Unknown"
            })
        
        if individual_data:
            df = pd.DataFrame(individual_data)
            df.to_csv(f"{self.output_dir}/individual_report_{timestamp}.csv", index=False)
            print(f"Generated individual report: individual_report_{timestamp}.csv")
        
        # Zone history report
        zone_history = []
        for person_id, metrics in self.productivity_data.items():
            for entry in metrics["zone_history"]:
                zone_history.append({
                    "Person ID": person_id,
                    "Zone": entry["zone"],
                    "Entry Time": entry["entry_time"],
                    "Exit Time": entry["exit_time"],
                    "Duration (s)": entry["duration"]
                })
        
        if zone_history:
            df = pd.DataFrame(zone_history)
            df.to_csv(f"{self.output_dir}/zone_history_{timestamp}.csv", index=False)
            print(f"Generated zone history: zone_history_{timestamp}.csv")
    
    def run(self):
        """Main loop to process video feed"""
        # Open video source
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return
        
        print(f"Starting monitoring with source {self.source}")
        print("Press 'q' to quit")
        
        frame_count = 0
        process_every_n_frames = 2  # Process every nth frame for efficiency
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                frame_count += 1
                
                # Process every nth frame for efficiency
                if frame_count % process_every_n_frames == 0:
                    # Detect people
                    detections = self.detect_people(frame)
                    
                    # Track people and update metrics
                    self.track_people(frame, detections)
                
                # Always draw visualizations
                vis_frame = self.draw_visualizations(frame)
                
                # Display frame
                cv2.imshow("Occupancy Monitoring", vis_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        finally:
            # Clean up
            self.running = False
            if self.dashboard_thread.is_alive():
                self.dashboard_thread.join(timeout=1)
            
            # Generate final report
            self.generate_report()
            
            cap.release()
            cv2.destroyAllWindows()
            print("Monitoring ended")

def define_custom_zones():
    """
    Interactive function to define custom monitoring zones
    Returns a dictionary with zone definitions
    """
    print("\n--- Custom Zone Definition ---")
    print("You'll define zones by specifying coordinates in normalized format (0-1)")
    print("Format: x1 y1 x2 y2 (top-left and bottom-right corners)")
    
    zones = {}
    zone_count = 1
    
    while True:
        zone_name = input(f"\nEnter zone {zone_count} name (or 'done' to finish): ")
        if zone_name.lower() == 'done':
            break
            
        try:
            coords_input = input(f"Enter coordinates for {zone_name} (x1 y1 x2 y2, values between 0-1): ")
            x1, y1, x2, y2 = map(float, coords_input.split())
            
            # Validate coordinates
            if not (0 <= x1 < x2 <= 1) or not (0 <= y1 < y2 <= 1):
                print("Invalid coordinates. Must be: 0 ≤ x1 < x2 ≤ 1 and 0 ≤ y1 < y2 ≤ 1")
                continue
                
            zone_type = input(f"Zone type for {zone_name} (productive/break): ").lower()
            if zone_type not in ['productive', 'break']:
                print("Invalid zone type. Using 'productive' as default.")
                zone_type = 'productive'
                
            zones[zone_name] = [(x1, y1, x2, y2), zone_type]
            zone_count += 1
            
        except ValueError:
            print("Invalid input format. Please try again.")
    
    if not zones:
        print("No custom zones defined. Using default zones.")
        return None
        
    return zones

def main():
    """Main function to start the monitoring system"""
    parser = argparse.ArgumentParser(description='Real-time Occupancy Monitoring System')
    parser.add_argument('--source', type=str, default='0', 
                        help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--output', type=str, default='output', 
                        help='Output directory for reports')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Detection confidence threshold')
    parser.add_argument('--idle-threshold', type=int, default=60, 
                        help='Seconds of no movement to consider someone idle')
    parser.add_argument('--break-threshold', type=int, default=300, 
                        help='Seconds to consider someone on break')
    parser.add_argument('--custom-zones', action='store_true',
                        help='Enable interactive custom zone definition')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (camera index)
    if args.source.isdigit():
        args.source = int(args.source)
    
    # Allow custom zone definition if requested
    zone_config = None
    if args.custom_zones:
        zone_config = define_custom_zones()
    
    # Create and run the monitoring system
    monitor = OccupancyMonitor(
        source=args.source,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        zone_config=zone_config,
        idle_threshold=args.idle_threshold,
        break_threshold=args.break_threshold
    )
    
    monitor.run()

if __name__ == "__main__":
    main()