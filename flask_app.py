from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import os
import json
import argparse
import numpy as np
from real_time_monitoring_system import OccupancyMonitor

app = Flask(__name__)

# Global variables
monitor = None
output_frame = None
lock = threading.Lock()
zone_stats = {}
person_stats = {}

def initialize_monitor(source=0, output_dir="output", confidence=0.5, 
                     idle_threshold=60, break_threshold=300, zone_config=None):
    """Initialize the monitoring system"""
    global monitor
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the monitor
    monitor = OccupancyMonitor(
        source=source,
        output_dir=output_dir,
        confidence_threshold=confidence,
        zone_config=zone_config,
        idle_threshold=idle_threshold,
        break_threshold=break_threshold
    )
    
    # Start processing in a separate thread
    t = threading.Thread(target=monitor_thread)
    t.daemon = True
    t.start()
    
    # Start stats update thread
    t2 = threading.Thread(target=stats_thread)
    t2.daemon = True
    t2.start()
    
    print(f"Monitor initialized with source: {source}")

def monitor_thread():
    """Thread that processes frames from the video source"""
    global output_frame, lock, monitor
    
    # Open video source
    if isinstance(monitor.source, str) and monitor.source.isdigit():
        cap = cv2.VideoCapture(int(monitor.source))
    else:
        cap = cv2.VideoCapture(monitor.source)
        
    if not cap.isOpened():
        print(f"Error: Could not open video source {monitor.source}")
        return
        
    print("Video capture started")
    
    # Process frames
    frame_count = 0
    process_every_n_frames = 2  # Process every nth frame for efficiency
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream, restarting capture...")
            # Try to reopen the camera
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(monitor.source)
            continue
            
        frame_count += 1
        
        # Process every nth frame for efficiency
        if frame_count % process_every_n_frames == 0:
            # Detect people
            detections = monitor.detect_people(frame)
            
            # Track people and update metrics
            monitor.track_people(frame, detections)
        
        # Draw visualizations
        processed_frame = monitor.draw_visualizations(frame)
        
        # Update the output frame
        with lock:
            output_frame = processed_frame.copy()

def stats_thread():
    """Thread that updates statistics for the web interface"""
    global zone_stats, person_stats, monitor
    
    while True:
        # Update zone statistics
        current_zones = {}
        for zone_name in monitor.zones:
            occupancy = 0
            for person_id in monitor.person_tracks:
                if monitor.productivity_data[person_id]["current_zone"] == zone_name:
                    occupancy += 1
            
            zone_type = monitor.zones[zone_name][1]
            current_zones[zone_name] = {
                "occupancy": occupancy,
                "type": zone_type
            }
        
        # Update person statistics
        current_people = {}
        for person_id, metrics in monitor.productivity_data.items():
            if person_id in monitor.person_tracks:  # Only include active tracks
                productive_time = metrics["productive_time"]
                break_time = metrics["break_time"]
                idle_time = metrics["idle_time"]
                
                current_people[person_id] = {
                    "productive_time": format_time(productive_time),
                    "break_time": format_time(break_time),
                    "idle_time": format_time(idle_time),
                    "current_zone": metrics["current_zone"] or "Unknown",
                    "is_idle": metrics["is_idle"]
                }
        
        # Update global variables with thread safety
        with lock:
            zone_stats = current_zones
            person_stats = current_people
            
        time.sleep(1)  # Update every second

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@app.route('/')
def index():
    """Serve the main monitoring page"""
    return render_template('index.html')

def generate_frames():
    """Generate frames for the video feed"""
    global output_frame, lock
    
    while True:
        # Wait until we have a frame
        if output_frame is None:
            time.sleep(0.1)
            continue
            
        # Encode the frame as JPEG
        with lock:
            if output_frame is None:
                continue
                
            # Resize for web streaming (smaller for better performance)
            # Adjust the width as needed
            width = 800
            height = int(output_frame.shape[0] * (width / output_frame.shape[1]))
            small_frame = cv2.resize(output_frame, (width, height))
            
            ret, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
        if not ret:
            continue
            
        # Yield the frame in the format expected by Response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for the video feed"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """API endpoint for current statistics"""
    global zone_stats, person_stats
    
    with lock:
        stats = {
            "zones": zone_stats,
            "people": person_stats,
            "total_count": len(person_stats),
            "active_count": sum(1 for p in person_stats.values() if not p["is_idle"]),
            "idle_count": sum(1 for p in person_stats.values() if p["is_idle"]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    return jsonify(stats)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Occupancy Monitoring Web Interface")
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port for the web server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host for the web server (0.0.0.0 allows external access)')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory for reports')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Detection confidence threshold')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Convert source to int if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create a templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the index.html file
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Occupancy Monitoring System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f5f5f5;
        }
        .video-container {
            margin-bottom: 20px;
            text-align: center;
        }
        
        .video-feed {
            max-width: 100%;
            border: 3px solid #343a40;
            border-radius: 5px;
        }
        .stats-card {
            margin-bottom: 20px;
        }
        .stats-header {
            background-color: #343a40;
            color: white;
        }
        .zone-productive {
            background-color: rgba(40, 167, 69, 0.2);
        }
        .zone-break {
            background-color: rgba(255, 193, 7, 0.2);
        }
        .person-idle {
            background-color: rgba(220, 53, 69, 0.1);
        }
        .dashboard {
            margin-top: 20px;
        }
        .counter-card {
            text-align: center;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            color: white;
        }
        .counter-total {
            background-color: #17a2b8;
        }
        .counter-active {
            background-color: #28a745;
        }
        .counter-idle {
            background-color: #dc3545;
        }
        h5 {
            margin-bottom: 0;
        }
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .counter-card {
                margin-bottom: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mt-3 mb-4">Occupancy Monitoring System</h1>
 
<!-- Control Buttons -->
     <div class="row control-buttons">
            <div class="col-md-12">
                <button id="startBtn" class="btn btn-control btn-start text-white">
                    <i class="bi bi-play-fill"></i> Start Monitoring
                </button>
                <button id="stopBtn" class="btn btn-control btn-stop text-white">
                    <i class="bi bi-stop-fill"></i> Stop Monitoring
                </button>
                <button id="downloadBtn" class="btn btn-control btn-download text-white">
                    <i class="bi bi-download"></i> Download Report
                </button>
            </div>
        </div>

        
        <!-- Dashboard Summary -->
        <div class="row dashboard">
            <div class="col-md-4">
                <div class="counter-card counter-total">
                    <h3 id="total-count">0</h3>
                    <h5>Total People</h5>
                </div>
            </div>
            <div class="col-md-4">
                <div class="counter-card counter-active">
                    <h3 id="active-count">0</h3>
                    <h5>Active</h5>
                </div>
            </div>
            <div class="col-md-4">
                <div class="counter-card counter-idle">
                    <h3 id="idle-count">0</h3>
                    <h5>Idle</h5>
                </div>
            </div>
        </div>
        
        <div class="row">
            <!-- Video Feed -->
            <div class="col-lg-8">
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" class="video-feed" alt="Live Feed">
                    <p class="text-muted mt-2">Last updated: <span id="timestamp">Loading...</span></p>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="col-lg-4">
                <!-- Zone Statistics -->
                <div class="card stats-card">
                    <div class="card-header stats-header">
                        <h4 class="mb-0">Zone Occupancy</h4>
                    </div>
                    <div class="card-body">
                        <div id="zones-container">
                            <p>Loading zone data...</p>
                        </div>
                    </div>
                </div>
                
                <!-- Person Statistics -->
                <div class="card stats-card">
                    <div class="card-header stats-header">
                        <h4 class="mb-0">Person Statistics</h4>
                    </div>
                    <div class="card-body">
                        <div id="people-container">
                            <p>Loading people data...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update statistics regularly
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    // Update summary counters
                    document.getElementById('total-count').textContent = data.total_count;
                    document.getElementById('active-count').textContent = data.active_count;
                    document.getElementById('idle-count').textContent = data.idle_count;
                    document.getElementById('timestamp').textContent = data.timestamp;
                    
                    // Update zone information
                    let zonesHTML = '';
                    for (const [zoneName, zoneData] of Object.entries(data.zones)) {
                        const zoneClass = zoneData.type === 'productive' ? 'zone-productive' : 'zone-break';
                        zonesHTML += `
                            <div class="card mb-2 ${zoneClass}">
                                <div class="card-body py-2">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <h5>${zoneName}</h5>
                                        <span class="badge bg-secondary">${zoneData.occupancy} people</span>
                                    </div>
                                    <small class="text-muted">Type: ${zoneData.type}</small>
                                </div>
                            </div>
                        `;
                    }
                    
                    if (zonesHTML === '') {
                        zonesHTML = '<p>No zones defined</p>';
                    }
                    document.getElementById('zones-container').innerHTML = zonesHTML;
                    
                    // Update people information
                    let peopleHTML = '';
                    for (const [personId, personData] of Object.entries(data.people)) {
                        const personClass = personData.is_idle ? 'person-idle' : '';
                        const statusClass = personData.is_idle ? 'bg-danger' : 'bg-success';
                        const statusText = personData.is_idle ? 'IDLE' : 'ACTIVE';
                        
                        peopleHTML += `
                            <div class="card mb-2 ${personClass}">
                                <div class="card-body py-2">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <h5>Person ID: ${personId}</h5>
                                        <span class="badge ${statusClass}">${statusText}</span>
                                    </div>
                                    <div class="small">
                                        <div>Zone: ${personData.current_zone}</div>
                                        <div>Productive time: ${personData.productive_time}</div>
                                        <div>Break time: ${personData.break_time}</div>
                                        <div>Idle time: ${personData.idle_time}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    if (peopleHTML === '') {
                        peopleHTML = '<p>No people detected</p>';
                    }
                    document.getElementById('people-container').innerHTML = peopleHTML;
                })
                .catch(error => {
                    console.error('Error fetching stats:', error);
                });
        }
        
        // Initial update
        updateStats();
        
        // Update every 2 seconds
        setInterval(updateStats, 2000);
    </script>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    # Write the HTML file
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    # Initialize the monitoring system
    initialize_monitor(
        source=source,
        output_dir=args.output,
        confidence=args.confidence
    )
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)