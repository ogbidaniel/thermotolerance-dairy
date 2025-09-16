#!/usr/bin/env python3
"""
Thermotolerance Dairy Cattle Behavior Analysis - Real-time Inference

This script provides real-time analysis of cattle behaviors using a webcam or video stream.
Useful for live monitoring of thermal stress behaviors in dairy cattle.

Usage:
    python realtime_inference.py --camera 0
    python realtime_inference.py --stream rtsp://camera_ip:554/stream
    python realtime_inference.py --camera 0 --model custom_model.pt --save_output
"""

import argparse
import cv2
import os
import sys
import json
import numpy as np
from datetime import datetime
import time
from collections import deque
import threading

# Add current script directory and utils to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import the main analyzer from inference.py
from inference import CattleBehaviorAnalyzer


class RealTimeAnalyzer:
    """
    Real-time cattle behavior analyzer for live video streams
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5, device='auto'):
        """
        Initialize real-time analyzer
        
        Args:
            model_path (str): Path to trained model
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device to run inference on
        """
        self.analyzer = CattleBehaviorAnalyzer(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Real-time tracking
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)  # Track FPS over last 30 frames
        self.behavior_history = deque(maxlen=300)  # Track last 10 seconds at 30fps
        self.session_stats = {
            'start_time': datetime.now(),
            'total_frames': 0,
            'total_detections': 0,
            'behavior_counts': {},
            'alerts': []
        }
        
        # Display settings
        self.colors = {
            'brushing': (0, 255, 0),      # Green
            'drinking': (255, 0, 0),      # Blue  
            'headbutting': (0, 0, 255),   # Red
            'grazing': (0, 255, 255),     # Yellow
            'resting': (255, 255, 0),     # Cyan
            'walking': (255, 0, 255)      # Magenta
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'headbutting': 3,   # Alert if >3 detections in 10 seconds
            'excessive_brushing': 10,  # Alert if >10 brushing detections in 10 seconds
        }
        
        # Video recording
        self.video_writer = None
        self.recording = False
        
    def draw_detections(self, frame, detections, frame_info):
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: OpenCV frame
            detections: List of detection dictionaries
            frame_info: Additional frame information
            
        Returns:
            frame: Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            behavior = detection['behavior']
            confidence = detection['confidence']
            
            # Get color for behavior
            color = self.colors.get(behavior, (128, 128, 128))
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{behavior}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_overlay(self, frame, fps, behavior_counts, alerts):
        """
        Draw information overlay on frame
        
        Args:
            frame: OpenCV frame
            fps: Current FPS
            behavior_counts: Current behavior counts
            alerts: List of current alerts
            
        Returns:
            frame: Frame with overlay
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Header
        cv2.putText(frame, "Cattle Behavior Monitor", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FPS and frame info
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {self.frame_count}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Behavior counts
        y_pos = 80
        for behavior, count in behavior_counts.items():
            color = self.colors.get(behavior, (255, 255, 255))
            cv2.putText(frame, f"{behavior}: {count}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 20
        
        # Alerts
        if alerts:
            alert_y = h - 50
            for alert in alerts[-3:]:  # Show last 3 alerts
                cv2.putText(frame, f"ALERT: {alert}", (20, alert_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                alert_y -= 25
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 60, 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def check_alerts(self, recent_behaviors):
        """
        Check for behavior-based alerts
        
        Args:
            recent_behaviors: List of recent behavior detections
            
        Returns:
            list: New alerts to display
        """
        alerts = []
        
        # Count behaviors in recent history
        behavior_counts = {}
        for behavior in recent_behaviors:
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        # Check thresholds
        if behavior_counts.get('headbutting', 0) >= self.alert_thresholds['headbutting']:
            alerts.append("High aggression detected!")
            
        if behavior_counts.get('brushing', 0) >= self.alert_thresholds['excessive_brushing']:
            alerts.append("Excessive grooming - possible heat stress!")
        
        return alerts
    
    def start_recording(self, output_path, fps, frame_size):
        """Start video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.recording = True
        print(f"Started recording to: {output_path}")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Recording stopped")
    
    def run_realtime_analysis(self, source, save_output=False, output_dir='realtime_output'):
        """
        Run real-time analysis on video source
        
        Args:
            source: Video source (camera index, file path, or stream URL)
            save_output: Whether to save annotated video
            output_dir: Directory for output files
            
        Returns:
            dict: Session analysis results
        """
        print(f"Starting real-time analysis...")
        print(f"Source: {source}")
        print(f"Demo mode: {self.analyzer.demo_mode}")
        
        # Open video source
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
            source_type = "camera"
        elif source.startswith(('rtsp://', 'http://', 'https://')):
            cap = cv2.VideoCapture(source)
            source_type = "stream"
        else:
            cap = cv2.VideoCapture(source)
            source_type = "file"
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
        
        # Setup recording if requested
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"realtime_analysis_{timestamp}.mp4")
            self.start_recording(output_path, fps, (width, height))
        
        # Analysis loop
        print("\nReal-time analysis started. Press 'q' to quit, 'r' to toggle recording, 's' to save stats")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if source_type == "file":
                        print("End of video file reached")
                        break
                    else:
                        print("Failed to read frame, retrying...")
                        continue
                
                # Track timing for FPS calculation
                frame_start_time = time.time()
                
                # Analyze frame
                frame_result = self.analyzer.analyze_frame(frame, self.frame_count)
                detections = frame_result['detections']
                
                # Update statistics
                self.frame_count += 1
                self.session_stats['total_frames'] += 1
                self.session_stats['total_detections'] += len(detections)
                
                # Track behavior history
                for detection in detections:
                    behavior = detection['behavior']
                    self.behavior_history.append(behavior)
                    self.session_stats['behavior_counts'][behavior] = \
                        self.session_stats['behavior_counts'].get(behavior, 0) + 1
                
                # Check for alerts
                recent_behaviors = list(self.behavior_history)
                new_alerts = self.check_alerts(recent_behaviors)
                for alert in new_alerts:
                    if alert not in [a['message'] for a in self.session_stats['alerts'][-5:]]:
                        self.session_stats['alerts'].append({
                            'message': alert,
                            'timestamp': datetime.now().isoformat(),
                            'frame': self.frame_count
                        })
                
                # Calculate FPS
                frame_time = time.time() - frame_start_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_counter.append(current_fps)
                avg_fps = sum(self.fps_counter) / len(self.fps_counter)
                
                # Draw annotations
                annotated_frame = self.draw_detections(frame, detections, frame_result)
                annotated_frame = self.draw_overlay(
                    annotated_frame, 
                    avg_fps, 
                    self.session_stats['behavior_counts'],
                    [a['message'] for a in self.session_stats['alerts'][-3:]]
                )
                
                # Record frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Cattle Behavior Analysis', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        if not save_output:
                            os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(output_dir, f"realtime_recording_{timestamp}.mp4")
                        self.start_recording(output_path, fps, (width, height))
                elif key == ord('s'):
                    self._save_session_stats(output_dir)
                    print("Session stats saved")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            if self.recording:
                self.stop_recording()
        
        # Finalize session stats
        self.session_stats['end_time'] = datetime.now()
        duration = (self.session_stats['end_time'] - self.session_stats['start_time']).total_seconds()
        self.session_stats['duration_seconds'] = duration
        self.session_stats['average_fps'] = self.frame_count / duration if duration > 0 else 0
        
        # Print session summary
        self._print_session_summary()
        
        return self.session_stats
    
    def _save_session_stats(self, output_dir):
        """Save current session statistics"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = os.path.join(output_dir, f"session_stats_{timestamp}.json")
        
        # Prepare serializable stats
        serializable_stats = self.session_stats.copy()
        serializable_stats['start_time'] = self.session_stats['start_time'].isoformat()
        if 'end_time' in serializable_stats:
            serializable_stats['end_time'] = serializable_stats['end_time'].isoformat()
        
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Session stats saved to: {stats_path}")
    
    def _print_session_summary(self):
        """Print session analysis summary"""
        print("\n" + "="*50)
        print("REAL-TIME SESSION SUMMARY")
        print("="*50)
        
        duration = self.session_stats.get('duration_seconds', 0)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total frames: {self.session_stats['total_frames']}")
        print(f"Total detections: {self.session_stats['total_detections']}")
        print(f"Average FPS: {self.session_stats.get('average_fps', 0):.1f}")
        
        print(f"\nBehavior Counts:")
        for behavior, count in self.session_stats['behavior_counts'].items():
            percentage = (count / self.session_stats['total_detections'] * 100) if self.session_stats['total_detections'] > 0 else 0
            print(f"  {behavior}: {count} ({percentage:.1f}%)")
        
        if self.session_stats['alerts']:
            print(f"\nAlerts Generated: {len(self.session_stats['alerts'])}")
            for alert in self.session_stats['alerts'][-5:]:  # Show last 5 alerts
                print(f"  - {alert['message']} (Frame {alert['frame']})")


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Real-time cattle behavior analysis")
    parser.add_argument('--camera', '-c', type=int, help='Camera index (e.g., 0 for default camera)')
    parser.add_argument('--stream', '-s', help='Stream URL (e.g., rtsp://camera_ip:554/stream)')
    parser.add_argument('--file', '-f', help='Video file path for analysis')
    parser.add_argument('--output_dir', '-o', default='realtime_output', help='Output directory')
    parser.add_argument('--model', '-m', help='Path to trained model (optional)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--save_output', action='store_true', help='Save annotated video output')
    
    args = parser.parse_args()
    
    # Determine video source
    source = None
    if args.camera is not None:
        source = args.camera
    elif args.stream:
        source = args.stream
    elif args.file:
        source = args.file
        if not os.path.exists(args.file):
            print(f"Error: Video file not found: {args.file}")
            return 1
    else:
        print("Error: Must specify either --camera, --stream, or --file")
        return 1
    
    # Initialize analyzer
    try:
        analyzer = RealTimeAnalyzer(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing real-time analyzer: {e}")
        return 1
    
    # Run real-time analysis
    try:
        results = analyzer.run_realtime_analysis(
            source=source,
            save_output=args.save_output,
            output_dir=args.output_dir
        )
        
        # Save final session stats
        analyzer._save_session_stats(args.output_dir)
        
        return 0
        
    except Exception as e:
        print(f"Error during real-time analysis: {e}")
        return 1


if __name__ == '__main__':
    exit(main())