#!/usr/bin/env python3
"""
Thermotolerance Dairy Cattle Behavior Analysis - Single Video Inference

This script analyzes a single video file to detect and classify dairy cattle behaviors
associated with thermal stress including brushing, drinking, and head-butting.

Usage:
    python inference.py --video path/to/video.mp4 --output results/
    python inference.py --video sample.mp4 --model custom_model.pt --confidence 0.8
"""

import argparse
import cv2
import os
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import sys

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Warning: Some dependencies not installed. Running in demo mode.")
    YOLO = None
    torch = None


class CattleBehaviorAnalyzer:
    """
    Main class for analyzing cattle behaviors in videos
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5, device='auto'):
        """
        Initialize the behavior analyzer
        
        Args:
            model_path (str): Path to trained model, if None uses demo mode
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        
        # Behavior categories
        self.behaviors = {
            0: 'brushing',
            1: 'drinking', 
            2: 'headbutting',
            3: 'grazing',
            4: 'resting',
            5: 'walking'
        }
        
        # Demo mode tracking
        self.demo_mode = model_path is None or not os.path.exists(model_path) if model_path else True
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            if torch and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_model(self, model_path):
        """Load the trained model or initialize demo mode"""
        if model_path and os.path.exists(model_path) and YOLO:
            try:
                model = YOLO(model_path)
                print(f"Loaded model from {model_path}")
                return model
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Running in demo mode")
                return None
        else:
            print("Running in demo mode (no model specified or dependencies missing)")
            return None
    
    def analyze_frame(self, frame, frame_idx):
        """
        Analyze a single frame for cattle behaviors
        
        Args:
            frame: OpenCV frame
            frame_idx: Frame number
            
        Returns:
            dict: Detection results for this frame
        """
        if self.model:
            # Real inference with trained model
            results = self.model(frame, conf=self.confidence_threshold)
            detections = self._process_yolo_results(results[0])
        else:
            # Demo mode - simulate detections
            detections = self._simulate_detections(frame, frame_idx)
            
        return {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / 30.0,  # Assuming 30 FPS
            'detections': detections
        }
    
    def _process_yolo_results(self, results):
        """Process YOLO model results"""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'behavior': self.behaviors.get(int(cls), 'unknown'),
                    'class_id': int(cls)
                }
                detections.append(detection)
                
        return detections
    
    def _simulate_detections(self, frame, frame_idx):
        """Simulate behavior detections for demo purposes"""
        detections = []
        h, w = frame.shape[:2]
        
        # Simulate different behaviors at different time intervals
        time_sec = frame_idx / 30.0
        
        # Simulate brushing behavior (every 5-8 seconds)
        if int(time_sec) % 7 < 3:
            detections.append({
                'bbox': [w*0.2, h*0.3, w*0.6, h*0.8],
                'confidence': 0.85 + np.random.normal(0, 0.05),
                'behavior': 'brushing',
                'class_id': 0
            })
        
        # Simulate drinking behavior (every 10-15 seconds)  
        if int(time_sec) % 12 < 2:
            detections.append({
                'bbox': [w*0.1, h*0.4, w*0.4, h*0.9],
                'confidence': 0.78 + np.random.normal(0, 0.08),
                'behavior': 'drinking', 
                'class_id': 1
            })
            
        # Simulate head-butting (occasionally)
        if int(time_sec) % 20 < 1:
            detections.append({
                'bbox': [w*0.5, h*0.2, w*0.9, h*0.7],
                'confidence': 0.72 + np.random.normal(0, 0.1),
                'behavior': 'headbutting',
                'class_id': 2
            })
            
        return detections
    
    def analyze_video(self, video_path, output_dir=None):
        """
        Analyze complete video for cattle behaviors
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save results
            
        Returns:
            dict: Complete analysis results
        """
        print(f"Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Analysis results
        results = {
            'video_path': video_path,
            'video_properties': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'demo_mode': self.demo_mode,
            'frame_results': [],
            'behavior_summary': {}
        }
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze frame
            frame_result = self.analyze_frame(frame, frame_idx)
            results['frame_results'].append(frame_result)
            
            # Progress update
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
                
            frame_idx += 1
            
        cap.release()
        
        # Calculate processing stats
        processing_time = time.time() - start_time
        results['processing_stats'] = {
            'processing_time': processing_time,
            'fps_processed': total_frames / processing_time,
            'frames_processed': frame_idx
        }
        
        # Generate behavior summary
        results['behavior_summary'] = self._generate_behavior_summary(results['frame_results'])
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir, video_path)
            
        print(f"Analysis complete! Processed {frame_idx} frames in {processing_time:.2f}s")
        
        return results
    
    def _generate_behavior_summary(self, frame_results):
        """Generate summary statistics for detected behaviors"""
        behavior_counts = {}
        behavior_durations = {}
        total_detections = 0
        
        for frame_result in frame_results:
            for detection in frame_result['detections']:
                behavior = detection['behavior']
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                total_detections += 1
                
        # Calculate percentages
        behavior_percentages = {}
        for behavior, count in behavior_counts.items():
            behavior_percentages[behavior] = (count / total_detections * 100) if total_detections > 0 else 0
            
        return {
            'total_detections': total_detections,
            'behavior_counts': behavior_counts,
            'behavior_percentages': behavior_percentages,
            'unique_behaviors': list(behavior_counts.keys())
        }
    
    def _save_results(self, results, output_dir, video_path):
        """Save analysis results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{video_name}_analysis_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Save summary CSV
        csv_path = os.path.join(output_dir, f"{video_name}_summary_{timestamp}.csv")
        self._save_summary_csv(results, csv_path)
        
    def _save_summary_csv(self, results, csv_path):
        """Save behavior summary as CSV"""
        try:
            import pandas as pd
            
            # Create summary DataFrame
            summary_data = []
            behavior_summary = results['behavior_summary']
            
            for behavior in behavior_summary.get('behavior_counts', {}):
                summary_data.append({
                    'behavior': behavior,
                    'count': behavior_summary['behavior_counts'][behavior],
                    'percentage': behavior_summary['behavior_percentages'][behavior]
                })
                
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
            print(f"Summary saved to: {csv_path}")
            
        except ImportError:
            print("Pandas not available, skipping CSV export")


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Analyze dairy cattle behaviors in video")
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='results', help='Output directory for results')
    parser.add_argument('--model', '-m', help='Path to trained model (optional, uses demo mode if not provided)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
        
    # Initialize analyzer
    try:
        analyzer = CattleBehaviorAnalyzer(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        return 1
    
    # Run analysis
    try:
        results = analyzer.analyze_video(args.video, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Video: {args.video}")
        print(f"Total detections: {results['behavior_summary']['total_detections']}")
        print(f"Processing time: {results['processing_stats']['processing_time']:.2f}s")
        print(f"Demo mode: {results['demo_mode']}")
        
        print("\nBehavior Distribution:")
        for behavior, percentage in results['behavior_summary']['behavior_percentages'].items():
            count = results['behavior_summary']['behavior_counts'][behavior]
            print(f"  {behavior}: {count} detections ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())