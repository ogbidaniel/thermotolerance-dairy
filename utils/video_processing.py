"""
Video processing utilities for thermotolerance dairy cattle behavior analysis
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime, timedelta


def get_video_info(video_path):
    """
    Get detailed information about a video file
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'file_size': os.path.getsize(video_path)
    }
    
    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    info['resolution'] = f"{info['width']}x{info['height']}"
    info['codec_str'] = ''.join([chr((info['codec'] >> 8 * i) & 0xFF) for i in range(4)])
    info['file_size_mb'] = info['file_size'] / (1024 * 1024)
    
    cap.release()
    return info


def extract_frames(video_path, output_dir, interval_seconds=1.0, max_frames=None):
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to save extracted frames
        interval_seconds (float): Time interval between extracted frames
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: Paths to extracted frame images
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if max_frames and extracted_count >= max_frames:
                break
                
            timestamp = frame_count / fps
            frame_filename = f"{video_name}_frame_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")
    
    return frame_paths


def create_video_summary(video_path, analysis_results=None):
    """
    Create a summary of video analysis results
    
    Args:
        video_path (str): Path to video file
        analysis_results (dict): Analysis results from inference
        
    Returns:
        dict: Video summary
    """
    video_info = get_video_info(video_path)
    
    summary = {
        'video_info': video_info,
        'analysis_timestamp': datetime.now().isoformat(),
        'summary_stats': {}
    }
    
    if analysis_results:
        behavior_summary = analysis_results.get('behavior_summary', {})
        
        summary['summary_stats'] = {
            'total_detections': behavior_summary.get('total_detections', 0),
            'unique_behaviors': len(behavior_summary.get('behavior_counts', {})),
            'behavior_distribution': behavior_summary.get('behavior_percentages', {}),
            'processing_time': analysis_results.get('processing_stats', {}).get('processing_time', 0),
            'detection_rate': behavior_summary.get('total_detections', 0) / video_info['duration'] if video_info['duration'] > 0 else 0
        }
    
    return summary


def resize_video(input_path, output_path, target_width=640, target_height=640):
    """
    Resize video to target dimensions
    
    Args:
        input_path (str): Input video path
        output_path (str): Output video path
        target_width (int): Target width
        target_height (int): Target height
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Resized video saved to: {output_path}")


def create_behavior_timeline(analysis_results, output_path=None):
    """
    Create a timeline visualization of detected behaviors
    
    Args:
        analysis_results (dict): Analysis results from inference
        output_path (str): Path to save timeline JSON
        
    Returns:
        dict: Timeline data
    """
    frame_results = analysis_results.get('frame_results', [])
    video_props = analysis_results.get('video_properties', {})
    fps = video_props.get('fps', 30)
    
    timeline = {
        'metadata': {
            'fps': fps,
            'total_frames': len(frame_results),
            'duration': len(frame_results) / fps,
            'created': datetime.now().isoformat()
        },
        'timeline': []
    }
    
    for frame_result in frame_results:
        frame_idx = frame_result['frame_idx']
        timestamp = frame_idx / fps
        
        timeline_entry = {
            'frame': frame_idx,
            'time': timestamp,
            'behaviors': []
        }
        
        for detection in frame_result.get('detections', []):
            behavior_entry = {
                'behavior': detection['behavior'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            }
            timeline_entry['behaviors'].append(behavior_entry)
        
        timeline['timeline'].append(timeline_entry)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        print(f"Timeline saved to: {output_path}")
    
    return timeline


def filter_detections_by_confidence(analysis_results, min_confidence=0.7):
    """
    Filter detection results by confidence threshold
    
    Args:
        analysis_results (dict): Analysis results
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        dict: Filtered analysis results
    """
    filtered_results = analysis_results.copy()
    filtered_frame_results = []
    
    for frame_result in analysis_results.get('frame_results', []):
        filtered_frame = frame_result.copy()
        filtered_detections = []
        
        for detection in frame_result.get('detections', []):
            if detection['confidence'] >= min_confidence:
                filtered_detections.append(detection)
        
        filtered_frame['detections'] = filtered_detections
        filtered_frame_results.append(filtered_frame)
    
    filtered_results['frame_results'] = filtered_frame_results
    
    # Recalculate behavior summary
    behavior_counts = {}
    total_detections = 0
    
    for frame_result in filtered_frame_results:
        for detection in frame_result['detections']:
            behavior = detection['behavior']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            total_detections += 1
    
    behavior_percentages = {}
    for behavior, count in behavior_counts.items():
        behavior_percentages[behavior] = (count / total_detections * 100) if total_detections > 0 else 0
    
    filtered_results['behavior_summary'] = {
        'total_detections': total_detections,
        'behavior_counts': behavior_counts,
        'behavior_percentages': behavior_percentages,
        'unique_behaviors': list(behavior_counts.keys()),
        'filter_confidence': min_confidence
    }
    
    return filtered_results


def create_detection_heatmap(analysis_results, video_width=1920, video_height=1080):
    """
    Create a heatmap of detection locations
    
    Args:
        analysis_results (dict): Analysis results
        video_width (int): Video width
        video_height (int): Video height
        
    Returns:
        numpy.ndarray: Heatmap array
    """
    heatmap = np.zeros((video_height, video_width), dtype=np.float32)
    
    for frame_result in analysis_results.get('frame_results', []):
        for detection in frame_result.get('detections', []):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, video_width - 1))
            y1 = max(0, min(y1, video_height - 1))
            x2 = max(0, min(x2, video_width - 1))
            y2 = max(0, min(y2, video_height - 1))
            
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] += confidence
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def export_to_csv(analysis_results, output_path):
    """
    Export analysis results to CSV format
    
    Args:
        analysis_results (dict): Analysis results
        output_path (str): Path to save CSV file
    """
    try:
        import pandas as pd
        
        # Prepare data for CSV
        csv_data = []
        
        for frame_result in analysis_results.get('frame_results', []):
            frame_idx = frame_result['frame_idx']
            timestamp = frame_result['timestamp']
            
            if frame_result.get('detections'):
                for detection in frame_result['detections']:
                    row = {
                        'frame': frame_idx,
                        'timestamp': timestamp,
                        'behavior': detection['behavior'],
                        'confidence': detection['confidence'],
                        'bbox_x1': detection['bbox'][0],
                        'bbox_y1': detection['bbox'][1],
                        'bbox_x2': detection['bbox'][2],
                        'bbox_y2': detection['bbox'][3]
                    }
                    csv_data.append(row)
            else:
                # Include frames with no detections
                row = {
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'behavior': None,
                    'confidence': None,
                    'bbox_x1': None,
                    'bbox_y1': None,
                    'bbox_x2': None,
                    'bbox_y2': None
                }
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to CSV: {output_path}")
        
    except ImportError:
        print("Pandas not available, cannot export to CSV")
        return False
    
    return True