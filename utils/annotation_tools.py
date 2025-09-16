"""
Annotation and visualization tools for thermotolerance dairy cattle behavior analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from collections import defaultdict


class BehaviorVisualizer:
    """
    Class for creating visualizations of cattle behavior analysis results
    """
    
    def __init__(self, style='seaborn-v0_8'):
        """Initialize visualizer with plotting style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color palette for different behaviors
        self.behavior_colors = {
            'brushing': '#2E8B57',      # Sea Green
            'drinking': '#4169E1',      # Royal Blue
            'headbutting': '#DC143C',   # Crimson
            'grazing': '#228B22',       # Forest Green
            'resting': '#8B4513',       # Saddle Brown
            'walking': '#FF8C00'        # Dark Orange
        }
    
    def plot_behavior_distribution(self, analysis_results, save_path=None, show=True):
        """
        Create pie chart of behavior distribution
        
        Args:
            analysis_results (dict): Analysis results from inference
            save_path (str): Path to save plot
            show (bool): Whether to display plot
        """
        behavior_summary = analysis_results.get('behavior_summary', {})
        behavior_counts = behavior_summary.get('behavior_counts', {})
        
        if not behavior_counts:
            print("No behavior data found for visualization")
            return
        
        # Prepare data
        behaviors = list(behavior_counts.keys())
        counts = list(behavior_counts.values())
        colors = [self.behavior_colors.get(behavior, '#808080') for behavior in behaviors]
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(counts, labels=behaviors, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        # Customize appearance
        ax.set_title('Cattle Behavior Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend with counts
        legend_labels = [f'{behavior}: {count}' for behavior, count in zip(behaviors, counts)]
        ax.legend(wedges, legend_labels, title="Behaviors", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Behavior distribution plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_temporal_analysis(self, analysis_results, save_path=None, show=True, window_minutes=5):
        """
        Create temporal analysis plot showing behavior over time
        
        Args:
            analysis_results (dict): Analysis results
            save_path (str): Path to save plot
            show (bool): Whether to display plot
            window_minutes (int): Time window for aggregation in minutes
        """
        frame_results = analysis_results.get('frame_results', [])
        video_props = analysis_results.get('video_properties', {})
        fps = video_props.get('fps', 30)
        
        if not frame_results:
            print("No frame data found for temporal analysis")
            return
        
        # Aggregate behaviors by time windows
        window_seconds = window_minutes * 60
        window_frames = int(window_seconds * fps)
        
        time_windows = defaultdict(lambda: defaultdict(int))
        
        for frame_result in frame_results:
            frame_idx = frame_result['frame_idx']
            window_idx = frame_idx // window_frames
            timestamp = window_idx * window_minutes
            
            for detection in frame_result.get('detections', []):
                behavior = detection['behavior']
                time_windows[timestamp][behavior] += 1
        
        # Prepare data for plotting
        timestamps = sorted(time_windows.keys())
        all_behaviors = set()
        for window_data in time_windows.values():
            all_behaviors.update(window_data.keys())
        all_behaviors = sorted(all_behaviors)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bottom = np.zeros(len(timestamps))
        for behavior in all_behaviors:
            counts = [time_windows[ts].get(behavior, 0) for ts in timestamps]
            color = self.behavior_colors.get(behavior, '#808080')
            ax.bar(timestamps, counts, bottom=bottom, label=behavior, color=color, alpha=0.8)
            bottom += counts
        
        ax.set_xlabel(f'Time (minutes)', fontsize=12)
        ax.set_ylabel('Behavior Detections', fontsize=12)
        ax.set_title(f'Cattle Behavior Over Time ({window_minutes}-minute windows)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal analysis plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confidence_distribution(self, analysis_results, save_path=None, show=True):
        """
        Create histogram of detection confidence scores
        
        Args:
            analysis_results (dict): Analysis results
            save_path (str): Path to save plot
            show (bool): Whether to display plot
        """
        frame_results = analysis_results.get('frame_results', [])
        
        # Collect confidence scores by behavior
        behavior_confidences = defaultdict(list)
        
        for frame_result in frame_results:
            for detection in frame_result.get('detections', []):
                behavior = detection['behavior']
                confidence = detection['confidence']
                behavior_confidences[behavior].append(confidence)
        
        if not behavior_confidences:
            print("No detection data found for confidence analysis")
            return
        
        # Create subplot for each behavior
        behaviors = sorted(behavior_confidences.keys())
        n_behaviors = len(behaviors)
        
        fig, axes = plt.subplots(n_behaviors, 1, figsize=(10, 3 * n_behaviors))
        if n_behaviors == 1:
            axes = [axes]
        
        for i, behavior in enumerate(behaviors):
            confidences = behavior_confidences[behavior]
            color = self.behavior_colors.get(behavior, '#808080')
            
            axes[i].hist(confidences, bins=20, alpha=0.7, color=color, edgecolor='black')
            axes[i].set_title(f'{behavior.capitalize()} Confidence Distribution')
            axes[i].set_xlabel('Confidence Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            axes[i].axvline(mean_conf, color='red', linestyle='--', 
                           label=f'Mean: {mean_conf:.3f}±{std_conf:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_detection_heatmap_plot(self, heatmap, save_path=None, show=True):
        """
        Create visualization of detection heatmap
        
        Args:
            heatmap (numpy.ndarray): Heatmap array from create_detection_heatmap
            save_path (str): Path to save plot
            show (bool): Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest', aspect='auto')
        ax.set_title('Cattle Detection Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Width (pixels)')
        ax.set_ylabel('Frame Height (pixels)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Detection Intensity', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detection heatmap plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_summary_report(self, analysis_results, output_dir, video_name):
        """
        Create comprehensive visual summary report
        
        Args:
            analysis_results (dict): Analysis results
            output_dir (str): Directory to save report
            video_name (str): Name of the analyzed video
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual plots
        plots = {}
        
        # 1. Behavior distribution
        dist_path = os.path.join(output_dir, f'{video_name}_behavior_distribution.png')
        self.plot_behavior_distribution(analysis_results, dist_path, show=False)
        plots['distribution'] = dist_path
        
        # 2. Temporal analysis
        temporal_path = os.path.join(output_dir, f'{video_name}_temporal_analysis.png')
        self.plot_temporal_analysis(analysis_results, temporal_path, show=False)
        plots['temporal'] = temporal_path
        
        # 3. Confidence distribution
        conf_path = os.path.join(output_dir, f'{video_name}_confidence_distribution.png')
        self.plot_confidence_distribution(analysis_results, conf_path, show=False)
        plots['confidence'] = conf_path
        
        # Create summary HTML report
        html_path = os.path.join(output_dir, f'{video_name}_analysis_report.html')
        self._create_html_report(analysis_results, plots, html_path, video_name)
        
        print(f"Summary report created: {html_path}")
        return html_path
    
    def _create_html_report(self, analysis_results, plots, html_path, video_name):
        """Create HTML report with embedded plots"""
        
        behavior_summary = analysis_results.get('behavior_summary', {})
        video_props = analysis_results.get('video_properties', {})
        processing_stats = analysis_results.get('processing_stats', {})
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cattle Behavior Analysis Report - {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #bdc3c7; border-radius: 5px; }}
        .behavior-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }}
        .behavior-item {{ background-color: #e8f4fd; padding: 10px; border-radius: 5px; text-align: center; }}
        .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #95a5a6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐄 Cattle Behavior Analysis Report</h1>
        <p style="text-align: center; color: #7f8c8d; font-style: italic;">Video: {video_name}</p>
        <p style="text-align: center; color: #7f8c8d;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Video Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{video_props.get('duration', 0):.1f}s</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{video_props.get('total_frames', 0)}</div>
                <div class="stat-label">Total Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{video_props.get('fps', 0):.1f}</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{behavior_summary.get('total_detections', 0)}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{processing_stats.get('processing_time', 0):.1f}s</div>
                <div class="stat-label">Processing Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{processing_stats.get('fps_processed', 0):.1f}</div>
                <div class="stat-label">Processing FPS</div>
            </div>
        </div>
        
        <h2>🎯 Detected Behaviors</h2>
        <div class="behavior-list">
        """
        
        # Add behavior cards
        for behavior, count in behavior_summary.get('behavior_counts', {}).items():
            percentage = behavior_summary.get('behavior_percentages', {}).get(behavior, 0)
            html_content += f"""
            <div class="behavior-item">
                <strong>{behavior.capitalize()}</strong><br>
                {count} detections<br>
                <small>{percentage:.1f}%</small>
            </div>
            """
        
        html_content += """
        </div>
        
        <h2>📈 Behavior Distribution</h2>
        <div class="plot-container">
        """
        
        # Add plots
        for plot_type, plot_path in plots.items():
            if os.path.exists(plot_path):
                plot_name = os.path.basename(plot_path)
                html_content += f'<img src="{plot_name}" alt="{plot_type.capitalize()} Plot"><br><br>'
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <p>Generated by Thermotolerance Dairy Cattle Behavior Analysis System</p>
            <p>Demo Mode: {analysis_results.get('demo_mode', False)}</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)


def draw_behavior_annotations(image, detections, colors=None):
    """
    Draw behavior annotations on an image
    
    Args:
        image (numpy.ndarray): Input image
        detections (list): List of detection dictionaries
        colors (dict): Color mapping for behaviors
        
    Returns:
        numpy.ndarray: Annotated image
    """
    if colors is None:
        colors = {
            'brushing': (0, 255, 0),
            'drinking': (255, 0, 0),
            'headbutting': (0, 0, 255),
            'grazing': (0, 255, 255),
            'resting': (255, 255, 0),
            'walking': (255, 0, 255)
        }
    
    annotated = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        behavior = detection['behavior']
        confidence = detection['confidence']
        
        color = colors.get(behavior, (128, 128, 128))
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{behavior}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated


def create_annotated_video(video_path, analysis_results, output_path, fps_scale=1.0):
    """
    Create video with behavior annotations overlaid
    
    Args:
        video_path (str): Input video path
        analysis_results (dict): Analysis results
        output_path (str): Output video path
        fps_scale (float): Scale factor for output FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) * fps_scale
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frame lookup for detections
    frame_detections = {}
    for frame_result in analysis_results.get('frame_results', []):
        frame_idx = frame_result['frame_idx']
        frame_detections[frame_idx] = frame_result.get('detections', [])
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        detections = frame_detections.get(frame_count, [])
        
        # Annotate frame
        annotated_frame = draw_behavior_annotations(frame, detections)
        
        # Write frame
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")