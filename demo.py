#!/usr/bin/env python3
"""
Demo script for thermotolerance dairy cattle behavior analysis

This script demonstrates the key features of the research system with
simulated data and generates example outputs.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from inference import CattleBehaviorAnalyzer
try:
    from annotation_tools import BehaviorVisualizer
    from video_processing import create_video_summary, export_to_csv
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: Some utility modules not available, running basic demo")


def create_demo_data():
    """Create simulated analysis results for demonstration"""
    
    # Simulate video properties
    video_properties = {
        'fps': 30.0,
        'total_frames': 3600,  # 2 minutes at 30 FPS
        'duration': 120.0,
        'width': 1280,
        'height': 720
    }
    
    # Simulate frame results
    frame_results = []
    np.random.seed(42)  # For reproducible demo
    
    for frame_idx in range(0, video_properties['total_frames'], 10):  # Sample every 10th frame
        timestamp = frame_idx / video_properties['fps']
        detections = []
        
        # Simulate different behaviors at different times
        if timestamp < 30:  # First 30 seconds - mostly grazing
            if np.random.random() < 0.7:
                detections.append({
                    'bbox': [320 + np.random.randint(-50, 50), 200 + np.random.randint(-30, 30), 
                            800 + np.random.randint(-50, 50), 600 + np.random.randint(-30, 30)],
                    'confidence': 0.85 + np.random.normal(0, 0.05),
                    'behavior': 'grazing',
                    'class_id': 3
                })
        elif timestamp < 60:  # Next 30 seconds - brushing behavior increases
            if np.random.random() < 0.4:
                detections.append({
                    'bbox': [400 + np.random.randint(-30, 30), 150 + np.random.randint(-20, 20),
                            750 + np.random.randint(-30, 30), 550 + np.random.randint(-20, 20)],
                    'confidence': 0.82 + np.random.normal(0, 0.07),
                    'behavior': 'brushing',
                    'class_id': 0
                })
            if np.random.random() < 0.3:
                detections.append({
                    'bbox': [200 + np.random.randint(-20, 20), 250 + np.random.randint(-20, 20),
                            600 + np.random.randint(-20, 20), 650 + np.random.randint(-20, 20)],
                    'confidence': 0.77 + np.random.normal(0, 0.06),
                    'behavior': 'grazing',
                    'class_id': 3
                })
        elif timestamp < 90:  # Next 30 seconds - drinking increases
            if np.random.random() < 0.6:
                detections.append({
                    'bbox': [100 + np.random.randint(-20, 20), 300 + np.random.randint(-20, 20),
                            450 + np.random.randint(-20, 20), 650 + np.random.randint(-20, 20)],
                    'confidence': 0.88 + np.random.normal(0, 0.04),
                    'behavior': 'drinking',
                    'class_id': 1
                })
            if np.random.random() < 0.2:
                detections.append({
                    'bbox': [500 + np.random.randint(-30, 30), 180 + np.random.randint(-20, 20),
                            850 + np.random.randint(-30, 30), 580 + np.random.randint(-20, 20)],
                    'confidence': 0.79 + np.random.normal(0, 0.08),
                    'behavior': 'brushing',
                    'class_id': 0
                })
        else:  # Last 30 seconds - some aggression
            if np.random.random() < 0.3:
                detections.append({
                    'bbox': [600 + np.random.randint(-40, 40), 200 + np.random.randint(-30, 30),
                            950 + np.random.randint(-40, 40), 600 + np.random.randint(-30, 30)],
                    'confidence': 0.75 + np.random.normal(0, 0.1),
                    'behavior': 'headbutting',
                    'class_id': 2
                })
            if np.random.random() < 0.4:
                detections.append({
                    'bbox': [150 + np.random.randint(-20, 20), 350 + np.random.randint(-20, 20),
                            500 + np.random.randint(-20, 20), 700 + np.random.randint(-20, 20)],
                    'confidence': 0.83 + np.random.normal(0, 0.05),
                    'behavior': 'drinking',
                    'class_id': 1
                })
        
        frame_results.append({
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'detections': detections
        })
    
    # Calculate behavior summary
    behavior_counts = {}
    total_detections = 0
    
    for frame_result in frame_results:
        for detection in frame_result['detections']:
            behavior = detection['behavior']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            total_detections += 1
    
    behavior_percentages = {}
    for behavior, count in behavior_counts.items():
        behavior_percentages[behavior] = (count / total_detections * 100) if total_detections > 0 else 0
    
    behavior_summary = {
        'total_detections': total_detections,
        'behavior_counts': behavior_counts,
        'behavior_percentages': behavior_percentages,
        'unique_behaviors': list(behavior_counts.keys())
    }
    
    # Complete analysis results
    analysis_results = {
        'video_path': 'demo_thermal_stress_behaviors.mp4',
        'video_properties': video_properties,
        'analysis_timestamp': datetime.now().isoformat(),
        'demo_mode': True,
        'frame_results': frame_results,
        'behavior_summary': behavior_summary,
        'processing_stats': {
            'processing_time': 15.3,
            'fps_processed': video_properties['total_frames'] / 15.3,
            'frames_processed': len(frame_results)
        }
    }
    
    return analysis_results


def run_demo(output_dir='demo_output', create_visualizations=True):
    """
    Run complete demo of the analysis system
    
    Args:
        output_dir (str): Directory for demo outputs
        create_visualizations (bool): Whether to create visualization plots
    """
    print("🐄 Thermotolerance Dairy Cattle Behavior Analysis - Demo")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate demo data
    print("📊 Generating demo analysis data...")
    analysis_results = create_demo_data()
    
    # Save raw results
    json_path = os.path.join(output_dir, 'demo_analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"✓ Demo results saved: {json_path}")
    
    # Export to CSV
    if UTILS_AVAILABLE:
        csv_path = os.path.join(output_dir, 'demo_analysis_results.csv')
        export_to_csv(analysis_results, csv_path)
        print(f"✓ CSV export saved: {csv_path}")
    
    # Create visualizations
    if create_visualizations and UTILS_AVAILABLE:
        print("\n📈 Creating visualization plots...")
        
        visualizer = BehaviorVisualizer()
        
        # Behavior distribution
        dist_path = os.path.join(output_dir, 'demo_behavior_distribution.png')
        visualizer.plot_behavior_distribution(analysis_results, dist_path, show=False)
        print(f"✓ Behavior distribution plot: {dist_path}")
        
        # Temporal analysis
        temporal_path = os.path.join(output_dir, 'demo_temporal_analysis.png')
        visualizer.plot_temporal_analysis(analysis_results, temporal_path, show=False)
        print(f"✓ Temporal analysis plot: {temporal_path}")
        
        # Confidence distribution
        conf_path = os.path.join(output_dir, 'demo_confidence_distribution.png')
        visualizer.plot_confidence_distribution(analysis_results, conf_path, show=False)
        print(f"✓ Confidence distribution plot: {conf_path}")
        
        # Create HTML report
        report_path = visualizer.create_summary_report(analysis_results, output_dir, 'demo')
        print(f"✓ HTML report created: {report_path}")
    
    # Print analysis summary
    print_demo_summary(analysis_results)
    
    # Create demo script examples
    create_demo_scripts(output_dir)
    
    print(f"\n🎉 Demo completed! All outputs saved to: {os.path.abspath(output_dir)}")
    return analysis_results


def print_demo_summary(analysis_results):
    """Print formatted demo summary"""
    print("\n" + "=" * 50)
    print("DEMO ANALYSIS SUMMARY")
    print("=" * 50)
    
    video_props = analysis_results['video_properties']
    behavior_summary = analysis_results['behavior_summary']
    processing_stats = analysis_results['processing_stats']
    
    print(f"Video Duration: {video_props['duration']:.1f} seconds")
    print(f"Total Frames: {video_props['total_frames']}")
    print(f"Processing Time: {processing_stats['processing_time']:.1f}s")
    print(f"Processing Speed: {processing_stats['fps_processed']:.1f} FPS")
    
    print(f"\nTotal Detections: {behavior_summary['total_detections']}")
    print(f"Unique Behaviors: {len(behavior_summary['unique_behaviors'])}")
    
    print("\nBehavior Distribution:")
    for behavior, percentage in behavior_summary['behavior_percentages'].items():
        count = behavior_summary['behavior_counts'][behavior]
        print(f"  {behavior:12}: {count:3d} detections ({percentage:5.1f}%)")
    
    print(f"\nThermal Stress Indicators:")
    stress_behaviors = ['brushing', 'drinking', 'headbutting']
    stress_count = sum(behavior_summary['behavior_counts'].get(b, 0) for b in stress_behaviors)
    stress_percentage = (stress_count / behavior_summary['total_detections'] * 100) if behavior_summary['total_detections'] > 0 else 0
    print(f"  Stress-related behaviors: {stress_count} ({stress_percentage:.1f}%)")
    
    if stress_percentage > 30:
        print("  ⚠️  HIGH thermal stress indicators detected!")
    elif stress_percentage > 15:
        print("  ⚡ MODERATE thermal stress indicators detected")
    else:
        print("  ✅ LOW thermal stress indicators")


def create_demo_scripts(output_dir):
    """Create example usage scripts"""
    
    # Example inference script
    inference_example = '''#!/bin/bash
# Example: Analyze a single video file

echo "Analyzing single video..."
python scripts/inference.py \\
    --video data/sample_videos/your_video.mp4 \\
    --output results/ \\
    --confidence 0.7

echo "Analysis complete! Check results/ directory for outputs."
'''
    
    script_path = os.path.join(output_dir, 'example_single_analysis.sh')
    with open(script_path, 'w') as f:
        f.write(inference_example)
    os.chmod(script_path, 0o755)
    
    # Example batch script
    batch_example = '''#!/bin/bash
# Example: Process multiple videos in batch

echo "Starting batch processing..."
python scripts/batch_inference.py \\
    --input_dir data/sample_videos/ \\
    --output_dir batch_results/ \\
    --pattern "*.mp4" \\
    --confidence 0.6

echo "Batch processing complete! Check batch_results/ directory."
'''
    
    batch_path = os.path.join(output_dir, 'example_batch_analysis.sh')
    with open(batch_path, 'w') as f:
        f.write(batch_example)
    os.chmod(batch_path, 0o755)
    
    # Example real-time script
    realtime_example = '''#!/bin/bash
# Example: Real-time analysis with webcam

echo "Starting real-time analysis..."
echo "Press 'q' to quit, 'r' to toggle recording, 's' to save stats"

python scripts/realtime_inference.py \\
    --camera 0 \\
    --confidence 0.7 \\
    --save_output \\
    --output_dir realtime_results/

echo "Real-time analysis stopped."
'''
    
    realtime_path = os.path.join(output_dir, 'example_realtime_analysis.sh')
    with open(realtime_path, 'w') as f:
        f.write(realtime_example)
    os.chmod(realtime_path, 0o755)
    
    print(f"✓ Example scripts created in {output_dir}")


def main():
    """Main function for demo script"""
    parser = argparse.ArgumentParser(description="Run thermotolerance dairy cattle analysis demo")
    parser.add_argument('--output_dir', '-o', default='demo_output', help='Output directory for demo results')
    parser.add_argument('--no_visualizations', action='store_true', help='Skip creating visualization plots')
    parser.add_argument('--quick', action='store_true', help='Run quick demo with minimal outputs')
    
    args = parser.parse_args()
    
    # Check if required modules are available
    if not UTILS_AVAILABLE and not args.no_visualizations:
        print("Warning: Visualization modules not available. Running demo without plots.")
        create_visualizations = False
    else:
        create_visualizations = not args.no_visualizations
    
    if args.quick:
        create_visualizations = False
    
    try:
        # Run the demo
        results = run_demo(
            output_dir=args.output_dir,
            create_visualizations=create_visualizations
        )
        
        print("\n" + "=" * 60)
        print("DEMO INSTRUCTIONS")
        print("=" * 60)
        print("1. Check the demo_output/ directory for generated files")
        print("2. View the HTML report for visual analysis results")
        print("3. Use the example scripts to test with your own videos")
        print("4. Install dependencies: pip install -r requirements.txt")
        print("5. Download models: python download_models.py")
        print("\nFor real analysis, replace demo data with actual video files!")
        
        return 0
        
    except Exception as e:
        print(f"Error running demo: {e}")
        return 1


if __name__ == '__main__':
    exit(main())