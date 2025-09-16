#!/usr/bin/env python3
"""
Thermotolerance Dairy Cattle Behavior Analysis - Batch Video Processing

This script processes multiple videos in a directory to detect and classify
dairy cattle behaviors associated with thermal stress.

Usage:
    python batch_inference.py --input_dir videos/ --output_dir results/
    python batch_inference.py --input_dir data/videos/ --pattern "*.mp4" --model custom_model.pt
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import glob
import time

# Add current script directory and utils to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import the main analyzer from inference.py
from inference import CattleBehaviorAnalyzer

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class BatchProcessor:
    """
    Batch processor for analyzing multiple cattle behavior videos
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5, device='auto'):
        """
        Initialize batch processor
        
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
        
    def find_videos(self, input_dir, pattern="*.mp4"):
        """
        Find all video files in input directory
        
        Args:
            input_dir (str): Directory to search for videos
            pattern (str): File pattern to match (e.g., "*.mp4", "*.avi")
            
        Returns:
            list: List of video file paths
        """
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"]
        
        if pattern != "*.mp4":
            # Use custom pattern
            patterns = [pattern]
        else:
            # Use all common video extensions
            patterns = video_extensions
            
        video_files = []
        for pattern in patterns:
            search_pattern = os.path.join(input_dir, "**", pattern)
            video_files.extend(glob.glob(search_pattern, recursive=True))
            
        return sorted(list(set(video_files)))  # Remove duplicates and sort
    
    def process_batch(self, input_dir, output_dir, pattern="*.mp4", skip_existing=True):
        """
        Process all videos in a directory
        
        Args:
            input_dir (str): Input directory containing videos
            output_dir (str): Output directory for results
            pattern (str): File pattern for video selection
            skip_existing (bool): Skip videos that already have results
            
        Returns:
            dict: Batch processing results
        """
        print(f"Starting batch processing...")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Pattern: {pattern}")
        
        # Find videos
        video_files = self.find_videos(input_dir, pattern)
        
        if not video_files:
            print(f"No video files found in {input_dir} with pattern {pattern}")
            return None
            
        print(f"Found {len(video_files)} video files")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize batch results
        batch_results = {
            'batch_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'pattern': pattern,
            'total_videos': len(video_files),
            'start_time': datetime.now().isoformat(),
            'video_results': [],
            'summary_stats': {},
            'errors': []
        }
        
        # Process each video
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
            
            try:
                # Check if results already exist
                if skip_existing and self._results_exist(video_path, output_dir):
                    print(f"Skipping {os.path.basename(video_path)} - results already exist")
                    skipped_count += 1
                    continue
                
                # Analyze video
                start_time = time.time()
                results = self.analyzer.analyze_video(video_path, output_dir)
                processing_time = time.time() - start_time
                
                # Add to batch results
                video_result = {
                    'video_path': video_path,
                    'video_name': os.path.basename(video_path),
                    'processing_time': processing_time,
                    'success': True,
                    'behavior_summary': results['behavior_summary'],
                    'video_properties': results['video_properties']
                }
                
                batch_results['video_results'].append(video_result)
                processed_count += 1
                
                print(f"✓ Completed in {processing_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Error processing {video_path}: {str(e)}"
                print(f"✗ {error_msg}")
                
                batch_results['errors'].append({
                    'video_path': video_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                error_count += 1
        
        # Finalize batch results
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['processing_summary'] = {
            'processed': processed_count,
            'skipped': skipped_count, 
            'errors': error_count,
            'total_time': time.time() - time.mktime(datetime.fromisoformat(batch_results['start_time']).timetuple())
        }
        
        # Generate aggregate statistics
        batch_results['summary_stats'] = self._generate_batch_summary(batch_results['video_results'])
        
        # Save batch results
        self._save_batch_results(batch_results, output_dir)
        
        # Print final summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _results_exist(self, video_path, output_dir):
        """Check if results already exist for a video"""
        video_name = Path(video_path).stem
        pattern = os.path.join(output_dir, f"{video_name}_analysis_*.json")
        return len(glob.glob(pattern)) > 0
    
    def _generate_batch_summary(self, video_results):
        """Generate aggregate statistics across all processed videos"""
        if not video_results:
            return {}
            
        # Aggregate behavior counts
        total_behavior_counts = {}
        total_detections = 0
        total_duration = 0
        
        for video_result in video_results:
            behavior_summary = video_result.get('behavior_summary', {})
            video_duration = video_result.get('video_properties', {}).get('duration', 0)
            total_duration += video_duration
            
            for behavior, count in behavior_summary.get('behavior_counts', {}).items():
                total_behavior_counts[behavior] = total_behavior_counts.get(behavior, 0) + count
                total_detections += count
        
        # Calculate percentages
        behavior_percentages = {}
        for behavior, count in total_behavior_counts.items():
            behavior_percentages[behavior] = (count / total_detections * 100) if total_detections > 0 else 0
        
        # Calculate rates (detections per hour)
        behavior_rates = {}
        for behavior, count in total_behavior_counts.items():
            behavior_rates[behavior] = (count / (total_duration / 3600)) if total_duration > 0 else 0
        
        return {
            'total_detections': total_detections,
            'total_duration_hours': total_duration / 3600,
            'behavior_counts': total_behavior_counts,
            'behavior_percentages': behavior_percentages,
            'behavior_rates_per_hour': behavior_rates,
            'average_processing_time': sum(vr.get('processing_time', 0) for vr in video_results) / len(video_results)
        }
    
    def _save_batch_results(self, batch_results, output_dir):
        """Save batch processing results"""
        # Save complete JSON results
        json_path = os.path.join(output_dir, f"batch_results_{batch_results['batch_id']}.json")
        with open(json_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"\nBatch results saved to: {json_path}")
        
        # Save summary CSV if pandas available
        if PANDAS_AVAILABLE:
            self._save_batch_csv(batch_results, output_dir)
    
    def _save_batch_csv(self, batch_results, output_dir):
        """Save batch summary as CSV"""
        try:
            # Video-level summary
            video_data = []
            for video_result in batch_results['video_results']:
                row = {
                    'video_name': video_result['video_name'],
                    'duration_seconds': video_result.get('video_properties', {}).get('duration', 0),
                    'processing_time': video_result.get('processing_time', 0),
                    'total_detections': video_result.get('behavior_summary', {}).get('total_detections', 0)
                }
                
                # Add behavior counts
                behavior_counts = video_result.get('behavior_summary', {}).get('behavior_counts', {})
                for behavior, count in behavior_counts.items():
                    row[f'{behavior}_count'] = count
                    
                video_data.append(row)
            
            # Save video summary
            df_videos = pd.DataFrame(video_data)
            csv_path = os.path.join(output_dir, f"batch_video_summary_{batch_results['batch_id']}.csv")
            df_videos.to_csv(csv_path, index=False)
            print(f"Video summary CSV saved to: {csv_path}")
            
            # Behavior summary
            summary_stats = batch_results.get('summary_stats', {})
            behavior_data = []
            for behavior in summary_stats.get('behavior_counts', {}):
                behavior_data.append({
                    'behavior': behavior,
                    'total_count': summary_stats['behavior_counts'][behavior],
                    'percentage': summary_stats['behavior_percentages'][behavior],
                    'rate_per_hour': summary_stats['behavior_rates_per_hour'][behavior]
                })
            
            df_behaviors = pd.DataFrame(behavior_data)
            csv_path = os.path.join(output_dir, f"batch_behavior_summary_{batch_results['batch_id']}.csv")
            df_behaviors.to_csv(csv_path, index=False)
            print(f"Behavior summary CSV saved to: {csv_path}")
            
        except Exception as e:
            print(f"Error saving CSV files: {e}")
    
    def _print_batch_summary(self, batch_results):
        """Print batch processing summary"""
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        summary = batch_results['processing_summary']
        print(f"Total videos found: {batch_results['total_videos']}")
        print(f"Successfully processed: {summary['processed']}")
        print(f"Skipped (existing results): {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Total processing time: {summary['total_time']:.2f}s")
        
        if batch_results['errors']:
            print(f"\nErrors encountered:")
            for error in batch_results['errors']:
                print(f"  - {os.path.basename(error['video_path'])}: {error['error']}")
        
        # Behavior statistics
        stats = batch_results.get('summary_stats', {})
        if stats:
            print(f"\nAggregate Behavior Statistics:")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Total duration: {stats['total_duration_hours']:.2f} hours")
            print(f"Average processing time: {stats['average_processing_time']:.2f}s per video")
            
            print(f"\nBehavior Distribution:")
            for behavior, percentage in stats.get('behavior_percentages', {}).items():
                count = stats['behavior_counts'][behavior]
                rate = stats['behavior_rates_per_hour'][behavior]
                print(f"  {behavior}: {count} detections ({percentage:.1f}%, {rate:.1f}/hour)")


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Batch process cattle behavior videos")
    parser.add_argument('--input_dir', '-i', required=True, help='Input directory containing videos')
    parser.add_argument('--output_dir', '-o', default='batch_results', help='Output directory for results')
    parser.add_argument('--pattern', '-p', default='*.mp4', help='File pattern to match (e.g., "*.mp4", "farm1_*.avi")')
    parser.add_argument('--model', '-m', help='Path to trained model (optional)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip videos with existing results')
    parser.add_argument('--force', action='store_true', help='Process all videos, even if results exist')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input path is not a directory: {args.input_dir}")
        return 1
    
    # Set skip_existing based on force flag
    skip_existing = not args.force
    
    # Initialize batch processor
    try:
        processor = BatchProcessor(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing batch processor: {e}")
        return 1
    
    # Run batch processing
    try:
        results = processor.process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            skip_existing=skip_existing
        )
        
        if results is None:
            return 1
            
        # Success
        processed = results['processing_summary']['processed']
        errors = results['processing_summary']['errors']
        
        if errors > 0:
            print(f"\nBatch completed with {errors} errors out of {processed + errors} videos")
            return 1
        else:
            print(f"\nBatch processing completed successfully! Processed {processed} videos.")
            return 0
            
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return 1


if __name__ == '__main__':
    exit(main())