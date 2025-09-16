#!/bin/bash
# Example: Process multiple videos in batch

echo "Starting batch processing..."
python scripts/batch_inference.py \
    --input_dir data/sample_videos/ \
    --output_dir batch_results/ \
    --pattern "*.mp4" \
    --confidence 0.6

echo "Batch processing complete! Check batch_results/ directory."
