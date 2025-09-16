#!/bin/bash
# Example: Analyze a single video file

echo "Analyzing single video..."
python scripts/inference.py \
    --video data/sample_videos/your_video.mp4 \
    --output results/ \
    --confidence 0.7

echo "Analysis complete! Check results/ directory for outputs."
