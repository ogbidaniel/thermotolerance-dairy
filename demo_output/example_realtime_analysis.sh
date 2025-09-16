#!/bin/bash
# Example: Real-time analysis with webcam

echo "Starting real-time analysis..."
echo "Press 'q' to quit, 'r' to toggle recording, 's' to save stats"

python scripts/realtime_inference.py \
    --camera 0 \
    --confidence 0.7 \
    --save_output \
    --output_dir realtime_results/

echo "Real-time analysis stopped."
