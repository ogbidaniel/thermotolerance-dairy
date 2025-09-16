# Sample Videos Directory

This directory contains example videos for testing the thermotolerance dairy cattle behavior analysis system.

## Video Files

### Demo Videos (Placeholder)

Since actual research videos cannot be included in the repository due to size and privacy constraints, this directory serves as a placeholder for:

1. **brushing_behavior_demo.mp4** - Example of cattle exhibiting increased brushing behavior under thermal stress
2. **drinking_behavior_demo.mp4** - Example of elevated water consumption patterns
3. **headbutting_behavior_demo.mp4** - Example of aggressive behaviors during heat stress

## Using Your Own Videos

To test the system with your own videos:

1. Place video files in this directory
2. Supported formats: MP4, AVI, MOV, MKV, WMV, FLV
3. Recommended resolution: 640x480 or higher
4. Recommended frame rate: 15-30 FPS

## Running Analysis

### Single Video Analysis
```bash
python scripts/inference.py --video data/sample_videos/your_video.mp4 --output results/
```

### Batch Processing
```bash
python scripts/batch_inference.py --input_dir data/sample_videos/ --output_dir results/
```

### Real-time Analysis (with recorded video)
```bash
python scripts/realtime_inference.py --file data/sample_videos/your_video.mp4 --save_output
```

## Expected Behaviors

The system is trained to detect the following behaviors:

- **Brushing**: Self-grooming and environmental rubbing (thermal stress indicator)
- **Drinking**: Water consumption events (strong thermal stress indicator)
- **Head-butting**: Aggressive interactions (moderate thermal stress indicator)
- **Grazing**: Feeding behaviors (baseline activity)
- **Resting**: Lying and ruminating (baseline activity)
- **Walking**: General locomotion (baseline activity)

## Demo Mode

When no trained models are available, the system runs in demo mode, generating simulated detections for demonstration purposes. This allows you to:

- Test the inference pipeline
- Understand the output format
- Visualize results and reports
- Validate the system setup

## Data Collection Guidelines

For best results when collecting your own data:

- **Lighting**: Ensure adequate lighting, avoid extreme shadows
- **Camera Position**: Mount camera to capture cattle at 15-45 degree angle
- **Resolution**: Use at least 720p (1280x720) resolution
- **Frame Rate**: 15-30 FPS is sufficient for behavior analysis
- **Duration**: Longer videos (>5 minutes) provide better statistical analysis
- **Environment**: Include environmental context (water sources, shade areas, brushes)

## Privacy and Ethics

When working with farm data:
- Obtain proper permissions from farm owners
- Ensure animal welfare standards are maintained
- Follow local data protection regulations
- Consider anonymizing location-specific information