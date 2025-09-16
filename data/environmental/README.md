# Environmental Data

This directory contains environmental monitoring data that can be correlated with cattle behavior analysis results.

## Data Files

### `sample_temperature_humidity.csv`
Example environmental monitoring data showing:
- **Timestamp**: Date and time of measurements
- **Temperature (°C)**: Ambient temperature readings
- **Humidity (%)**: Relative humidity measurements  
- **Heat Index**: Calculated heat stress index
- **Farm ID**: Identifier for the farm location
- **Sensor Location**: Specific sensor placement (barn, pasture, etc.)

## Data Collection

Environmental data should be collected using:
- **Temperature sensors**: Accurate to ±0.5°C
- **Humidity sensors**: Accurate to ±3% RH
- **Data loggers**: Recording at 15-30 minute intervals
- **Weather stations**: For comprehensive environmental monitoring

## Thermal Stress Thresholds

Based on research literature:
- **Mild stress**: 25-28°C with >60% humidity
- **Moderate stress**: 28-32°C with >50% humidity  
- **Severe stress**: >32°C with any humidity level
- **Heat index**: Combination of temperature and humidity effects

## Integration with Behavior Analysis

Environmental data can be correlated with behavior detections using:

```python
from thermotolerance_dairy import CattleBehaviorAnalyzer

# Initialize analyzer with environmental correlation
analyzer = CattleBehaviorAnalyzer(
    environmental_data='data/environmental/sample_temperature_humidity.csv'
)

# Analyze video with environmental context
results = analyzer.analyze_video('video.mp4')
correlation = results.get_environmental_correlation()
```

## Expected Correlations

Research shows strong correlations between environmental conditions and behaviors:

| Temperature Range | Expected Behavior Changes |
|------------------|---------------------------|
| 18-25°C | Normal baseline behaviors |
| 25-28°C | Increased drinking (+15-25%) |
| 28-32°C | Increased brushing (+30-50%), drinking (+40-60%) |
| >32°C | High stress behaviors, reduced grazing (-20-40%) |

## Data Privacy

When collecting farm environmental data:
- Anonymize location-specific information
- Obtain proper permissions from farm owners
- Follow local data protection regulations
- Consider aggregating data across multiple farms