# Traffic Signal Timing ML System

An offline batch ML system that recommends full traffic signal timing plans for intersections along Highway 102 in Bentonville, AR.

## Overview

This system uses historical traffic volume data and signal timing data to predict optimal timing plans that improve Level of Service (LOS). It implements multiple ML approaches and compares them against a HCM2010 deterministic baseline.

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd COB-Traffic-Project

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# 1. Ingest data
cd ml/data && python ingest.py

# 2. Preprocess data
python preprocess.py

# 3. Train models
cd ../models && python train.py

# 4. Generate predictions
python predict.py
```

### Using Docker

```bash
# Build image
docker build -t traffic-ml .

# Run predictions
docker run -v $(pwd)/output:/app/output traffic-ml
```

## Project Structure

```
├── volume/                 # Volume (turning movement count) CSV files
├── times/                  # Phase timing CSV files
├── ml/
│   ├── data/
│   │   ├── ingest.py      # Data ingestion from CSV files
│   │   └── preprocess.py  # Missing value handling, feature extraction
│   ├── models/
│   │   ├── hcm2010.py     # HCM2010 deterministic optimizer
│   │   ├── train.py       # Model training pipeline
│   │   └── predict.py     # Inference pipeline
│   └── los_wrapper.py     # Wrapper around LOS.py
├── tests/                  # Unit tests
├── LOS.py                  # Level of Service calculations
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── README.md              # This file
```

## Data Formats

### Volume Files (volume/*.csv)

15-minute turning movement counts:

```csv
DATE,TIME,INTID,NBL,NBT,NBR,SBL,SBT,SBR,EBL,EBT,EBR,WBL,WBT,WBR
10/11/2025,="0000",6,0,11,0,23,9,5,1,39,*,2,44,*
```

- TIME is in Excel escaped format `="HHMM"`
- Missing values are marked with `*`

### Timing Files (times/*.csv)

Phase timing configurations:

```csv
Phase,,1 EBLT,2WB,3NBLT,4SB,5WBLT,6 EB,7SBLT,8NB,Offset
,25,15,77,15,33,15,77,17,31,125
Yellow Change,,4,4.5,4,4.5,4,4.5,4,4.5,
Red Clearence,,1,1,1,1,1,1,1,1,
```

## Model Approaches

### 1. HCM2010 Deterministic Baseline

Uses Webster's formula for optimal cycle length and proportional green split allocation based on critical volume ratios.

### 2. Gradient Boosted Trees

Predicts green splits for each phase using engineered features:
- Temporal features (hour, day of week, weekend)
- Rolling volume statistics
- Historical timing patterns

### 3. Sequence Model

Uses sliding window approach to capture temporal patterns in traffic volumes for timing prediction.

### 4. Hybrid ML + Optimization

Combines ML predictions with HCM2010 constraints:
1. ML model predicts initial green splits
2. HCM2010 optimizer applies safety constraints
3. LOS validation ensures plan improvement

## Output Format

Predictions are output as NDJSON with one recommendation per line:

```json
{
  "intersection_id": "102_A",
  "timestamp": "2025-10-11T08:00:00",
  "cycle_length": 120.0,
  "phases": [
    {"phase": "1 EBLT", "green": 30.0, "yellow": 4.0, "red_clearance": 1.0}
  ],
  "recommended_change_score": 65.5,
  "notes": ["Estimated delay reduction: 15.5%", "LOS improvement: D -> C"],
  "is_valid": true,
  "los_before": "D",
  "los_after": "C",
  "delay_before": 45.2,
  "delay_after": 38.2
}
```

## Validation and Safety

All recommended plans are validated against:

1. **Cycle length constraints**: 60-180 seconds
2. **Minimum green times**: 7 seconds (pedestrian safety)
3. **Maximum green times**: 90 seconds
4. **Yellow time minimums**: 3 seconds
5. **Red clearance minimums**: 1 second

If a plan violates any constraint or worsens LOS by more than 5%, the system falls back to the HCM2010 baseline.

## Extending the System

### Adding New Models

1. Create a new class in `ml/models/` inheriting from `TimingModel`
2. Implement `fit()`, `predict()`, `save()`, and `load()` methods
3. Add the model to the training pipeline in `train.py`

### Adding New Features

1. Modify `VolumePreprocessor` in `ml/data/preprocess.py`
2. Add feature extraction logic in the appropriate method
3. Update the model training to use new features

### Adding New Intersections

1. Add volume CSV to `volume/` directory
2. Add timing CSV to `times/` directory
3. Files must follow the naming convention: `102_<name>.csv`

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_los.py -v

# Run with coverage
pytest tests/ --cov=ml --cov-report=html
```

## LOS Reference

HCM2010 Level of Service thresholds for signalized intersections:

| LOS | Delay (sec/veh) | Description |
|-----|-----------------|-------------|
| A   | ≤ 10           | Free flow   |
| B   | 10-20          | Stable flow |
| C   | 20-35          | Stable flow |
| D   | 35-55          | Approaching unstable |
| E   | 55-80          | Unstable    |
| F   | > 80           | Forced flow |

## License

Copyright (c) 2025. All rights reserved.
