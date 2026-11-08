# IPL Match Score Prediction Engine

An end-to-end Deep Learning regression system that predicts the first-innings total score of an IPL match based on match conditions (venue, batting team, bowling team, batsman, and bowler).

## Repository Architecture

```text
├── data/
│   └── ipl_data.csv          # Pre-packaged full IPL historical dataset
├── train_and_evaluate.py     # End-to-end data pipeline, ANN model training & evaluation
├── predict.py                # Standalone inference module for custom match queries
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules for virtual environments & caches
└── README.md                 # Complete documentation
```

## Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model
The dataset is bundled inside `data/ipl_data.csv`. You can run training immediately:
```bash
python train_and_evaluate.py
```

### 3. Run Predictions
```bash
python predict.py
```

## Model Architecture
- **Input Features (5):** Venue, Batting Team, Bowling Team, Batsman, Bowler
- **Layer 1:** Dense (512 units, ReLU activation)
- **Layer 2:** Dense (216 units, ReLU activation)
- **Output:** Dense (1 unit, Linear activation)
- **Loss:** Huber Loss (`delta=1.0`)
- **Optimizer:** Adam
