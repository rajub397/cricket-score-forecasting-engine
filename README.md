# Cricket Score Forecasting Engine

End-to-end IPL 1st-innings score forecasting pipeline using Scikit-Learn Random Forest, match-grouped validation (GroupShuffleSplit), and leak-free ColumnTransformer encoding.

## Files
- data/ipl_data.csv: Historical IPL dataset
- train_and_evaluate.py: Model training and evaluation (MAE, RMSE, R2)
- predict.py: CLI live-match score inference
- requirements.txt: Python dependencies

## Usage
1. pip install -r requirements.txt
2. python train_and_evaluate.py
3. python predict.py
