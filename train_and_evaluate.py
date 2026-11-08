"""
IPL Score Prediction Model using Deep Learning (Keras/TensorFlow)
Reconstructed from notebook implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
import joblib

def main():
    # Step 2: Loading dataset
    data_path = 'data/ipl_data.csv'
    ipl = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with shape: {ipl.shape}")

    # Step 3: Data Pre-processing
    drop_cols = ['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker']
    existing_drop_cols = [c for c in drop_cols if c in ipl.columns]
    df = ipl.drop(existing_drop_cols, axis=1)

    X = df.drop(['total'], axis=1)
    y = df['total']

    # Label Encoding for categorical features
    venue_encoder = LabelEncoder()
    batting_team_encoder = LabelEncoder()
    bowling_team_encoder = LabelEncoder()
    striker_encoder = LabelEncoder()
    bowler_encoder = LabelEncoder()

    X['venue'] = venue_encoder.fit_transform(X['venue'])
    X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
    X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
    X['batsman'] = striker_encoder.fit_transform(X['batsman'])
    X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

    # Train Test Split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Define Neural Network
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(216, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])

    huber_loss = tf.keras.losses.Huber(delta=1.0)
    model.compile(optimizer='adam', loss=huber_loss)
    model.summary()

    # Step 5: Model Training
    history = model.fit(
        X_train_scaled, 
        y_train, 
        epochs=50, 
        batch_size=64, 
        validation_data=(X_test_scaled, y_test)
    )

    # Step 6: Model Evaluation
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print("------------------------------------------")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print("------------------------------------------")

    # Save artifacts
    model.save("ipl_score_nn_model.keras")
    joblib.dump({
        'venue_encoder': venue_encoder,
        'batting_team_encoder': batting_team_encoder,
        'bowling_team_encoder': bowling_team_encoder,
        'striker_encoder': striker_encoder,
        'bowler_encoder': bowler_encoder,
        'scaler': scaler
    }, "preprocessors.joblib")
    print("Model and encoders saved successfully.")

if __name__ == '__main__':
    main()
