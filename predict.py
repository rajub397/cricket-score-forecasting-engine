"""
Inference script to predict final match score based on current match situation.
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

def predict_score(venue, bat_team, bowl_team, batsman, bowler):
    try:
        model = tf.keras.models.load_model("ipl_score_nn_model.keras")
        preprocessors = joblib.load("preprocessors.joblib")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        print("Please train the model first by running `python train_and_evaluate.py`.")
        return

    try:
        v_enc = preprocessors['venue_encoder'].transform([venue])[0]
        bat_enc = preprocessors['batting_team_encoder'].transform([bat_team])[0]
        bowl_enc = preprocessors['bowling_team_encoder'].transform([bowl_team])[0]
        striker_enc = preprocessors['striker_encoder'].transform([batsman])[0]
        bowler_enc = preprocessors['bowler_encoder'].transform([bowler])[0]
    except ValueError as ve:
        print(f"Categorical encoding error: {ve}")
        return

    input_df = pd.DataFrame([{
        'venue': v_enc,
        'bat_team': bat_enc,
        'bowl_team': bowl_enc,
        'batsman': striker_enc,
        'bowler': bowler_enc
    }])

    scaled_input = preprocessors['scaler'].transform(input_df)
    predicted_score = model.predict(scaled_input, verbose=0)[0][0]
    print(f"\nMatch Conditions:")
    print(f"  Venue:        {venue}")
    print(f"  Batting Team: {bat_team}")
    print(f"  Bowling Team: {bowl_team}")
    print(f"  Striker:      {batsman}")
    print(f"  Bowler:       {bowler}")
    print(f"\nPredicted 1st Innings Total Score: {round(float(predicted_score))} runs")

if __name__ == "__main__":
    predict_score(
        venue="M Chinnaswamy Stadium",
        bat_team="Kolkata Knight Riders",
        bowl_team="Royal Challengers Bangalore",
        batsman="BB McCullum",
        bowler="P Kumar"
    )
