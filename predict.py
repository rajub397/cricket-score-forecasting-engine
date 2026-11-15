"""
IPL 1st Innings Score Forecasting Engine - Inference Script
Predicts final 1st innings score from live match state and player/venue context.
Handles unseen players or venues gracefully without crashing.
"""

import argparse
import os
import joblib
import pandas as pd

MODEL_ARTIFACT_PATH = "ipl_score_pipeline.joblib"


def predict_score(
    venue: str,
    bat_team: str,
    bowl_team: str,
    batsman: str,
    bowler: str,
    runs: int,
    wickets: int,
    overs: float,
    runs_last_5: int,
    wickets_last_5: int,
    striker: int,
    non_striker: int,
) -> float:
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Model artifact '{MODEL_ARTIFACT_PATH}' not found. "
            "Run `python train_and_evaluate.py` first."
        )

    artifact = joblib.load(MODEL_ARTIFACT_PATH)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    input_row = pd.DataFrame(
        [
            {
                "venue": venue,
                "bat_team": bat_team,
                "bowl_team": bowl_team,
                "batsman": batsman,
                "bowler": bowler,
                "runs": runs,
                "wickets": wickets,
                "overs": overs,
                "runs_last_5": runs_last_5,
                "wickets_last_5": wickets_last_5,
                "striker": striker,
                "non-striker": non_striker,
            }
        ]
    )[feature_columns]

    predicted_total = float(pipeline.predict(input_row)[0])

    print("\n=== Live Match State ===")
    print(f"Venue:           {venue}")
    print(f"Matchup:         {bat_team} vs {bowl_team}")
    print(f"Score:           {runs}/{wickets} in {overs} overs")
    print(f"Last 5 Overs:    {runs_last_5} runs, {wickets_last_5} wickets")
    print(f"Crease / Bowler: {batsman} ({striker}*) & non-striker ({non_striker}) vs {bowler}")
    print("========================")
    print(f"Projected 1st Innings Total: {round(predicted_total)} runs ({predicted_total:.2f})")

    return predicted_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IPL 1st innings final score.")
    parser.add_argument("--venue", default="Eden Gardens")
    parser.add_argument("--bat_team", default="Kolkata Knight Riders")
    parser.add_argument("--bowl_team", default="Royal Challengers Bangalore")
    parser.add_argument("--batsman", default="BB McCullum")
    parser.add_argument("--bowler", default="P Kumar")
    parser.add_argument("--runs", type=int, default=112)
    parser.add_argument("--wickets", type=int, default=2)
    parser.add_argument("--overs", type=float, default=12.4)
    parser.add_argument("--runs_last_5", type=int, default=48)
    parser.add_argument("--wickets_last_5", type=int, default=1)
    parser.add_argument("--striker", type=int, default=54)
    parser.add_argument("--non_striker", type=int, default=28)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_score(
        venue=args.venue,
        bat_team=args.bat_team,
        bowl_team=args.bowl_team,
        batsman=args.batsman,
        bowler=args.bowler,
        runs=args.runs,
        wickets=args.wickets,
        overs=args.overs,
        runs_last_5=args.runs_last_5,
        wickets_last_5=args.wickets_last_5,
        striker=args.striker,
        non_striker=args.non_striker,
    )
