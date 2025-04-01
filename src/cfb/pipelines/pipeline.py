import pandas as pd
from preprocessing import preprocess_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocess = preprocess_pipeline()
columns_to_drop = [
    # ------ Extraneous Features ------
    "latitude",
    "longitude",
    # ------ Duplicate Columns ------
    "team",
    "team_away",
    "previous_game",
    "previous_game_home",
    "previous_game_away",
    "venue",
    "venue_id",
    # ------ Score Data ------
    "home_q1",
    "home_q2",
    "home_q3",
    "home_q4",
    "home_h1",
    "home_h2",
    "home_total",
    "home_ot",
    "away_q1",
    "away_q2",
    "away_q3",
    "away_q4",
    "away_h1",
    "away_h2",
    "away_total",
    "away_ot",
]
betting_cols = ["min_ou", "max_ou"]

drop_transformer = ColumnTransformer(
    transformers=[
        ("drop_columns", "drop", columns_to_drop),
        ("drop_betting", "drop", betting_cols),
    ],
    verbose_feature_names_out=False,
)
pipeline = Pipeline(
    steps=[("preprocessing", preprocess), (), ("drop_cols", drop_transformer)]
)
