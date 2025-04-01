import pandas as pd
from pipelines.features import feature_pipeline
from pipelines.preprocessing import preprocess_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_pipeline():
    preprocess = preprocess_pipeline()
    features = feature_pipeline()

    columns_to_drop = [
        # ------ Extraneous Features ------
        "latitude",
        "longitude",
        "venue",
        "venue_id",
        "home_team",
        "away_team",
        "timezone",
        # ------ Genned Columns ------
        "team_away",
        "previous_game",
        # ------ Collinear Data ------
        "home_q1",
        "home_q2",
        "home_q3",
        "home_q4",
        "home_h1",
        "home_h2",
        "home_ot",
        "away_q1",
        "away_q2",
        "away_q3",
        "away_q4",
        "away_h1",
        "away_h2",
        "away_ot",
        "ot",
    ]
    betting_cols = ["min_ou", "max_ou"]

    drop_transformer = ColumnTransformer(
        transformers=[
            ("drop_columns", "drop", columns_to_drop),
            ("drop_betting", "drop", betting_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocess),
            ("features", features),
            ("drop_cols", drop_transformer),
        ]
    )
    return pipeline
