from lightgbm.sklearn import LGBMRegressor
from pipelines.features import feature_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_features_and_model_pipeline() -> Pipeline:
    """
    Final composition that takes preprocessed data and feature engineers into a model.

    Returns:
        Pipeline: Composed feature engineering and model pipeline.
    """
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
        "start_date",
        "home_conference",
        "home_line_scores",
        "away_conference",
        "away_line_scores",
        "ot",
        "excitement_index",
        # ------ Genned Columns ------
        "team_away",
        "previous_game",
        # ------ Collinear Data ------
        "home_points",
        "away_points",
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
        "home_postgame_win_probability",
        "away_postgame_win_probability",
    ]

    drop_transformer = ColumnTransformer(
        transformers=[
            ("drop_columns", "drop", columns_to_drop),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("features", features),
            ("drop_cols", drop_transformer),
            ("light_gbm", LGBMRegressor(verbose=-1)),
        ]
    )
    return pipeline
