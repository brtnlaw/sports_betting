from lightgbm.sklearn import LGBMRegressor
from pipelines.feature_transformers.print_transformer import PrintTransformer
from pipelines.features import feature_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, VarianceThreshold
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
        "constructionyear",
        "elevation",
        "capacity",
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
        "home_postgame_elo",
        "away_postgame_elo",
        # ------ Future Looking Data ------
        "home_game_id",
        "home_team_id",
        "home_completion_attempts",  # Not an int
        "home_first_downs",
        "home_fourth_down_eff",
        "home_fumbles_lost",
        "home_fumbles_recovered",
        "home_interception_tds",
        "home_interception_yards",
        "home_interceptions",
        "home_kick_return_tds",
        "home_kick_return_yards",
        "home_kick_returns",
        "home_kicking_points",
        "home_net_passing_yards",
        "home_passes_intercepted",
        "home_passing_tds",
        "home_possession_time",
        "home_rushing_attempts",
        "home_rushing_tds",
        "home_rushing_yards",
        "home_third_down_eff",
        "home_total_penalties_yards",
        "home_total_yards",
        "home_turnovers",
        "home_yards_per_pass",
        "home_yards_per_rush_attempt",
        "home_punt_return_tds",
        "home_punt_return_yards",
        "home_punt_returns",
        "away_game_id",
        "away_team_id",
        "away_completion_attempts",
        "away_first_downs",
        "away_fourth_down_eff",
        "away_fumbles_lost",
        "away_fumbles_recovered",
        "away_interception_tds",
        "away_interception_yards",
        "away_interceptions",
        "away_kick_return_tds",
        "away_kick_return_yards",
        "away_kick_returns",
        "away_kicking_points",
        "away_net_passing_yards",
        "away_passes_intercepted",
        "away_passing_tds",
        "away_possession_time",
        "away_rushing_attempts",
        "away_rushing_tds",
        "away_rushing_yards",
        "away_third_down_eff",
        "away_total_penalties_yards",
        "away_total_yards",
        "away_turnovers",
        "away_yards_per_pass",
        "away_yards_per_rush_attempt",
        "away_punt_return_tds",
        "away_punt_return_yards",
        "away_punt_returns",
        "home_third_down_successes",
        "home_third_down_attempts",
        "home_fourth_down_successes",
        "home_fourth_down_attempts",
        "home_receptions",
        "home_passes",
        "home_penalties",
        "home_penalty_yds",
        "away_third_down_successes",
        "away_third_down_attempts",
        "away_fourth_down_successes",
        "away_fourth_down_attempts",
        "away_receptions",
        "away_passes",
        "away_penalties",
        "away_penalty_yds",
    ]

    drop_transformer = ColumnTransformer(
        transformers=[
            ("drop_columns", "drop", columns_to_drop),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    lgbm_params = {"verbose": -1, "reg_lambda": 1, "reg_alpha": 1}
    pipeline = Pipeline(
        steps=[
            ("features", features),
            ("drop_cols", drop_transformer),
            ("variance_threshold", VarianceThreshold()),
            ("print_rfecv", PrintTransformer("Starting RFECV...")),
            (
                "recurs_feature_elimination_cv",
                RFECV(
                    LGBMRegressor(**lgbm_params),
                    min_features_to_select=5,
                    step=10,
                    scoring="neg_mean_squared_error",
                ),
            ),
            ("print_fit", PrintTransformer("Fitting model...")),
            ("light_gbm_regressor", LGBMRegressor(**lgbm_params)),
        ]
    )
    return pipeline
