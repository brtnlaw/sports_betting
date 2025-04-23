from pipelines.feature_transformers.days_since_last_game_transformer import (
    DaysSinceLastGameTransformer,
)
from pipelines.feature_transformers.kalman_transformer import KalmanTransformer
from pipelines.feature_transformers.rolling_transformer import RollingTransformer
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")


def offense_pipeline() -> Pipeline:
    """
    Pipeline for all offensive features.

    Returns:
        Pipeline: Pipeline with offensive features.
    """
    offense_pipeline = Pipeline(
        [
            (
                "rolling_offense_1_3_5",
                RollingTransformer(
                    "points_for", "home_points", "away_points", [1, 3, 5], 1, "mean"
                ),
            ),
            (
                "rolling_third_down_attempts_1_3_5",
                RollingTransformer(
                    "third_down_attempts",
                    "home_third_down_attempts",
                    "away_third_down_attempts",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_third_down_successes_1_3_5",
                RollingTransformer(
                    "third_down_successes",
                    "home_third_down_successes",
                    "away_third_down_successes",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_fourth_down_attempts_1_3_5",
                RollingTransformer(
                    "fourth_down_attempts",
                    "home_fourth_down_attempts",
                    "away_fourth_down_attempts",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_fourth_down_successes_1_3_5",
                RollingTransformer(
                    "fourth_down_successes",
                    "home_fourth_down_successes",
                    "away_fourth_down_successes",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_passing_yds_for_1_3_5",
                RollingTransformer(
                    "passing_yds_for",
                    "home_net_passing_yards",
                    "away_net_passing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_ints_thrown_1_3_5",
                RollingTransformer(
                    "ints_thrown",
                    "home_interceptions",
                    "away_interceptions",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_rushing_yds_for_1_3_5",
                RollingTransformer(
                    "rushing_yds_for",
                    "home_rushing_yards",
                    "away_rushing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_passing_tds_1_3_5",
                RollingTransformer(
                    "passing_tds",
                    "home_passing_tds",
                    "away_passing_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_rushing_tds_1_3_5",
                RollingTransformer(
                    "rushing_tds",
                    "home_rushing_tds",
                    "away_rushing_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "kalman_offense",
                KalmanTransformer(
                    "points_for",
                    "home_points",
                    "away_points",
                ),
            ),
        ]
    )
    return offense_pipeline


def defense_pipeline() -> Pipeline:
    """
    Pipeline for all defensive features.

    Returns:
        Pipeline: Pipeline with defensive features.
    """
    defense_pipeline = Pipeline(
        [
            (
                "rolling_defense_1_3_5",
                RollingTransformer(
                    "points_against", "away_points", "home_points", [1, 3, 5], 1, "mean"
                ),
            ),
            (
                "rolling_passing_yds_given_up_1_3_5",
                RollingTransformer(
                    "passing_yds_given_up",
                    "away_net_passing_yards",
                    "home_net_passing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "kalman_defense",
                KalmanTransformer(
                    "points_against",
                    "away_points",
                    "home_points",
                ),
            ),
        ]
    )
    return defense_pipeline


def feature_pipeline() -> Pipeline:
    """
    Combines all types of features into one pipeline.

    Returns:
        Pipeline: Combined pipeline of features.
    """
    pipeline = Pipeline(
        [
            ("days_since", DaysSinceLastGameTransformer()),
            ("offense_pipeline", offense_pipeline()),
            ("defense_pipeline", defense_pipeline()),
        ]
    )
    return pipeline
