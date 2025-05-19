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
                "rolling_points_for",
                RollingTransformer(
                    "rolling_points_for",
                    "home_points",
                    "away_points",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_third_down_attempts",
                RollingTransformer(
                    "rolling_third_down_attempts",
                    "home_third_down_attempts",
                    "away_third_down_attempts",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_third_down_successes",
                RollingTransformer(
                    "rolling_third_down_successes",
                    "home_third_down_successes",
                    "away_third_down_successes",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_fourth_down_attempts",
                RollingTransformer(
                    "rolling_fourth_down_attempts",
                    "home_fourth_down_attempts",
                    "away_fourth_down_attempts",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_fourth_down_successes",
                RollingTransformer(
                    "rolling_fourth_down_successes",
                    "home_fourth_down_successes",
                    "away_fourth_down_successes",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "kalman_points_for",
                KalmanTransformer(
                    "kalman_points_for",
                    "home_points",
                    "away_points",
                ),
            ),
            (
                "rolling_plays_40_plus_for",
                RollingTransformer(
                    "rolling_plays_40_plus_for",
                    "home_plays_40_plus",
                    "away_plays_40_plus",
                    [1, 3, 5],
                    1,
                    "mean",
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
                "rolling_points_against",
                RollingTransformer(
                    "rolling_points_against",
                    "away_points",
                    "home_points",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_passing_yds_given_up",
                RollingTransformer(
                    "rolling_passing_yds_given_up",
                    "away_net_passing_yards",
                    "home_net_passing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "kalman_points_against",
                KalmanTransformer(
                    "kalman_points_against",
                    "away_points",
                    "home_points",
                ),
            ),
        ]
    )
    return defense_pipeline


def pass_game_pipeline() -> Pipeline:
    pass_game_pipeline = Pipeline(
        [
            (
                "rolling_passing_yds_for",
                RollingTransformer(
                    "rolling_passing_yds_for",
                    "home_net_passing_yards",
                    "away_net_passing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_ints_thrown",
                RollingTransformer(
                    "rolling_ints_thrown",
                    "home_interceptions",
                    "away_interceptions",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_passing_tds",
                RollingTransformer(
                    "rolling_passing_tds",
                    "home_passing_tds",
                    "away_passing_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_receptions_efficiency",
                RollingTransformer(
                    "rolling_receptions_efficiency",
                    "home_receptions_efficiency",
                    "away_receptions_efficiency",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
        ]
    )
    return pass_game_pipeline


def run_game_pipeline() -> Pipeline:
    run_game_pipeline = Pipeline(
        [
            (
                "rolling_rushing_tds",
                RollingTransformer(
                    "rolling_rushing_tds",
                    "home_rushing_tds",
                    "away_rushing_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_rushing_yds_for",
                RollingTransformer(
                    "rolling_rushing_yds_for",
                    "home_rushing_yards",
                    "away_rushing_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
        ]
    )
    return run_game_pipeline


def special_pipeline() -> Pipeline:
    special_pipeline = Pipeline(
        [
            (
                "rolling_punt_yds_for",
                RollingTransformer(
                    "rolling_punt_yds_for",
                    "home_punt_return_yards",
                    "away_punt_return_yards",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
            (
                "rolling_punt_tds_for",
                RollingTransformer(
                    "rolling_punt_tds_for",
                    "home_punt_return_tds",
                    "away_punt_return_tds",
                    [1, 3, 5],
                    1,
                    "mean",
                ),
            ),
        ]
    )
    return special_pipeline


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
            ("pass_game_pipeline", pass_game_pipeline()),
            ("run_game_pipeline", run_game_pipeline()),
        ]
    )
    return pipeline
