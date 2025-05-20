from pipelines.feature_transformers.efficiency_transformer import EfficiencyTransformer
from pipelines.feature_transformers.expand_efficiency_transformer import (
    ExpandEfficiencyTransformer,
)
from pipelines.feature_transformers.group_mean_imputer import GroupMeanImputer
from pipelines.feature_transformers.group_mode_imputer import GroupModeImputer
from pipelines.feature_transformers.multiple_value_imputer import MultipleValueImputer
from pipelines.feature_transformers.net_transformer import NetTransformer
from pipelines.feature_transformers.quarters_total_transformer import (
    QuartersTotalTransformer,
)
from pipelines.feature_transformers.spread_transformer import SpreadTransformer
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

set_config(transform_output="pandas")


def get_preprocess_pipeline() -> Pipeline:
    """
    Generates the entire preprocessing pipeline.

    Returns:
        Pipeline: Preprocessing pipeline.
    """
    col_transformers_1 = ColumnTransformer(
        transformers=[
            (
                "impute_attendance",
                GroupMeanImputer("venue"),
                ["venue", "attendance"],
            ),
            (
                "impute_home_elo",
                GroupMeanImputer("home_team"),
                ["home_team", "home_pregame_elo"],
            ),
            (
                "impute_away_elo",
                GroupMeanImputer("away_team"),
                ["away_team", "away_pregame_elo"],
            ),
            (
                "impute_nan_grass_dome",
                MultipleValueImputer([float("nan"), None], False),
                ["dome", "grass"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    col_transformers_2 = ColumnTransformer(
        transformers=[
            (
                "impute_conf_class_home",
                GroupModeImputer("home_team"),
                ["home_team", "home_conference", "home_classification"],
            ),
            (
                "impute_conf_class_away",
                GroupModeImputer("away_team"),
                ["away_team", "away_conference", "away_classification"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    col_transformers_3 = ColumnTransformer(
        transformers=[
            (
                "encode_classification",
                OneHotEncoder(sparse_output=False),
                ["home_classification", "away_classification", "season_type"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    col_transformers_4 = ColumnTransformer(
        transformers=[
            (
                "home_passing_efficiency",
                EfficiencyTransformer("home_receptions", "home_passes"),
                ["home_receptions", "home_passes"],
            ),
            (
                "away_passing_efficiency",
                EfficiencyTransformer("away_receptions", "away_passes"),
                ["away_receptions", "away_passes"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        [
            (
                "set_id_index",
                FunctionTransformer(
                    lambda df: df.set_index("id"),
                    validate=False,
                ),
            ),
            # NOTE: The below is okay because we do this before we separate X and y.
            (
                "remove_nans",
                FunctionTransformer(
                    lambda df: df.dropna(
                        subset=["home_line_scores", "away_line_scores"]
                    ),
                    validate=False,
                ),
            ),
            ("col_transformers_1", col_transformers_1),
            ("col_transformers_2", col_transformers_2),
            ("col_transformers_3", col_transformers_3),
            ("quarter_total", QuartersTotalTransformer()),
            ("spread", SpreadTransformer()),
            ("expand_efficiency", ExpandEfficiencyTransformer()),
            ("col_transformers_4", col_transformers_4),
        ]
    )
    return pipeline
