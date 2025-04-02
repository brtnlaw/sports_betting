import datetime as dt

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

set_config(transform_output="pandas")


class MultipleValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, null_values, impute_val):
        self.null_values = null_values
        self.impute_val = impute_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_.replace(self.null_values, self.impute_val, inplace=True)
        return X_


class GroupMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col):
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in X_.columns:
            if col != self.group_col:
                # Gets the mean by group
                X_.loc[(X_[col].isna()) & X_[self.group_col].notna(), col] = X_[
                    self.group_col
                ].map(X_.groupby(self.group_col)[col].mean())
                # Fills in the rest if not available
                X_[col] = X_[col].fillna(X_[col].mean())
        return X_


class GroupModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col):
        self.group_col = group_col

    def fit(self, X, y=None):
        return self  # No fitting needed, just transformation

    def transform(self, X):
        X_ = X.copy()
        for col in X_.columns:
            if col != self.group_col:
                mode_df = (
                    X_.groupby(self.group_col)[col]
                    .apply(lambda group: group.mode())
                    .reset_index(level=0)
                )
                mode_dict = dict(zip(mode_df[self.group_col], mode_df[col]))
                X_[col] = X_.apply(
                    lambda row: mode_dict.get(row[self.group_col], row[col]), axis=1
                )
        return X_


class QuartersTotalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_.dropna(subset=["home_line_scores"], inplace=True)
        X_ = X_[
            X_["home_line_scores"].apply(lambda x: len(x) >= 4)
            & X_["away_line_scores"].apply(lambda x: len(x) >= 4)
        ]
        X_["ot"] = X_["home_line_scores"].apply(lambda x: int(len(x) > 4))

        for prefix in ["home", "away"]:
            line_score = f"{prefix}_line_scores"
            X_[line_score] = X_[line_score].apply(
                lambda x: x[:4] + [sum(x[5:])] if len(x) >= 5 else x + [0]
            )
            X_[
                [
                    f"{prefix}_q1",
                    f"{prefix}_q2",
                    f"{prefix}_q3",
                    f"{prefix}_q4",
                    f"{prefix}_ot",
                ]
            ] = pd.DataFrame(X_[line_score].tolist(), index=X_.index)

            X_[f"{prefix}_h1"] = X_[f"{prefix}_q1"] + X_[f"{prefix}_q2"]
            X_[f"{prefix}_h2"] = X_[f"{prefix}_q3"] + X_[f"{prefix}_q4"]
        X_["total"] = X_["home_points"] + X_["away_points"]
        return X_


def preprocess_pipeline():
    col_transformers_1 = ColumnTransformer(
        transformers=[
            ("impute_attendance", GroupMeanImputer("venue"), ["venue", "attendance"]),
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
        ]
    )
    return pipeline
