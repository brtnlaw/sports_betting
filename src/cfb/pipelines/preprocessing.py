import datetime as dt

import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

set_config(transform_output="pandas")


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


class RemoveNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns is not None:
            X_ = X.dropna(subset=self.columns).copy()
        else:
            X_ = X.dropna().copy()
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
    col_transformers = ColumnTransformer(
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
                "encode_classification",
                OneHotEncoder(sparse_output=False),
                ["home_classification", "away_classification"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        [
            ("imputation", col_transformers),
            (
                "remove_nans",
                RemoveNaNTransformer(["home_line_scores", "away_line_scores"]),
            ),
            ("quarter_total", QuartersTotalTransformer()),
        ]
    )
    return pipeline
