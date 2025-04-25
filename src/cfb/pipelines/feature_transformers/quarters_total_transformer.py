import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuartersTotalTransformer(BaseEstimator, TransformerMixin):
    """Expands line_scores into quarter and total columns."""

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Expands the line_scores columns.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
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
