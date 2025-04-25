import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExpandEfficiencyTransformer(BaseEstimator, TransformerMixin):
    """Expands efficiency columns into attempts and successes. Hardcodes to simplify naming."""

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Expands the efficiency column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        sides = ["home", "away"]
        for side in sides:
            # 3rd down
            X_[[f"{side}_third_down_successes", f"{side}_third_down_attempts"]] = (
                X_[f"{side}_third_down_eff"]
                .where(X_[f"{side}_third_down_eff"].str.contains(r"^\d+-\d+$"))
                .str.split("-", expand=True)
                .astype(float)
            )

            # 4th down
            X_[[f"{side}_fourth_down_successes", f"{side}_fourth_down_attempts"]] = (
                X_[f"{side}_fourth_down_eff"]
                .where(X_[f"{side}_fourth_down_eff"].str.contains(r"^\d+-\d+$"))
                .str.split("-", expand=True)
                .astype(float)
            )

            # Passing attempts
            X_[[f"{side}_receptions", f"{side}_passes"]] = (
                X_[f"{side}_completion_attempts"]
                .where(X_[f"{side}_completion_attempts"].str.contains(r"^\d+-\d+$"))
                .str.split("-", expand=True)
                .astype(float)
            )

            # Penalties
            X_[[f"{side}_penalties", f"{side}_penalty_yds"]] = (
                X_[f"{side}_total_penalties_yards"]
                .where(X_[f"{side}_total_penalties_yards"].str.contains(r"^\d+-\d+$"))
                .str.split("-", expand=True)
                .astype(float)
            )
        return X_
