import datetime as dt

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DaysSinceLastGameTransformer(BaseEstimator, TransformerMixin):
    """Generates the days since last game for each team. If first game, fills since 1/1/2000."""

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates days since last game column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        df_home = X_[["start_date", "home_team"]].rename(columns={"home_team": "team"})
        df_away = X_[["start_date", "away_team"]].rename(columns={"away_team": "team"})
        df_combined = pd.concat([df_home, df_away]).sort_values(
            by=["team", "start_date"]
        )
        df_combined["previous_game"] = df_combined.groupby("team")["start_date"].shift(
            1
        )
        X_["previous_game"] = None
        for side in ["home", "away"]:
            # Maintain the original index through merge by resetting, and then setting again
            X_ = (
                X_.reset_index()
                .merge(
                    df_combined[["team", "start_date", "previous_game"]],
                    left_on=[f"{side}_team", "start_date"],
                    right_on=["team", "start_date"],
                    how="left",
                    suffixes=("", f"_{side}"),
                )
                .set_index(X_.index.name)
            )
            X_[f"previous_game_{side}"] = X_[f"previous_game_{side}"].fillna(
                dt.date(2000, 1, 1)
            )
            X_[f"{side}_days_since_last_game"] = (
                X_["start_date"] - X_[f"previous_game_{side}"]
            ).apply(lambda x: x.days)
        X_.drop(
            columns=["team", "previous_game_home", "previous_game_away"],
            inplace=True,
            errors="ignore",
        )
        return X_
