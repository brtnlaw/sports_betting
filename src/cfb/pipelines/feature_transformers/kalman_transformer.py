import numpy as np
import pandas as pd
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from sklearn.base import BaseEstimator, TransformerMixin


class KalmanTransformer(BaseEstimator, TransformerMixin):
    """Applies a simple Kalman Filter for temporal data."""

    def __init__(
        self,
        new_col: str,
        home_col: str,
        away_col: str,
    ):
        """
        Initializes class to generate Kalman filter column.

        Args:
            new_col (str): Name of the new column.
            home_col (str): Data representing home team to apply filter.
            away_col (str): Data representing away team to apply filter.
        """
        self.new_col = new_col
        self.home_col = home_col
        self.away_col = away_col

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def _apply_kalman_filter(self, series: pd.Series) -> pd.Series:
        """
        Helper function to apply a simple 1D Kalman filter to a series.

        Args:
            series (pd.Series): Noisy data with signal.

        Returns:
            pd.Series: Kalman filter applied signal data.
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        # Location and velocity
        kf.x = np.array([0.0, 0.0])
        # State change matrix
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        # Measurement function
        kf.H = np.array([[1.0, 0.0]])
        # Covariance matrix
        kf.P *= 10.0
        # Measurement noise
        kf.R = 1e-5
        # Process noise, assumed Gaussian
        diffs = series.diff().dropna()
        # Pare down the estimated variance
        var_hat = diffs.var() * 0.6
        kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=var_hat)

        estimates = []
        for value in series:
            if pd.notnull(value):
                kf.predict()
                estimates.append(kf.x[0])
                kf.update(np.array([[value]]))
            else:
                estimates.append(np.nan)
        return pd.Series(estimates, index=series.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Kalman filter to column in DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with filtered values.
        """
        X_ = X.copy()

        # Prepare data for Kalman filtering
        home_df = X_[["start_date", "home_team", self.home_col]].rename(
            columns={"home_team": "team", self.home_col: self.new_col}
        )
        away_df = X_[["start_date", "away_team", self.away_col]].rename(
            columns={"away_team": "team", self.away_col: self.new_col}
        )

        game_df = pd.concat([home_df, away_df])
        game_df.sort_values(by=["team", "start_date"], inplace=True)
        game_df.drop_duplicates(
            subset=["start_date", "team"], keep="last", inplace=True
        )

        # Maintain kalman somewhere in name
        if self.new_col.startswith("kalman"):
            kalman_col_name = self.new_col
        else:
            kalman_col_name = f"kalman_{self.new_col}"
        game_df[kalman_col_name] = game_df.groupby("team")[self.new_col].transform(
            self._apply_kalman_filter
        )

        X_ = (
            X_.reset_index()
            .merge(
                game_df[["start_date", "team", kalman_col_name]],
                left_on=["start_date", "home_team"],
                right_on=["start_date", "team"],
                how="left",
            )
            .rename(columns={kalman_col_name: f"home_{kalman_col_name}"})
            .drop(columns=["team"])
            .set_index(X_.index.name)
        )

        X_ = (
            X_.reset_index()
            .merge(
                game_df[["start_date", "team", kalman_col_name]],
                left_on=["start_date", "away_team"],
                right_on=["start_date", "team"],
                how="left",
            )
            .rename(columns={kalman_col_name: f"away_{kalman_col_name}"})
            .drop(columns=["team"])
            .set_index(X_.index.name)
        )
        return X_
