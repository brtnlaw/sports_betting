import pandas as pd


class Feature:
    """Base class that handles different types of features to be built."""

    def __init__(self, X: pd.DataFrame):
        self.X = X

    def transform(self) -> pd.DataFrame:
        """This method should be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement the transform method.")

    def rolling_feature(
        self, new_col, home_col, away_col, window, min_periods=1, agg_func="mean"
    ):
        # new_col is the name of the new feature i.e. offensive points, etc.
        """Takes a desired set of columns and returns a rolling window for both the home and away team."""
        assert (
            home_col in self.X.columns and away_col in self.X.columns
        ), f"Invalid column, {home_col} and/or {away_col} not in column list."
        # For defense, need to flip home and visitor for the col
        home_df = self.X[
            [
                "start_date",
                "home_team",
                home_col,
            ]
        ].rename(
            columns={
                "home_team": "team",
                home_col: new_col,
            }
        )
        away_df = self.X[["start_date", "away_team", away_col]].rename(
            columns={
                "away_team": "team",
                away_col: new_col,
            }
        )
        game_df = pd.concat([home_df, away_df])
        game_df.sort_values(by=["team", "start_date"], inplace=True)
        game_df.drop_duplicates(
            subset=["start_date", "team"], keep="last", inplace=True
        )
        func_map = {
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
            "max": lambda x: x.max(),
            "min": lambda x: x.min(),
        }

        rolling_col_name = f"rolling_{window}_{agg_func}_{new_col}"
        game_df[rolling_col_name] = (
            game_df.groupby("team")[new_col]
            .shift(1)
            .rolling(window=window, min_periods=min_periods)
            .apply(func_map[agg_func], raw=True)
            .reset_index(level=0, drop=True)
        )
        self.X = (
            self.X.reset_index()
            .merge(
                game_df[["start_date", "team", rolling_col_name]],
                left_on=["start_date", "home_team"],
                right_on=["start_date", "team"],
                how="left",
            )
            .set_index("id")
        )
        self.X = self.X.rename(
            columns={rolling_col_name: f"home_{rolling_col_name}"}
        ).drop(columns=["team"])

        self.X = (
            self.X.reset_index()
            .merge(
                game_df[["start_date", "team", rolling_col_name]],
                left_on=["start_date", "away_team"],
                right_on=["start_date", "team"],
                how="left",
            )
            .set_index("id")
        )
        self.X = self.X.rename(
            columns={rolling_col_name: f"away_{rolling_col_name}"}
        ).drop(columns=["team"])


class OffensiveFeatures(Feature):
    # TODO: Kalman filter for scoring in each quarter
    # TODO: Percent of yards come from rushing/passing - both Offense and Defense
    # TODO: Different time windows
    def rolling_points_for(self):
        self.rolling_feature("points_for", "home_points", "away_points", window=3)
        self.rolling_feature("points_for", "home_points", "away_points", window=5)

    def transform(self):
        self.rolling_points_for()
        return self.X


class DefensiveFeatures(Feature):
    def rolling_points_against(self):
        self.rolling_feature("points_against", "away_points", "home_points", window=3)
        self.rolling_feature("points_against", "away_points", "home_points", window=5)

    def transform(self):
        self.rolling_points_against()
        return self.X


class NeutralSiteFeatures(Feature):
    # TODO: is bowl game?
    pass


class FeaturePipeline:
    """
    Orchestrates the feature engineering process in order.
    """

    def __init__(self, X: pd.DataFrame):
        """
        Provides the steps for the pipeline.

        Args:
            X (pd.DataFrame): Data and venue merged DataFrame.
        """
        self.X = X
        self.steps = [
            OffensiveFeatures,
            # DefensiveFeatures,
        ]
        # At the end, get rid of any categorical variables. Until then, we need columns like "team", etc.

    def engineer_features(self) -> pd.DataFrame:
        """
        Executes all feature engineering steps.
        """
        for step in self.steps:
            feature_step = step(self.X)
            self.X = feature_step.transform()
        # Any remaining categorical columns are dropped.
        # TODO: Update
        categorical_cols = [
            col for col in self.X.select_dtypes(include=["object", "category"]).columns
        ] + ["home_points", "away_points", "longitude", "latitude"]
        self.X.drop(columns=categorical_cols, inplace=True)
        return self.X
