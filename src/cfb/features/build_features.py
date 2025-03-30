import pandas as pd


class Feature:
    """Base class that handles different types of features to be built."""

    def __init__(self, X: pd.DataFrame):
        self.X = X

    def transform(self) -> pd.DataFrame:
        """This method should be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement the transform method.")


class OffensiveFeatures(Feature):
    # TODO: Kalman filter for scoring in each quarter
    # TODO: Percent of yards come from rushing/passing - both Offense and Defense
    # TODO: Different time windows
    def rolling_points():
        pass

    pass


class DefensiveFeatures(Feature):
    pass


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
            # OffensiveFeatures(df),
            # DefensiveFeatures(df),
            # NeutralSiteFeatures(df),
        ]
        # At the end, get rid of any categorical variables. Until then, we need columns like "team", etc.

    def engineer_features(self) -> pd.DataFrame:
        """
        Executes all feature engineering steps.
        """
        for step in self.steps:
            self.X = step.transform()
        # Any remaining categorical columns are dropped.
        categorical_cols = [
            col for col in self.X.select_dtypes(include=["object", "category"]).columns
        ]
        self.X.drop(columns=categorical_cols, inplace=True)
        return self.X
