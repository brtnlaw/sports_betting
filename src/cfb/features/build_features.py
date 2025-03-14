import pandas as pd


class Feature:
    """
    Base class that handles different types of features to be built.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def transform(self) -> pd.DataFrame:
        """
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Each subclass must implement the transform method.")


class OffensiveFeatures(Feature):
    # TODO: Kalman filter for scoring in each quarter
    # TODO: Percent of yards come from rushing/passing - both Offense and Defense
    # TODO: Different time windows
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

    def __init__(self, df: pd.DataFrame):
        """
        Provides the steps for the pipeline.

        Args:
            df (pd.DataFrame): Data and venue merged DataFrame.
        """
        self.df = df
        self.steps = [
            Preprocessing(df),
            # OffensiveFeatures(df),
            # DefensiveFeatures(df),
            # NeutralSiteFeatures(df),
        ]

    def run(self) -> pd.DataFrame:
        """
        Executes all feature engineering steps.
        """
        for step in self.steps:
            self.df = step.transform()
        return self.df
