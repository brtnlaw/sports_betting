import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Optional


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


class Preprocessing(Feature):
    """
    Handles preprocessing steps such as missing values, scaling, and encoding.

    Inherits from:
        Feature: The base feature engineering class.
    """

    def remove_nan_rows(df):
        # Gets rid of empty home and away points and quarterly data
        df = df.dropna(subset=["home_points", "away_points"])
        df = df[
            df["home_line_scores"].apply(lambda x: x != [])
            & df["away_line_scores"].apply(lambda x: x != [])
        ]
        return df

    def encode_categorical_cols(df):
        # TODO: ensure that the names are separated by _
        pass

    def transform(self) -> pd.DataFrame:
        pass


class OffensiveFeatures(Feature):
    # TODO: Kalman filter for scoring in each quarter
    # TODO: Percent of yards come from rushing/passing
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
