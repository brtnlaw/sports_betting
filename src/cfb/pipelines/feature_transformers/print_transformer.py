import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PrintTransformer(BaseEstimator, TransformerMixin):
    """Dummy class to incorporate print statements into feature pipeline."""

    def __init__(self, message: str = ""):
        """
        Which statement to print.

        Args:
            message (str, optional): Message to print. Defaults to "".
        """
        self.message = message

    def fit(self, X, y=None):
        """Dummy class for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Trivially prints a message.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Same input DataFrame.
        """
        print(f"{self.message}")
        return X
