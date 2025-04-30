from typing import Any, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option("future.no_silent_downcasting", True)


class MultipleValueImputer(BaseEstimator, TransformerMixin):
    """Imputes multiple null values."""

    def __init__(self, null_values: List[Any], impute_val: Any):
        """
        Initializes class with valid null values and desired imputed value.

        Args:
            null_values (List[Any]): List of possible null values.
            impute_val (Any): Value to impute with.
        """
        self.null_values = null_values
        self.impute_val = impute_val

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes any of the null values.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        X_.replace(self.null_values, self.impute_val, inplace=True)

        for col in X_.columns:
            if X_[col].dtype == "object":
                X_[col] = pd.to_numeric(X_[col])
                if X_[col].dtype == "object":
                    X_[col] = pd.to_datetime(X_[col])
                if X_[col].dtype == "object" and X_[col].isin(["True", "False"]).all():
                    X_[col] = X_[col].map({"True": True, "False": False})
        return X_
