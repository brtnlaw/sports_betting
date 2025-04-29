import os
from typing import Union

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


def load_pkl_if_exists(
    name_str: str,
    target_str: str = "total",
    betting_fnc: str = "spread_probs",
    file_type: str = "odds_df",
) -> Union[Pipeline, pd.DataFrame]:
    """
    Helper function to load 'pipeline', 'contrib_df', or 'odds_df' from a str.

    Args:
        name_str (str): Prefix of file string.
        target_str (str, optional): For model, the target column. Defaults to "total".
        betting_fnc (str, optional): Function to determine bets. Defaults to "spread_probs".
        file_type (str, optional): Type of file to retrieve. Defaults to "df".

    Raises:
        Exception: Necessarily is a 'pipeline' or 'df' file.

    Returns:
        Any[Pipeline, pd.DataFrame]: Returns either a pipeline or DataFrame.
    """
    assert file_type in [
        "pipeline",
        "contrib_df",
        "odds_df",
    ], "Pick a file_type in 'pipeline', 'contrib_df', 'odds_df'"
    if file_type == "pipeline":
        file_path = os.path.join(
            PROJECT_ROOT, f"src/cfb/models/{name_str}_{target_str}_pipeline.pkl"
        )
    elif file_type == "contrib_df":
        file_path = os.path.join(
            PROJECT_ROOT,
            f"src/cfb/models/{name_str}_{target_str}_contrib.pkl",
        )
    elif file_type == "odds_df":
        file_path = os.path.join(
            PROJECT_ROOT,
            f"src/cfb/models/{name_str}_{target_str}_{betting_fnc}.pkl",
        )
    if not os.path.exists(file_path):
        raise Exception(f"No properly configured {file_type} file.")
    result = joblib.load(file_path)
    return result


# NOTE: Simple post-pipeline df helper
