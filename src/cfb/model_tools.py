import os
from typing import Union

import joblib
import pandas as pd
from data.data_prep import DataPrep
from pipelines.pipeline import get_features_and_model_pipeline
from pipelines.preprocessing import get_preprocess_pipeline
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


def get_transformed_data(target_col: str = "home_away_spread") -> pd.DataFrame:
    """
    Helper function to get the transformed data.

    Args:
        target_col (str, optional): Target column to drop from X. Defaults to "home_away_spread".

    Returns:
        pd.DataFrame: Finalized DataFrame through pipeline.
    """
    data_prep = DataPrep(dataset="cfb")
    raw_data = data_prep.get_data()
    preprocessed_data = get_preprocess_pipeline().fit_transform(raw_data)
    target_line_dict = {
        "total": ["min_ou", "max_ou"],
        "home_away_spread": ["min_spread", "max_spread"],
    }
    all_betting_cols = [
        line for line_list in target_line_dict.values() for line in line_list
    ]

    X = preprocessed_data.drop(columns=[target_col] + all_betting_cols)

    pipeline = get_features_and_model_pipeline()

    # Fit the pipeline up to the last step
    pipeline_no_regress = Pipeline(pipeline.steps[:-1])
    return pipeline_no_regress.transform(X)
