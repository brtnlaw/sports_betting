import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Optional

def clean_df(df: pd.DataFrame, useless_cols: Optional[list] = ["ot", "unique_id", "created_at"]) -> pd.DataFrame: 
    """
    Clean up unnecesary columns from df. Fills in NaNs.

    Args:
        df (pd.DataFrame): Raw dataframe.
        useless_cols (Optional[list], optional): List of unused columns. Defaults to ["ot", "unique_id", "created_at"].

    Returns:
        pd.DataFrame: Cleaned df.
    """
    df.drop(columns=useless_cols, errors="ignore", inplace=True)
    df.fillna(0, inplace=True)
    return df
    

def encode_team_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the categorical variable of 'home' and 'visitor' into binary features.

    Args:
        df (pd.DataFrame): Dataframe with 'home' and 'visitor'.

    Returns:
        pd.DataFrame: Dataframe with dummy variables for team name.
    """
    if "home" not in df.columns and "visitor" not in df.columns:
        return df
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    one_hot_encoded = one_hot_encoder.fit_transform(df[["home", "visitor"]])
    feature_names = one_hot_encoder.get_feature_names_out(["home", "visitor"])
    df_one_hot = pd.DataFrame(one_hot_encoded, columns=feature_names)
    df_encoded = pd.concat([df.drop(["home", "visitor"], axis=1), df_one_hot], axis=1)
    return df_encoded

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the date of the game into temporal features.

    Args:
        df (pd.DataFrame): Dataframe with 'date'.

    Returns:
        pd.DataFrame: Dataframe with temporal features.
    """
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['days_since'] = (pd.to_datetime("today") - df['date']).dt.days
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df.drop(columns=['date'], inplace=True)
    return df

def add_halves(df):
    df["home_half"] = df["home_first_quarter"] + df["home_second_quarter"]
    df["visitor_half"] = df["visitor_first_quarter"] + df["visitor_second_quarter"]
    df.drop(columns=["home_first_quarter", "home_second_quarter", "home_third_quarter", "home_fourth_quarter",
                     "visitor_first_quarter", "visitor_second_quarter", "visitor_third_quarter", "visitor_fourth_quarter"],
                     inplace=True)
    return df

def add_features(df, functions = [encode_team_name, add_temporal_features, add_halves]):
    df = clean_df(df)
    for function in functions:
        df = function(df)
    return df
