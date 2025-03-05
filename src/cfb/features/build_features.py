import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Optional

"""
Ideas:
- Rolling 3 game window of points gained/allowed, or if not, perhaps on the season (would be on the prior games) "Recent form"
- Same as above, but yardage
- How many games into the season? Bowl game?
- Kalman filter for rolling

TODOs:
- Categorizing features (examples given were Offense, Defense, etc.)
"""
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the date of the game into temporal features.
    """
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['days_since'] = (pd.to_datetime("today") - df['date']).dt.days
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['year'] = df['date'].dt.year
    return df

def extract_halves(df):
    """
    Turn the quarters into halves to use
    Individual quarters serve no purpose
    """
    df["home_half"] = df["home_first_quarter"] + df["home_second_quarter"]
    df["visitor_half"] = df["visitor_first_quarter"] + df["visitor_second_quarter"]
    return df

def extract_data(df: pd.DataFrame, useless_cols: Optional[list] = ["ot", "unique_id", "created_at"]) -> pd.DataFrame: 
    """
    Clean up unnecesary columns from df. Fills in NaNs.
    """
    df.drop(columns=useless_cols, errors="ignore", inplace=True)
    df.fillna(0, inplace=True)
    extract_functions = [extract_date_features, extract_halves]
    for function in extract_functions:
        df = function(df)
    df.columns = df.columns.str.replace(' ', '_')
    return df

def add_rolling_recent_form(df, window=3, min_periods=1):
    # TODO: Kalman -> Honestly a more sophisticated verseion of the above
    # TODO: can possibly simplify?
    """Simple recent form as a visitor and as the home team"""
    home_df = df[['date', 'home', 'home_half', 'home_points', 'visitor_half', 'visitor_points']].rename(
        columns={'home': 'team', 'home_half': 'half', 'home_points': 'points', 'visitor_half': 'opp_half', 'visitor_points': 'opp_points'})
    visitor_df = df[['date', 'visitor', 'visitor_half', 'visitor_points']].rename(
        columns={'visitor': 'team', 'visitor_half': 'half','visitor_points': 'points', 'home_half': 'opp_half', 'home_points': 'opp_points'})
    game_df = pd.concat([home_df, visitor_df])
    game_df.sort_values(by=["team", "date"], inplace=True)
    game_df[["rolling_half", "rolling_points", "rolling_opp_half", "rolling_opp_points"]] = game_df.groupby(
        'team')[['half', 'points', 'opp_half', "opp_points"]].rolling(window=window, min_periods=min_periods).mean().reset_index(level=0, drop=True)

    df = df.merge(game_df[['date', 'team', "rolling_half", "rolling_points", "rolling_opp_half", "rolling_opp_points" ]],
                   left_on=['date', 'home'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={'rolling_points': 'home_rolling_points', 
                            'rolling_opp_half': 'home_rolling_opp_half', 
                            'rolling_opp_points': 'home_rolling_opp_points'}).drop(columns=['team'])
    
    df = df.merge(game_df[['date', 'team', "rolling_half", "rolling_points", "rolling_opp_half", "rolling_opp_points"]], 
                left_on=['date', 'visitor'], right_on=['date', 'team'], how='left')
    df = df.rename(columns={'rolling_half': 'visitor_rolling_half', 
                            'rolling_points': 'visitor_rolling_points', 
                            'rolling_opp_half': 'visitor_rolling_opp_half', 
                            'rolling_opp_points': 'visitor_rolling_opp_points'}).drop(columns=['team'])
    return df

def add_encoder(df: pd.DataFrame) -> pd.DataFrame:
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
    df_encoded = pd.concat([df, df_one_hot], axis=1)
    return df_encoded

def add_features(df, feature_functions=[add_rolling_recent_form]):
    """
    Workflow is first you make the columns numeric, then build features, then clean up the categorical columns
    """
    df = extract_data(df)
    for function in feature_functions:
        df = function(df)
    extra_cols = ["home_first_quarter", "home_second_quarter", "home_third_quarter", "home_fourth_quarter", 
                  "visitor_first_quarter", "visitor_second_quarter", "visitor_third_quarter", "visitor_fourth_quarter",
                  "date", "home", "visitor"]
    df.drop(columns=extra_cols, inplace=True)
    return df
