import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def clean_df(df, useless_cols=["ot", "unique_id", "created_at"]):
    """
    Clean unused variables
    """
    df.drop(columns=useless_cols, errors="ignore", inplace=True)
    return df
    

def encode_team_name(df):
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    one_hot_encoded = one_hot_encoder.fit_transform(df[["home", "visitor"]])
    feature_names = one_hot_encoder.get_feature_names_out(["home", "visitor"])
    df_one_hot = pd.DataFrame(one_hot_encoded, columns=feature_names)
    df_encoded = pd.concat([df.drop(["home", "visitor"], axis=1), df_one_hot], axis=1)
    return df_encoded

def add_temporal_features(df):
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
    return df
