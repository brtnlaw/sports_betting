import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler


def generate_features(data: pd.DataFrame, input_data: pd.Series) -> pd.DataFrame:
    """
    Takes in the stat sheet data and generates features to generate resampling weights.

    Args:
        data (pd.DataFrame): Statsheet data.
        input_data (pd.Series): Generates features based on the stat sheet input.

    Returns:
        pd.DataFrame: Statsheet data with additional features.
    """
    data["same_opponent"] = (data["opponent"] == input_data["opponent"]).apply(
        lambda x: int(x)
    )
    data["same_player"] = (data["player"] == input_data["player"]).apply(
        lambda x: int(x)
    )
    data["same_venue_status"] = (
        (data["home"] == data["team"]) == (input_data["home"] == input_data["team"])
    ).apply(lambda x: int(x))
    data["days_since_game"] = -(data["date"] - input_data["date"]).apply(
        lambda x: x.days
    )

    to_normalize_columns = ["days_since_game"]
    scaler = MinMaxScaler().fit(data[to_normalize_columns])
    data[to_normalize_columns] = scaler.transform(data[to_normalize_columns])
    return data


def loss(
    kde: KernelDensity,
):
    # TODO: some form of -log(p;y), for a given observation
    pass


# def gradient(
#         kde: KernelDensity,
# ) -> pd.DataFrame:
