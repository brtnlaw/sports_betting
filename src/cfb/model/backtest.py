import sys

sys.path.insert(0, "../")
from features import build_features
import betting_logic
from db_utils import retrieve_data
from train import train_model
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np

## TODO: add args
warnings.simplefilter(action="ignore", category=FutureWarning)


def prepare_data(line="ou", col="total"):
    # Prepares the data
    venue_df = retrieve_data("cfb", "venues")
    game_df = retrieve_data("cfb", "games")
    df = pd.merge(
        game_df,
        venue_df,
        "left",
        left_on=["venue_id", "venue"],
        right_on=["id", "name"],
        suffixes=("", "_venue"),
    )
    fp = build_features.FeaturePipeline(df)
    df = fp.run()

    # Merge odds data
    line_df = retrieve_data("cfb", "lines")
    if line == "ou":
        col_names = ["min_ou", "max_ou"]
        ou_df = line_df.groupby("id")["over_under"].agg(["min", "max"])
        ou_df.columns = col_names
        df = pd.merge(df, ou_df, how="left", on="id")

    # Clean up
    df.sort_values(by="start_date", inplace=True)
    df.reset_index(inplace=True, drop=True)
    X = df.drop(columns=[col, "id", "season", "start_date"] + col_names)
    y = df[col]
    return df, X, y


def plot_pnl(df):
    # Ease of seeing, can include breakpoints for season
    plot_df = df[df["unit_pnl"] != 0]
    seasons = df.groupby("season").head(1).index
    # Includes the first 5
    # for season_start in seasons:
    #     plt.axvline(
    #         x=season_start,
    #         color="r",
    #         linestyle="--",
    #         label="Season Start" if season_start == seasons[0] else "",
    #     )
    plt.plot(plot_df["unit_pnl"].cumsum())
    plt.show()


def cv(df, X, y):
    cv_year_indices = df.groupby("season").head(1).index
    init_train_yrs = 5
    cv_year_indices = cv_year_indices[init_train_yrs:].append(pd.Index([df.index[-1]]))
    for i in range(len(cv_year_indices) - 1):
        idx = cv_year_indices[i]
        next_idx = cv_year_indices[i + 1]

        # Slice data correctly for training and testing
        X_train = X.iloc[:idx]
        y_train = y.iloc[:idx]
        X_test = X.iloc[idx:next_idx]
        model = train_model(X_train, y_train)
        df["unit_pnl"] = 0
        # Make predictions and update df
        predictions = model.predict(X_test)
        df.loc[idx : next_idx - 1, "pred"] = predictions

    df = betting_logic.simple_percentage(df)
    plot_pnl(df)
    return df


if __name__ == "__main__":
    # python src/cfb/model/backtest.py
    df, X, y = prepare_data()
    cv(df, X, y)
