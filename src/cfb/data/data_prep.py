import pandas as pd
from db_utils import retrieve_data


class DataPrep:
    """
    Class whose function is solely to generate the desired data. Does not manipulate, only provides the necessary raw data.
    """

    def __init__(self, dataset="cfb"):
        self.dataset = dataset
        self.df = None

    def load_data(self):
        """Fetch game, venue, and odds data from the database or other sources."""
        game_df = retrieve_data(self.dataset, "games")
        venue_df = retrieve_data(self.dataset, "venues")
        line_df = retrieve_data(self.dataset, "lines")

        # Merge game and venue data on venue_id
        self.df = pd.merge(
            game_df,
            venue_df,
            how="left",
            left_on=["venue_id", "venue"],
            right_on=["id", "name"],
        )

        # Merge betting odds
        ou_df = line_df.groupby("id")["over_under"].agg(["min", "max"])
        ou_df.columns = ["min_ou", "max_ou"]
        self.df = pd.merge(self.df, ou_df, how="left", on="id")

    def remove_columns(self):
        """Remove unnecessary columns to simplify the dataset."""
        useless_cols = [
            "start_time_tbd",
            "completed",
            "home_id",
            "away_id",
            "highlights",
            "notes",
            "id_venue",
            "name",
            "city",
            "state",
            "zip",
            "countrycode",
        ]
        self.df.drop(
            columns=[col for col in useless_cols if col in self.df.columns],
            inplace=True,
        )

    def get_data(self):
        """Returns the processed data for further transformations."""
        self.load_data()
        self.remove_columns()
        return self.df
