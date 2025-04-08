import os

import pandas as pd

from db_utils import execute_sql_script, retrieve_data

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


class DataPrep:
    """
    Class whose function is solely to generate the desired data. Does not manipulate, only provides the necessary raw data, except for cases of obviously erroneous data.
    """

    def __init__(self, dataset="cfb"):
        """
        Initializes which data to load.

        Args:
            dataset (str, optional): The type of sport. Defaults to "cfb".
        """
        self.dataset = dataset
        self.df = None

    def make_schemas_tables(self):
        """Generates all the schemas and tables, if they don't already exist."""
        queries_path = os.path.join(PROJECT_ROOT, f"src/cfb/data/sql_queries/")
        if os.path.isdir(queries_path):
            for patch_file in os.listdir(queries_path):
                patch_path = os.path.join(queries_path, patch_file)
                execute_sql_script(patch_path)

    def load_and_patch_errors(self):
        """
        To be run once first. Loads tables if they don't exist, then the data, if it doesn't exist.
        Patches any manual errors, i.e. wrong team entered by API. Not part of the data engineering.
        """
        self.make_schemas_tables()
        retrieve_data(self.dataset, "games")
        retrieve_data(self.dataset, "venues")
        retrieve_data(self.dataset, "lines")

        patches_path = os.path.join(PROJECT_ROOT, f"src/cfb/data/patches/")
        if os.path.isdir(patches_path):
            for patch_file in os.listdir(patches_path):
                patch_path = os.path.join(patches_path, patch_file)
                execute_sql_script(patch_path)

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
            suffixes=("", "_venue"),
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
        """Returns the un-processed data for further transformations."""
        self.load_data()
        self.remove_columns()
        return self.df
