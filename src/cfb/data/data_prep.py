import os

import pandas as pd

from db_utils import execute_sql_script, pull_from_db, retrieve_data


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
        self.project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        self.dataset = dataset
        self.df = None

    # TODO: Fix the ordering
    def make_schemas_tables(self):
        """Generates all the schemas and tables, if they don't already exist."""
        queries_path = os.path.join(self.project_root, f"src/cfb/data/sql_queries/")
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

        patches_path = os.path.join(self.project_root, f"src/cfb/data/patches/")
        if os.path.isdir(patches_path):
            for patch_file in os.listdir(patches_path):
                patch_path = os.path.join(patches_path, patch_file)
                execute_sql_script(patch_path)

    def load_data(self):
        """Fetch game, venue, and odds data from the database or other sources."""
        venue_df = retrieve_data(self.dataset, "venues")
        game_df = retrieve_data(self.dataset, "games")
        line_df = retrieve_data(self.dataset, "lines")
        game_team_stat_df = retrieve_data(self.dataset, "game_team_stats")

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
        bet_df = line_df.groupby("id").agg(
            {"over_under": ["min", "max"], "spread": ["min", "max"]}
        )
        bet_df.columns = ["min_ou", "max_ou", "min_spread", "max_spread"]
        self.df = pd.merge(self.df, bet_df, how="left", on="id")

        # Merge box score data
        for side in ["home", "away"]:
            side_gts = game_team_stat_df.add_prefix(f"{side}_")
            self.df = self.df.merge(
                side_gts,
                how="left",
                left_on=["id", f"{side}_id", f"{side}_team"],
                right_on=[f"{side}_game_id", f"{side}_team_id", f"{side}_team"],
            )

        # Get necessary stats from play-by-play
        pbp_query = """
            SELECT
                game_id,
                offense AS team,
                COUNT(CASE WHEN yards_gained >= 30 THEN 1 END) AS plays_30_plus,
                COUNT(CASE WHEN yards_gained >= 35 THEN 1 END) AS plays_35_plus,
                COUNT(CASE WHEN yards_gained >= 40 THEN 1 END) AS plays_40_plus,
            FROM
                cfb.play_by_play
            GROUP BY
                game_id,
                offense
            ORDER BY
                game_id,
                offense;
            """
        pbp_df = pull_from_db(pbp_query)
        pbp_cols = [col for col in pbp_df.columns if col not in ("game_id", "team")]

        for side in ["home", "away"]:
            self.df = self.df.merge(
                pbp_df,
                how="left",
                left_on=["id", f"{side}_team"],
                right_on=["game_id", "team"],
            )
            for col in pbp_cols:
                self.df.rename(columns={col: f"{side}_{col}"}, inplace=True)
            self.df.drop(columns=["team", "game_id"], inplace=True)
        # NOTE: There's a separate PPA database, consider using that instead?

    def remove_columns(self):
        """Remove truly unnecessary columns to simplify the dataset."""
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
