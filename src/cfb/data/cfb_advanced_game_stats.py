import os
import pickle as pkl
import time

import cfbd
import pandas as pd
from cfb_base import CFBBase
from cfbd.models.advanced_game_stat import AdvancedGameStat
from cfbd.rest import ApiException

from db_utils import insert_data_to_db


class CFBAdvancedGameStats(CFBBase):
    """Handles fetching, storing, and uploading CFB advanced stats. Also retrieves from PostgreSQL."""

    def __init__(self):
        """Loads the API keys, as well as configures the API connection to CFBD."""
        super().__init__()
        self.api = cfbd.StatsApi(self.api_client)

    def _get_pkl_path(self, year: int) -> str:
        """
        Returns file path to pkl for a given year.

        Args:
            year (int): Year of interest.

        Returns:
            str: Path to pkl file.
        """
        return os.path.join(
            self.project_root,
            f"src/cfb/data/pkl_files/advanced_game_{year}.pkl",
        )

    def fetch_and_pickle_advanced_game_stats_at_year(self, year: int) -> None:
        """
        Calls the API for advanced game stats and pickles at a given year, errors if it exists already.

        Args:
            year (int): Season.

        Raises:
            Exception: Path already exists.
        """
        path = self._get_pkl_path(year)
        if os.path.isfile(path):
            raise Exception(f"{path} already exists.")
        try:
            advanced_stats = self.api.get_advanced_game_stats(year=year)
            with open(path, "wb") as f:
                pkl.dump(advanced_stats, f)
            print(f"Successfully pickled to {path}.")
        except ApiException as e:
            print(f"Error fetching games for {year}: {e}")

    def _ags_expand_to_df(self, ags: AdvancedGameStat) -> pd.DataFrame:
        """
        Expands AdvancedGameStat to a DataFrame.

        Args:
            ags (AdvancedGameStat): Input AdvancedGameStat object.

        Returns:
            pd.DataFrame: AdvancedGameStat object as DataFrame.
        """
        flat_df = pd.json_normalize(ags.dict(), sep="_")
        return flat_df

    def load_advanced_game_stats_from_pkl_at_year(self, year: int) -> pd.DataFrame:
        """
        Loads the game team stats directly from the pkl.

        Args:
            year (int): Season.

        Returns:
            pd.DataFrame: Pickled data in DataFrame form.
        """
        path = self._get_pkl_path(year)
        if not os.path.isfile(path):
            self.fetch_and_pickle_advanced_game_stats_at_year(year)

        with open(path, "rb") as f:
            advanced_game_stat_list = pkl.load(f)

        year_df = pd.concat(
            [self._ags_expand_to_df(ags) for ags in advanced_game_stat_list]
        ).applymap(lambda x: None if pd.isna(x) else x)

        # Organize order of columns
        year_df = year_df[
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
                "offense_passing_plays_explosiveness",
                "offense_passing_plays_success_rate",
                "offense_passing_plays_total_ppa",
                "offense_passing_plays_ppa",
                "offense_rushing_plays_explosiveness",
                "offense_rushing_plays_success_rate",
                "offense_rushing_plays_total_ppa",
                "offense_rushing_plays_ppa",
                "offense_passing_downs_explosiveness",
                "offense_passing_downs_success_rate",
                "offense_passing_downs_ppa",
                "offense_standard_downs_explosiveness",
                "offense_standard_downs_success_rate",
                "offense_standard_downs_ppa",
                "offense_open_field_yards_total",
                "offense_open_field_yards",
                "offense_second_level_yards_total",
                "offense_second_level_yards",
                "offense_line_yards_total",
                "offense_line_yards",
                "offense_stuff_rate",
                "offense_power_success",
                "offense_explosiveness",
                "offense_success_rate",
                "offense_total_ppa",
                "offense_ppa",
                "offense_drives",
                "offense_plays",
                "defense_passing_plays_explosiveness",
                "defense_passing_plays_success_rate",
                "defense_passing_plays_total_ppa",
                "defense_passing_plays_ppa",
                "defense_rushing_plays_explosiveness",
                "defense_rushing_plays_success_rate",
                "defense_rushing_plays_total_ppa",
                "defense_rushing_plays_ppa",
                "defense_passing_downs_explosiveness",
                "defense_passing_downs_success_rate",
                "defense_passing_downs_ppa",
                "defense_standard_downs_explosiveness",
                "defense_standard_downs_success_rate",
                "defense_standard_downs_ppa",
                "defense_open_field_yards_total",
                "defense_open_field_yards",
                "defense_second_level_yards_total",
                "defense_second_level_yards",
                "defense_line_yards_total",
                "defense_line_yards",
                "defense_stuff_rate",
                "defense_power_success",
                "defense_explosiveness",
                "defense_success_rate",
                "defense_total_ppa",
                "defense_ppa",
                "defense_drives",
                "defense_plays",
            ]
        ]
        return year_df

    def upload_advanced_game_stats_to_db(
        self, start: int = 2013, end: int = 2025
    ) -> None:
        """
        Uploads the game data from pkl to PostgreSQL.

        Args:
            start (int): Start season.
            end (int): Ending season, not included.
        """
        query = """
            INSERT INTO cfb.advanced_game_stats
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        for year in range(start, end):
            print(f"Uploading advanced game stats for {year}...")
            data = self.load_advanced_game_stats_from_pkl_at_year(year)
            insert_data_to_db(query, data)
            time.sleep(0.5)


if __name__ == "__main__":
    # python src/cfb/data/cfb_advanced_game_stats.py
    instance = CFBAdvancedGameStats()
    instance.upload_advanced_game_stats_to_db()
