import os
import pickle as pkl
import time

import cfbd
import numpy as np
import pandas as pd
from cfb_base import CFBBase
from cfbd.models.play import Play
from cfbd.rest import ApiException

from db_utils import insert_data_to_db


class CFBPlayByPlayData(CFBBase):
    """Handles fetching, storing, and uploading CFB play-by-play data. Also retrieves from PostgreSQL."""

    def __init__(self):
        """Loads the API keys, as well as configures the API connection to CFBD."""
        super().__init__()
        self.api = cfbd.PlaysApi(self.api_client)

    def _get_pkl_path(self, year: int, week: int) -> str:
        """
        Returns file path to pkl for a given year and week.

        Args:
            year (int): Year of interest.
            week (int): Week of interest.

        Returns:
            str: Path to pkl file.
        """
        return os.path.join(
            self.project_root, f"src/cfb/data/pkl_files/play_by_play_{year}_{week}.pkl"
        )

    def fetch_and_pickle_play_by_play_at_year_week(self, year: int, week: int) -> None:
        """
        Calls the API for play by play stats and pickles at a given year and week, errors if it exists already.

        Args:
            year (int): Season.
            week (int): Week of interest.

        Raises:
            Exception: Path already exists.
        """
        path = self._get_pkl_path(year, week)
        if os.path.isfile(path):
            raise Exception(f"{path} already exists.")
        try:
            plays = self.api.get_plays(year=year, week=week)
            with open(path, "wb") as f:
                pkl.dump(plays, f)
            print(f"Successfully pickled to {path}.")
        except ApiException as e:
            print(f"Error fetching games for {year}, week {week}: {e}")

    def _pbp_expand_to_df(self, play: Play) -> pd.DataFrame:
        """
        Expands Play to a DataFrame.

        Args:
            play (Play): Input Play object.

        Returns:
            pd.DataFrame: Play object as DataFrame.
        """
        play_dict = play.dict()
        play_dict["clock_minutes"] = play_dict["clock"]["minutes"]
        play_dict["clock_seconds"] = play_dict["clock"]["seconds"]
        del play_dict["clock"]
        return pd.DataFrame([play_dict])

    def load_play_by_play_from_pkl_at_year_week(
        self, year: int, week: int
    ) -> pd.DataFrame:
        """
        Loads the play-by-play stats directly from the pkl.

        Args:
            year (int): Season.
            week (int): Week of interest.

        Returns:
            pd.DataFrame: Pickled data in DataFrame form.
        """
        path = self._get_pkl_path(year, week)
        if not os.path.isfile(path):
            self.fetch_and_pickle_play_by_play_at_year_week(year, week)

        with open(path, "rb") as f:
            play_list = pkl.load(f)

        year_df = pd.concat(
            [self._pbp_expand_to_df(play) for play in play_list]
        ).applymap(lambda x: None if pd.isna(x) else x)

        year_df = year_df.replace({np.nan: None})

        # Organize order of columns
        year_df = year_df[
            [
                "id",
                "drive_id",
                "game_id",
                "drive_number",
                "play_number",
                "offense",
                "offense_conference",
                "offense_score",
                "defense",
                "home",
                "away",
                "defense_conference",
                "defense_score",
                "period",
                "offense_timeouts",
                "defense_timeouts",
                "yardline",
                "yards_to_goal",
                "down",
                "distance",
                "yards_gained",
                "scoring",
                "play_type",
                "play_text",
                "ppa",
                "wallclock",
                "clock_minutes",
                "clock_seconds",
            ]
        ]
        return year_df

    def upload_play_by_play_to_db(self, start: int = 2013, end: int = 2025) -> None:
        """
        Uploads the play-by-play data from pkl to PostgreSQL.

        Args:
            start (int): Start season.
            end (int): Ending season, not included.
        """
        query = """
            INSERT INTO cfb.play_by_play
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        for year in range(start, end):
            for week in range(1, 17):
                if (
                    year in [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
                    and week == 16
                ):
                    continue
                print(f"Uploading play_by_play stats for {year}, week {week}...")
                data = self.load_play_by_play_from_pkl_at_year_week(year, week)
                insert_data_to_db(query, data)
                time.sleep(0.5)


if __name__ == "__main__":
    # python src/cfb/data/cfb_play_by_play_data.py
    pipeline = CFBPlayByPlayData()
    pipeline.upload_play_by_play_to_db()
