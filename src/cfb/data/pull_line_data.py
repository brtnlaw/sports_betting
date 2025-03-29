import os
import pickle as pkl
import warnings

import cfbd
import pandas as pd
from cfbd.models.betting_game import BettingGame
from cfbd.rest import ApiException
from dotenv import load_dotenv

from db_utils import insert_data_to_db


# TODO: This is only using major markets. Future work necessarily must involve derivative markets (i.e. NCAAF halves). We will use this as a starting point.
class CFBLineData:
    """
    Handles fetching, storing, and uploading CFB game data. Also retrieves from PostgreSQL.
    """

    def __init__(self):
        """
        Loads the API keys, as well as configures the API connection to CFBD.
        """
        load_dotenv()
        self.cfbd_api_key = os.getenv("CFBD_API_KEY")
        self.project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.configuration = cfbd.Configuration(
            host="https://apinext.collegefootballdata.com",
            access_token=self.cfbd_api_key,
        )
        self.api_client = cfbd.ApiClient(self.configuration)
        self.api = cfbd.BettingApi(self.api_client)

    def _get_pkl_path(self, year: int) -> str:
        """
        Returns file path to pkl for a given year.

        Args:
            year (int): Year of interest.

        Returns:
            str: Path to pkl file.
        """
        return os.path.join(self.project_root, f"src/cfb/data/pkl/lines_{year}.pkl")

    def fetch_and_pickle_lines_at_year(self, year: int) -> None:
        """
        Calls the API for game data and pickles at a given year, errors if it exists already.

        Args:
            year (int): Season.

        Raises:
            Exception: Path already exists.
        """
        path = self._get_pkl_path(year)
        if os.path.isfile(path):
            raise Exception(f"{path} already exists.")
        try:
            lines = self.api.get_lines(year=year)
            with open(path, "wb") as f:
                pkl.dump(lines, f)
            print(f"Successfully pickled to {path}")
        except ApiException as e:
            print("Error fetching lines for {year}: {e}")

    def _bg_expand_to_df(self, bg: BettingGame):
        bg_dict = pd.DataFrame(bg.dict())
        expanded_lines = pd.json_normalize(bg_dict["lines"])
        return pd.concat([bg_dict.drop("lines", axis=1), expanded_lines], axis=1)

    def load_lines_from_pkl_at_year(self, year: int) -> pd.DataFrame:
        """
        Loads the lines directly from the pkl.

        Args:
            year (int): Season.

        Returns:
            pd.DataFrame: Pickled data in DataFrame form.
        """
        path = self._get_pkl_path(year)
        if not os.path.isfile(path):
            self.fetch_and_pickle_lines_at_year(year)

        with open(path, "rb") as f:
            line_list = pkl.load(f)

        year_df = pd.concat([self._bg_expand_to_df(bg) for bg in line_list])

        # Organize order of columns
        year_df = year_df[
            [
                "id",
                "season",
                "season_type",
                "week",
                "start_date",
                "home_team",
                "home_conference",
                "home_classification",
                "home_score",
                "away_team",
                "away_conference",
                "away_classification",
                "away_score",
                "provider",
                "spread",
                "formatted_spread",
                "spread_open",
                "over_under",
                "over_under_open",
                "home_moneyline",
                "away_moneyline",
            ]
        ]
        return year_df

    def upload_lines_to_db(self, start: int = 2013, end: int = 2025) -> None:
        """
        Uploads the game data from pkl to PostgreSQL.

        Args:
            start (int): Start season.
            end (int): Ending season, not included.
        """
        query = """
            INSERT INTO cfb.lines
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        for year in range(start, end):
            print(f"Uploading lines data for {year}...")
            data = self.load_lines_from_pkl_at_year(year)
            insert_data_to_db(query, data)


if __name__ == "__main__":
    # python src/cfb/data/pull_line_data.py
    pipeline = CFBLineData()
    pipeline.upload_lines_to_db()
