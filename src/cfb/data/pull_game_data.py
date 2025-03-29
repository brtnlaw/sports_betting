import os
import pickle as pkl
import warnings

import cfbd
import pandas as pd
from cfbd.rest import ApiException
from dotenv import load_dotenv

from db_utils import insert_data_to_db


class CFBGameData:
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
        self.api = cfbd.GamesApi(self.api_client)

    def _get_pkl_path(self, year: int) -> str:
        """
        Returns file path to pkl for a given year.

        Args:
            year (int): Year of interest.

        Returns:
            str: Path to pkl file.
        """
        return os.path.join(self.project_root, f"src/cfb/data/pkl/games_{year}.pkl")

    def fetch_and_pickle_games_at_year(self, year: int) -> None:
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
            games = self.api.get_games(year=year)
            with open(path, "wb") as f:
                pkl.dump(games, f)
            print(f"Successfully pickled to {path}")
        except ApiException as e:
            print("Error fetching games for {year}: {e}")

    def load_games_from_pkl_at_year(self, year: int) -> pd.DataFrame:
        """
        Loads the games directly from the pkl.

        Args:
            year (int): Season.

        Returns:
            pd.DataFrame: Pickled data in DataFrame form.
        """
        path = self._get_pkl_path(year)
        if not os.path.isfile(path):
            self.fetch_and_pickle_games_at_year(year)

        with open(path, "rb") as f:
            game_list = pkl.load(f)

        game_df_list = [pd.DataFrame([game.to_dict()]) for game in game_list]
        year_df = pd.concat(game_df_list)

        # Organize order of columns
        year_df = year_df[
            [
                "id",
                "season",
                "week",
                "seasonType",
                "startDate",
                "startTimeTBD",
                "completed",
                "neutralSite",
                "conferenceGame",
                "attendance",
                "venueId",
                "venue",
                "homeId",
                "homeTeam",
                "homeConference",
                "homeClassification",
                "homePoints",
                "homeLineScores",
                "homePostgameWinProbability",
                "homePregameElo",
                "homePostgameElo",
                "awayId",
                "awayTeam",
                "awayConference",
                "awayClassification",
                "awayPoints",
                "awayLineScores",
                "awayPostgameWinProbability",
                "awayPregameElo",
                "awayPostgameElo",
                "excitementIndex",
                "highlights",
                "notes",
            ]
        ]
        return year_df

    def upload_games_to_db(self, start: int = 2013, end: int = 2025) -> None:
        """
        Uploads the game data from pkl to PostgreSQL.

        Args:
            start (int): Start season.
            end (int): Ending season, not included.
        """
        query = """
            INSERT INTO cfb.games
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        for year in range(start, end):
            print(f"Uploading games data for {year}...")
            data = self.load_games_from_pkl_at_year(year)
            insert_data_to_db(query, data)


if __name__ == "__main__":
    # python src/cfb/data/pull_game_data.py
    pipeline = CFBGameData()
    pipeline.upload_games_to_db()
