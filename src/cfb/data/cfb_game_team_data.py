import os
import pickle as pkl
import warnings

import cfbd
import pandas as pd
from cfbd.models.game_team_stats import GameTeamStats
from cfbd.rest import ApiException
from dotenv import load_dotenv

from db_utils import insert_data_to_db

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


class CFBGameTeamData:
    """Handles fetching, storing, and uploading CFB team box data. Also retrieves from PostgreSQL."""

    def __init__(self):
        """Loads the API keys, as well as configures the API connection to CFBD."""
        load_dotenv()
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.configuration = cfbd.Configuration(
            host="https://apinext.collegefootballdata.com",
            access_token=CFBD_API_KEY,
        )
        self.api_client = cfbd.ApiClient(self.configuration)
        self.api = cfbd.GamesApi(self.api_client)

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
            PROJECT_ROOT, f"src/cfb/data/pkl_files/game_team_{year}_{week}.pkl"
        )

    def fetch_and_pickle_game_team_stats_at_year_week(
        self, year: int, week: int
    ) -> None:
        """
        Calls the API for game team stats and pickles at a given year and week, errors if it exists already.

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
            games = self.api.get_game_team_stats(year=year, week=week)
            with open(path, "wb") as f:
                pkl.dump(games, f)
            print(f"Successfully pickled to {path}.")
        except ApiException as e:
            print(f"Error fetching games for {year}, week {week}: {e}")

    def _gts_expand_to_df(self, gts: GameTeamStats) -> pd.DataFrame:
        """
        Expands GameTeamStats to a DataFrame.

        Args:
            gts (GameTeamStats): Input GameTeamStats object.

        Returns:
            pd.DataFrame: GameTeamStats object as DataFrame.
        """
        teams_df = pd.json_normalize(gts.to_dict(), record_path=["teams"], meta="id")
        teams_df = teams_df.explode("stats").reset_index(drop=True)
        stats_df = pd.json_normalize(teams_df["stats"], meta="id")
        final_df = pd.concat([teams_df.drop(columns="stats"), stats_df], axis=1)
        flat_df = final_df.pivot(
            index=["id", "teamId", "team"], columns="category", values="stat"
        ).reset_index()
        return flat_df

    def load_game_team_stats_from_pkl_at_year_week(
        self, year: int, week: int
    ) -> pd.DataFrame:
        """
        Loads the game team stats directly from the pkl.

        Args:
            year (int): Season.
            week (int): Week of interest.

        Returns:
            pd.DataFrame: Pickled data in DataFrame form.
        """
        path = self._get_pkl_path(year, week)
        if not os.path.isfile(path):
            self.fetch_and_pickle_game_team_stats_at_year_week(year, week)

        with open(path, "rb") as f:
            game_team_stats_list = pkl.load(f)

        year_df = pd.concat(
            [self._gts_expand_to_df(gts) for gts in game_team_stats_list]
        ).applymap(lambda x: None if pd.isna(x) else x)

        for col in [
            "interceptionTDs",
            "interceptionYards",
            "passesIntercepted",
            "puntReturnTDs",
            "puntReturnYards",
            "puntReturns",
        ]:
            if col not in year_df.columns:
                year_df[col] = None

        # Organize order of columns
        year_df = year_df[
            [
                "id",
                "teamId",
                "team",
                "completionAttempts",
                "firstDowns",
                "fourthDownEff",
                "fumblesLost",
                "fumblesRecovered",
                "interceptionTDs",
                "interceptionYards",
                "interceptions",
                "kickReturnTDs",
                "kickReturnYards",
                "kickReturns",
                "kickingPoints",
                "netPassingYards",
                "passesIntercepted",
                "passingTDs",
                "possessionTime",
                "rushingAttempts",
                "rushingTDs",
                "rushingYards",
                "thirdDownEff",
                "totalPenaltiesYards",
                "totalYards",
                "turnovers",
                "yardsPerPass",
                "yardsPerRushAttempt",
                "puntReturnTDs",
                "puntReturnYards",
                "puntReturns",
            ]
        ]
        return year_df

    def upload_game_team_stats_to_db(self, start: int = 2013, end: int = 2025) -> None:
        """
        Uploads the game data from pkl to PostgreSQL.

        Args:
            start (int): Start season.
            end (int): Ending season, not included.
        """
        query = """
            INSERT INTO cfb.game_team_stats
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        for year in range(start, end):
            # 2020 had an additional week 20, but inexplicably does not exist in the API
            for week in range(1, 17):
                # 2015 week 16 is also inexplicably missing
                if (
                    year in [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
                    and week == 16
                ):
                    continue
                print(f"Uploading game team stats for {year}, week {week}...")
                data = self.load_game_team_stats_from_pkl_at_year_week(year, week)
                insert_data_to_db(query, data)


if __name__ == "__main__":
    # python src/cfb/data/pull_game_team_data.py
    pipeline = CFBGameTeamData()
    pipeline.upload_game_team_stats_to_db()
