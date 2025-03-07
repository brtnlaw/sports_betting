import cfbd
from cfbd.rest import ApiException
from dotenv import load_dotenv
import pandas as pd
import pickle as pkl
from db_utils import insert_data
import os
import warnings

load_dotenv()
CFBD_API_KEY = os.getenv("CFBD_API_KEY")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
warnings.simplefilter(action='ignore', category=FutureWarning)

def pickle_games_at_year(year: int) -> None:
    """
    Generates pkl file from CFDB for game data per year.

    Args:
        year (int): Season.

    Raises:
        Exception: To not overload the API, checks if the file exists.
    """
    path = os.path.join(PROJECT_ROOT, f"src/cfb/data/pkl/games_{year}.pkl")
    if os.path.isfile(path):
        raise Exception(f"{path} already exists.")
    configuration = cfbd.Configuration(
        host = "https://apinext.collegefootballdata.com",
        access_token = CFBD_API_KEY
        )
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        try:
            api_response = api_instance.get_games(year=2024)
            with open(path, 'wb') as f:
                pkl.dump(api_response, f)
        except ApiException as e:
            print("Exception when calling GamesApi->get_games: %s\n" % e)


def get_games_at_year_from_pkl(year: int) -> pd.DataFrame:
    """
    Gets the game data for a given year from the pkl file in a DataFrame.

    Args:
        year (int): Season.

    Returns:
        pd.DataFrame: DataFrame with season's game data and information.
    """
    path = os.path.join(PROJECT_ROOT, f"src/cfb/data/pkl/games_{year}.pkl")
    if not os.path.isfile(path):
        pickle_games_at_year(path)
    with open(path, "rb") as f:
        game_list = pkl.load(f)
    game_df_list = [pd.DataFrame([game.to_dict()]) for game in game_list]
    year_df = pd.concat(game_df_list)

    # Organize order of columns
    year_df = year_df[[
            'id', 'season', 'week', 'seasonType', 'startDate', 'startTimeTBD',
            'completed', 'neutralSite', 'conferenceGame', 'attendance', 'venueId',
            'venue', 'homeId', 'homeTeam', 'homeConference', 'homeClassification',
            'homePoints', 'homeLineScores', 'homePostgameWinProbability',
            'homePregameElo', 'homePostgameElo', 'awayId', 'awayTeam',
            'awayConference', 'awayClassification', 'awayPoints', 'awayLineScores',
            'awayPostgameWinProbability', 'awayPregameElo', 'awayPostgameElo',
            'excitementIndex', 'highlights', 'notes'       
        ]]
    return year_df


def upload_all_games(start: int = 2013, end: int = 2025) -> None:
    """
    Loops from start to end and uploads seasonal game data into PostgreSQL.

    Args:
        start (int): Starting season. Defaults to 2013.
        end (int): Ending season, itself not included in the range. Defaults to 2025.
    """
    query = """
        INSERT INTO cfb.games
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    for year in range(start, end):
        print(f"Inserting games data from {year}.")
        data = get_games_at_year_from_pkl(year)
        insert_data(query, data)

if __name__ == "__main__":
    # python src/cfb/data/pull_game_data.py
    upload_all_games()
