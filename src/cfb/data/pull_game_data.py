import cfbd
from cfbd.rest import ApiException
import pickle as pkl
import os

key = "9r80iKHpwhYgeDQfwlv3BqQ7Tf/Yy0PLtN4D1L5s2VYRs+ulL/lJ5MCfTKnG8Xu3"

# steps:

# if pkl file does not exist, generate pkl file using API
# pull venue data, pull game_data, 

def pickle_games_at_year(year):
    path = f"pkl/games_{year}.pkl"
    if os.path.isfile(path):
        raise Exception(f"{path} already exists.")
    configuration = cfbd.Configuration(
        host = "https://apinext.collegefootballdata.com",
        access_token = key
        )
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        try:
            api_response = api_instance.get_games(year=2024)
            with open(path, 'wb') as f:
                pkl.dump(api_response, f)
        except ApiException as e:
            print("Exception when calling GamesApi->get_games: %s\n" % e)


def get_games_at_year(year):
    path = f"pkl/games_{year}.pkl"
    if not with open("pkl/games_2024.pkl", "rb") as f:
        data = pkl.load(f)