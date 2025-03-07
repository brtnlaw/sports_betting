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

def pickle_venues() -> None:
    """
    Generates pkl file from CFDB for venue data.

    Raises:
        Exception: To not overload the API, checks if the file exists.
    """
    path = os.path.join(PROJECT_ROOT, f"src/cfb/data/pkl/venues.pkl")
    if os.path.isfile(path):
        raise Exception(f"{path} already exists.")
    configuration = cfbd.Configuration(
        host = "https://apinext.collegefootballdata.com",
        access_token = CFBD_API_KEY
        )
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.VenuesApi(api_client)
        try:
            api_response = api_instance.get_venues()
            with open(path, 'wb') as f:
                pkl.dump(api_response, f)
        except ApiException as e:
            print("Exception when calling VenuesApi->get_venues: %s\n" % e)


def get_venues_from_pkl() -> pd.DataFrame:
    """
    Gets the venue data from the pkl file in a DataFrame.

    Returns:
        pd.DataFrame: DataFrame with venue data and information.
    """
    path = os.path.join(PROJECT_ROOT, f"src/cfb/data/pkl/venues.pkl")
    if not os.path.isfile(path):
        pickle_venues(path)
    with open(path, "rb") as f:
        venue_list = pkl.load(f)
    venue_df_list = [pd.DataFrame([game.to_dict()]) for game in venue_list]
    venue_df = pd.concat(venue_df_list)

    # Organizes order of columns
    venue_df = venue_df[[
        'id', 'name', 'city', 'state', 'zip', 'countryCode', 'latitude',
        'longitude', 'capacity', 'dome', 'timezone', 'elevation',
        'constructionYear', 'grass'
    ]]
    return venue_df


def upload_venues() -> None:
    """
    Uploads venue data into PostgreSQL.
    """
    query = """
        INSERT INTO cfb.venues
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    print(f"Inserting venue data.")
    data = get_venues_from_pkl()
    insert_data(query, data)

if __name__ == "__main__":
    # python src/cfb/data/pull_venue_data.py
    upload_venues()
