import cfbd
from cfbd.rest import ApiException
from dotenv import load_dotenv
import pandas as pd
import pickle as pkl
from db_utils import insert_data_to_db
import os
import warnings


class CFBVenueData:
    """
    Handles fetching, storing, and uploading CFB venue data. Also retrieves from PostgreSQL.
    """

    def __init__(self):
        """
        Loads the API keys, as well as configures the API connection to CFBD.
        """
        load_dotenv()
        self.cfbd_api_key = os.getenv("CFBD_API_KEY")
        self.project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        warnings.simplefilter(action="ignore", category=FutureWarning)

        self._pkl_path = os.path.join(self.project_root, f"src/cfb/data/pkl/venues.pkl")
        self.configuration = cfbd.Configuration(
            host="https://apinext.collegefootballdata.com",
            access_token=self.cfbd_api_key,
        )
        self.api_client = cfbd.ApiClient(self.configuration)
        self.api = cfbd.VenuesApi(self.api_client)

    def fetch_and_pickle_venues(self) -> None:
        """
        Generates pkl file from CFDB for venue data.

        Raises:
            Exception: To not overload the API, checks if the file exists.
        """
        if os.path.isfile(self._pkl_path):
            raise Exception(f"{self._pkl_path} already exists.")
        try:
            venues = self.api.get_venues()
            with open(self._pkl_path, "wb") as f:
                pkl.dump(venues, f)
            print(f"Successfully pickled to {self._pkl_path}")
        except ApiException as e:
            print("Error fetching venues: {e}")

    def load_venues_from_pkl(self) -> pd.DataFrame:
        """
        Gets the venue data from the pkl file in a DataFrame.

        Returns:
            pd.DataFrame: DataFrame with venue data and information.
        """
        if not os.path.isfile(self._pkl_path):
            self.fetch_and_pickle_venues()
        with open(self._pkl_path, "rb") as f:
            venue_list = pkl.load(f)
        venue_df_list = [pd.DataFrame([game.to_dict()]) for game in venue_list]
        venue_df = pd.concat(venue_df_list)

        # Organizes order of columns
        venue_df = venue_df[
            [
                "id",
                "name",
                "city",
                "state",
                "zip",
                "countryCode",
                "latitude",
                "longitude",
                "capacity",
                "dome",
                "timezone",
                "elevation",
                "constructionYear",
                "grass",
            ]
        ]
        return venue_df

    def upload_venues_to_db(self) -> None:
        """
        Uploads venue data into PostgreSQL.
        """
        query = """
            INSERT INTO cfb.venues
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        print(f"Inserting venue data.")
        data = self.load_venues_from_pkl()
        insert_data_to_db(query, data)


if __name__ == "__main__":
    # python src/cfb/data/pull_venue_data.py
    pipeline = CFBVenueData()
    pipeline.upload_venues_to_db()
