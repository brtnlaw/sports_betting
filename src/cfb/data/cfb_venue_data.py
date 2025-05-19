import os
import pickle as pkl

import cfbd
import pandas as pd
from cfbd.rest import ApiException

from db_utils import insert_data_to_db


class CFBVenueData:
    """Handles fetching, storing, and uploading CFB venue data. Also retrieves from PostgreSQL."""

    def __init__(self):
        """Loads the API keys, as well as configures the API connection to CFBD."""
        super().__init__()
        self._pkl_path = os.path.join(
            self.project_root, f"src/cfb/data/pkl_files/venues.pkl"
        )
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
            print(f"Successfully pickled to {self._pkl_path}.")
        except ApiException as e:
            print(f"Error fetching venues: {e}")

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
    # python src/cfb/data/cfb_venue_data.py
    pipeline = CFBVenueData()
    pipeline.upload_venues_to_db()
