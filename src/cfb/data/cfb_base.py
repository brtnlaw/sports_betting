import os
import warnings

import cfbd
from dotenv import load_dotenv


class CFBBase:
    def __init__(self):
        """Base class to load API keys and set up the CFBD API connection."""
        load_dotenv()
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.configuration = cfbd.Configuration(
            host="https://apinext.collegefootballdata.com",
            access_token=os.getenv("CFBD_API_KEY"),
        )
        self.project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        self.api_client = cfbd.ApiClient(self.configuration)
