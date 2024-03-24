from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
from types import List

# https://www.basketball-reference.com/boxscores/202110190MIL.html
# how do we get all the above links? think have to parse https://www.basketball-reference.com/leagues/NBA_2022_games.html

def get_all_nba_games(year: int) -> List[str]:
    """
    Retrieve URLs for NBA games for a specific year.

    Parameters:
        year (int): The year for which NBA games are to be scraped.

    Returns:
        List[str]: A list of NBA games for the specified year.
    """

    # URL to scrape, notice f string:
    url = "https://www.basketball-reference.com/boxscores/202110190MIL.html"
    html = urlopen(url)

    soup = BeautifulSoup(html, features="html.parser")
    months = ["October", "November", "December", "January", "February", "March", "April", "May", "June"]

    


def scrape_NBA_team_data():

    # URL to scrape, notice f string:
    url = "https://www.basketball-reference.com/boxscores/202110190MIL.html"

    # collect HTML data
    html = urlopen(url)

    # create beautiful soup object from HTML
    soup = BeautifulSoup(html, features="html.parser")

    tables = soup.find_all('table', {'id': re.compile('box-.*-game-basic')})

    team_data = pd.read_html(str(tables), header=1)

    # export to csv
    