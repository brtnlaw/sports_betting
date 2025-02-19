import re
from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
from io import StringIO
from urllib.request import urlopen
from db_utils import retrieve_data, get_rank_from_row, generate_unique_game_id
from typing import Optional


def get_quarters_data_at_date_home(date: str, home: str) -> pd.DataFrame:
    """
    Gets quarters data for a game given by its date and home team

    Args:
        date (str): Date of game.
        home (str): Home team.

    Returns:
        pd.DataFrame: Quarter score data. 
    """
    home_str = home.replace(" ", "-").lower()
    base_url = "https://www.sports-reference.com/cfb/boxscores/index.cgi"
    url = f"{base_url}/{date.strftime('%Y-%m-%d')}-{home_str}.html"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="html.parser")
    divs = soup.find_all(attrs={"class": re.compile("linescore nohover")})
    html_io = StringIO(str(divs[0]))
    whole_df = pd.read_html(html_io)

    # Get visitor for the unique id, home already given
    visitor_row = whole_df.iloc[1]
    visitor_row["Unnamed: 1"].replace(u'\xa0', ' ')
    visitor = get_rank_from_row(visitor_row, "Unnamed: 1")[0]

    unique_id = generate_unique_game_id(pd.Series({'Date:': date, 'Visitor': visitor, 'Home': home}))

    # TODO: assert that the scores line up with that in the dataset
    pass


def backfill_data_for_table() -> None:
    """
    Generates quarters data for dates in CFB table.
    """
    # Given that games are done a full day at a time, should not be possible to have partial slates logged
    date_query = """
    SELECT DISTINCT date, home
    FROM cfb.all_games_log
    WHERE all_quarters_scrape IS NOT TRUE
    ORDER BY date ASC
    """
    ungenned_games = retrieve_data(date_query)
    # Iterates through the games ordered by date, adds quarter data, then removes it from the ungenned_dates
    while ungenned_games:
        game = ungenned_games.iloc[0]
        date = game["date"]
        home = game["home"]
        print(f"Backfilling {date} at {home}")
        fill_quarters_data(date, home)
        # Drop the row with game after everything done
        ungenned_games = ungenned_games[1:]
    pass