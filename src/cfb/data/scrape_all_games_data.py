import datetime as dt
import pandas as pd
import psycopg2
import pytz
import re
import sys
import time
import warnings
from bs4 import BeautifulSoup
from io import StringIO
from psycopg2.extras import execute_values
from psycopg2.extensions import AsIs
from typing import List
from urllib.request import urlopen
import sys
sys.path.insert(0, '../..')
from db_utils import generate_unique_game_id

def is_valid_date(date_string):
    # Regex pattern for "Month Day, Year" with an optional day of the week
    pattern = r'^(?:\w{3,9},\s)?\w{3,9}\s\d{1,2},\s\d{4}$'
    return bool(re.match(pattern, date_string))

def get_rank_from_row(row):
    pattern2 = r"^(.*?)\s\((\d{1,2})\)$"
    university = row[0]
    match = re.fullmatch(pattern2, university)
    if match:
         return match.group(1), int(match.group(2))
    else:
         return university, None

def clean_df(df: pd.DataFrame, date: dt.datetime): 
    pattern = "^[A-Za-z]{3} \d{1,2}/\d{2}$"
    match_string = df.iloc[0][0]
    if re.fullmatch(pattern, match_string):
        match_string.to_strftime()

    home_row = df.iloc[1]
    home, home_rank = get_rank_from_row(home_row)
    home_points = home_row[1]
    visitor_row = df.iloc[2]
    visitor, visitor_rank = get_rank_from_row(visitor_row)
    visitor_points = visitor_row[1]
    
    id_row = pd.Series({
        'Date': [date],
        'Home': [home],
        'Visitor': [visitor]
    })

    return pd.DataFrame({
        'date': [date],
        'home': [home],
        'home_points': [home_points],
        'visitor': [visitor],
        'visitor_points': [visitor_points],
        'home_rank': [home_rank],
        'visitor_rank': [visitor_rank],
        'unique_id': [generate_unique_game_id(id_row)]
    })

def get_daily_games_at_url(url: str):
    html = urlopen(url)
    soup = BeautifulSoup(html, features="html.parser")
    cand_divs = soup.find_all(attrs={"class": re.compile("game_summaries")})
    date_format = "%A, %B %d, %Y"

    for cand_div in cand_divs:
        h2_text = cand_div.find('h2').text
        if is_valid_date(h2_text):
            date = dt.datetime.strptime(h2_text, date_format)
            html_io = StringIO(str(cand_div))
            all_games_table = pd.read_html(html_io)
            break
    
    game_list = []
    for game in all_games_table:
        game_list.append(clean_df(game), date)
    return pd.concat(game_list)
    


def get_all_games_table(month: str, year: int) -> List[str]:
    """
    Retrieve URLs for NBA games for a specific season.

    Args:
        month (str): Full name of month for which you want to query data.
        year (int): The year which the season began.

    Returns:
        List[str]: A list of NBA games for the specified year.
    """
    month = month.lower()

    # URL to scrape, notice f string:
    url = (
        f"https://www.basketball-reference.com/leagues/NBA_%(year)s_games-%(month)s.html"
        % {"year": year, "month": month}
    )
    try:
        html = urlopen(url)
    except Exception as e:
        print(e)
        print(
            "Failure: %(month)s, %(year)s all_games URL not pulled"
            % {"month": month.capitalize(), "year": year}
        )
        return None

    # Kicks you out if you request over 20 times over a minute
    time.sleep(3.5)

    soup = BeautifulSoup(html, features="html.parser")

    # Convert the tables into a string and wrap it with StringIO
    tables = soup.find_all("table", {"id": re.compile("schedule")})
    html_string = "\n".join(str(table) for table in tables)
    html_io = StringIO(html_string)

    all_games_table = pd.read_html(html_io, header=0)[0]
    all_games_table = clean_all_games_table(all_games_table)
    if all_games_table is not None:
        all_games_table["Unique_ID"] = all_games_table.apply(
            generate_unique_game_id, axis=1
        )
    return all_games_table

