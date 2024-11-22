import datetime as dt
import pandas as pd
import re
import sys
from bs4 import BeautifulSoup
from io import StringIO
from urllib.request import urlopen
import sys
sys.path.insert(0, '../..')
from db_utils import generate_unique_game_id, insert_data

def is_valid_date(date_string: str) -> bool:
    """
    Checks for valid date_string.

    Args:
        date_string (str): Takes in date, checks if valid for CFB games

    Returns:
        bool: Returns whether or not the date string is legitimate
    """
    # Regex pattern for "Month Day, Year" with an optional day of the week
    pattern = r'^(?:\w{3,9},\s)?\w{3,9}\s\d{1,2},\s\d{4}$'
    return bool(re.match(pattern, date_string))

def get_rank_from_row(row: pd.Series):
    pattern2 = r"^(.*?)\s\((\d{1,2})\)$"
    university = row[0]
    match = re.fullmatch(pattern2, university)
    if match:
         return match.group(1), int(match.group(2))
    else:
         return university, None

def clean_df(df: pd.DataFrame, date: dt.datetime) -> pd.DataFrame: 
    """
    Cleans the scraped dataframe of games.

    Args:
        df (pd.DataFrame): Web scraped dataframe
        date (dt.datetime): Given date

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    pattern = "^[A-Za-z]{3} \d{1,2}/\d{2}$"
    match_string = df.iloc[0][0]
    if re.fullmatch(pattern, match_string):
        match_string.to_strftime()

    # The last two rows will always be home, followed by visitor. At times, there will be a bowl header row
    home_row = df.iloc[-2]
    visitor_row = df.iloc[-1]
    home, home_rank = get_rank_from_row(home_row)
    home_points = home_row[1]
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

def get_daily_games_at_date(date: dt.date) -> pd.DataFrame:
    """
    Gets the entire cleaned slate of CFB games at a given date

    Args:
        date (dt.date): Date of interest

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    base_url = "https://www.sports-reference.com/cfb/boxscores/index.cgi"
    url = f"{base_url}?month={date.month}&day={date.day}&year={date.year}"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="html.parser")
    cand_divs = soup.find_all(attrs={"class": re.compile("game_summaries")})
    date_format = "%A, %B %d, %Y"
    all_games_table = None

    for cand_div in cand_divs:
        h2_text = cand_div.find('h2').text
        if is_valid_date(h2_text):
            date = dt.datetime.strptime(h2_text, date_format)
            html_io = StringIO(str(cand_div))
            all_games_table = pd.read_html(html_io)
    if not all_games_table:
        # If no new games that day, return None
        return None
    game_list = []
    for game in all_games_table:
        game_list.append(clean_df(game, date))
    return pd.concat(game_list)

    
def insert_daily_games_at_date(date: dt.date) -> None:
    """
    Given a date, scrapes all CFB games on that date at once. Logs the final score and any rankings. Creates log entries.

    Args:
        date (dt.date): Date of interest.
    """
    get_table_function = get_daily_games_at_date
    params = {
        'date': date
    }
    query = """
        INSERT INTO cfb.all_games(date, home, home_points, visitor, visitor_points, home_rank, visitor_rank, unique_id)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    log_query = """
        INSERT INTO cfb.all_games_log(unique_id, date)
        VALUES %s
        ON CONFLICT (unique_id, date) 
        DO UPDATE SET
            all_quarters_scrape = EXCLUDED.all_quarters_scrape
    """
    insert_data(get_table_function=get_table_function, params=params, query=query, log_query=log_query)

def backfill_games(starting_date: dt.date=dt.date(2014, 1, 1)) -> None:
    # Should automatically know the starting point, check each day and see if the number of games matches the number of games on the date, and start from there
    starting_date = dt.date(2014, 1, 1)
    pass
