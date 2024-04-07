import datetime as dt
import pandas as pd
import psycopg2
import pytz
import re
import sys
import time
import warnings
from bball_utils import TEAM_CODE_DICT, generate_unique_game_id
from bs4 import BeautifulSoup
from io import StringIO
from psycopg2.extras import execute_values
from psycopg2.extensions import AsIs
from typing import List
from urllib.request import urlopen

def clean_all_games_table(all_games_table: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean up the all_games_table into proper format.

    Args:
        all_games_table (pd.DataFrame): Raw all_games_table from basketball-reference.

    Returns:
        pd.DataFrame: Clean table.
    '''
    # Remove rows where there's no PTS (see: Apr 2020)
    all_games_table = all_games_table[~all_games_table['PTS'].isna()]
    if len(all_games_table) == 0:
        return None

    # Date to date object
    all_games_table['Date'] = all_games_table['Date'].apply(lambda x: dt.datetime.strptime(x, "%a, %b %d, %Y").date())

    # Map teams to their standard three letter code
    all_games_table['Visitor/Neutral'] = all_games_table['Visitor/Neutral'].map(TEAM_CODE_DICT)
    all_games_table['Home/Neutral'] = all_games_table['Home/Neutral'].map(TEAM_CODE_DICT)

    # Drop useless columns, rename
    all_games_table.drop(columns=['Unnamed: 6', 'Arena', 'Notes'], inplace=True)
    all_games_table = all_games_table.rename(columns={'PTS': 'VisitorPTS', 'PTS.1': 'HomePTS', 'Unnamed: 7': 'OT', 'Visitor/Neutral': 'Visitor', 'Home/Neutral': 'Home'})

    # Need to fill NaN with Null
    all_games_table.fillna(AsIs('NULL'), inplace=True)

    # Turn start_time into a time object, need to pad with a 0 if not one
    def convert_to_est(time_str: str) -> dt.time:
        '''
        Converts something like 7:30p to a time object in EST

        Args:
            time_str (str): String in the format of like 7:30p.

        Returns:
            dt.date: Proper time object.
        '''
        # Preprocess the time string to add leading zero, add m
        if len(time_str) < 6:
            time_str = '0' + time_str
        time_str = time_str + 'm'

        # Parse the time string
        time_obj = dt.datetime.strptime(time_str.upper(), '%I:%M%p')
        
        # Set the timezone to EST
        est = pytz.timezone('America/New_York')
        time_obj = est.localize(time_obj)
        return time_obj.time()
    
    all_games_table['Start (ET)'] = all_games_table['Start (ET)'].apply(convert_to_est)
    return all_games_table  

def get_all_games_table(month: str, year: int) -> List[str]:
    '''
    Retrieve URLs for NBA games for a specific season.

    Args:
        month (str): Full name of month for which you want to query data.
        year (int): The year which the season began.

    Returns:
        List[str]: A list of NBA games for the specified year.
    '''
    month = month.lower()

    # URL to scrape, notice f string:
    url = f'https://www.basketball-reference.com/leagues/NBA_%(year)s_games-%(month)s.html' % {'year': year, 'month': month}
    try:
        html = urlopen(url)
    except Exception as e:
        print(e)
        print('Failure: %(month)s, %(year)s all_games URL not pulled' % {'month': month.capitalize(), 'year': year})
        return None
    
    # Kicks you out if you request over 20 times over a minute
    time.sleep(3.5)

    soup = BeautifulSoup(html, features='html.parser')

    # Convert the tables into a string and wrap it with StringIO
    tables = soup.find_all('table', {'id': re.compile('schedule')})
    html_string = "\n".join(str(table) for table in tables)
    html_io = StringIO(html_string)

    all_games_table = pd.read_html(html_io, header=0)[0]
    all_games_table = clean_all_games_table(all_games_table)
    if all_games_table is not None:
        all_games_table['Unique_ID'] = all_games_table.apply(generate_unique_game_id, axis=1)
    return all_games_table

def insert_all_games_table(month: str, year: int) -> None:
    '''
    Commits all_games table to the basketball.all_games table. Run this to synthesize a month of player data.

    Args: 
        month (str): Desired month of season.
        year (int): Desired year to get player data for.
    '''
    query = '''
        INSERT INTO basketball.all_games(date, start_time, visitor, visitor_points, home, home_points, ot, attendance, unique_id)
        VALUES %s
        ON CONFLICT DO NOTHING
    '''

    # Connect to postgres db
    try:
        conn = psycopg2.connect(
            dbname='sports_data',
            user='postgres',
            password='postgres',
            host='localhost',
            port='5432'
        )
    except:
        print('Failure to connect to database.')
        
    all_games_table = get_all_games_table(month, year)
    if all_games_table is None:
        print('Failure: %(month)s, %(year)s all_games data does not exist' % {'month': month.capitalize(), 'year': year})
        conn.close()
        return

    # Split up each row value into tuples
    row_tuples = [tuple(row) for row in all_games_table.values]

    with conn.cursor() as cursor:
        try:
            execute_values(cursor, query, row_tuples)
            print('Successfully executed %(month)s, %(year)s all_games data' % {'month': month.capitalize(), 'year': year})
        except Exception as e:
            print(e)

    # Commit and close connection
    conn.commit()
    conn.close()

def insert_all_games_table_year(year: int) -> None:
    '''
    Inserts all_games for entire year.

    Args:
        year (int): Desired year to insert for.
    '''
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
    for month in months:
        insert_all_games_table(month, year)


if __name__ == '__main__':
    # python nba_data/scrape_all_games_data.py {year}
    warnings.simplefilter(action='ignore', category=FutureWarning)
    insert_all_games_table_year(sys.argv[1])