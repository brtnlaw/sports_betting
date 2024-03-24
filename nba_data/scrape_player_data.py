import hashlib
import pandas as pd
import psycopg2
import re
import sys
from bball_utils import generate_unique_player_id
from bs4 import BeautifulSoup
from io import StringIO
from psycopg2.extras import execute_values
from urllib.request import urlopen

def clean_player_table(player_table: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleans duplicate values, fixes NaN's in player data.

    Args: 
        player_table (pd.DataFrame): Raw dataframe from basketball-reference table.

    Returns:
        pd.DataFrame: Cleaned table.
    '''
    # Remove the total rows that aggregate switching teams / positions
    player_table = player_table[~(player_table['Tm'] == 'TOT')]

    # Fill NaN values
    player_table = player_table.fillna(0)

    # Remove random headers
    player_table = player_table[~(player_table['Player'] == 'Player')]
    player_table = player_table.rename(columns={'Tm': 'Team'})
    return player_table


def get_player_id_table(year: int) -> pd.DataFrame:
    '''
    Gets data for all the players that played a given season.

    Args: 
        year (int): The year in which the given season started.

    Returns:
        pd.DataFrame: All of the players who played that season.
    '''
    url = f'https://www.basketball-reference.com/leagues/NBA_%(year)s_totals.html' % {'year': str(int(year)-1)}
    html = urlopen(url)
    soup = BeautifulSoup(html, features='html.parser')
    tables = soup.find_all('table', {'id': re.compile('totals_stats')})

    # Convert the tables into a string and wrap it with StringIO
    html_string = "\n".join(str(table) for table in tables)
    html_io = StringIO(html_string)

    # Use read_html with the StringIO object; remove 'Reserves' and 'Team Totals' rows, fill in NaN
    player_table = pd.read_html(html_io, header=0)[0]
    player_table = clean_player_table(player_table)

    # Formats into PostgreSQL format, adds Unique_ID
    player_table['Unique_ID'] = player_table.apply(generate_unique_player_id, axis=1)
    return player_table[['Player', 'Team', 'Unique_ID']]


def insert_player_id_table(year: int) -> None:
    '''
    Commits player_id_table to the basketball.public_id table. Run this to synthesize a year of player data.

    Args: 
        year (int): Desired year to get player data for.
    '''
    query = '''
        INSERT INTO basketball.public_id(player, team, unique_id)
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
        print("Failure to connect to database.")
        
    player_table = get_player_id_table(year)
    
    # Split up each row value into tuples
    row_tuples = [tuple(row) for row in player_table.values]

    with conn.cursor() as cursor:
        try:
            execute_values(cursor, query, row_tuples)
            print('Successfully executed %(year)s player_id data' % {'year': year})
        except Exception as e:
            print(e)

    # Commit and close connection
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # python nba_data/scrape_player_data.py {year}
    insert_player_id_table(sys.argv[1])
    