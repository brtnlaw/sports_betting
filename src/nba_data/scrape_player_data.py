import pandas as pd
import psycopg2
import re
import sys
import time
from bball_utils import generate_unique_player_id, load_config
from bs4 import BeautifulSoup
from io import StringIO
from psycopg2.extras import execute_values
from urllib.request import urlopen

config = load_config()


def clean_player_table(player_table: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans duplicate values, fixes NaN's in player data.

    Args:
        player_table (pd.DataFrame): Raw dataframe from basketball-reference table.

    Returns:
        pd.DataFrame: Cleaned table.
    """
    # Remove the total rows that aggregate switching teams / positions
    player_table = player_table[~(player_table["Tm"] == "TOT")]

    # Fill NaN values
    player_table = player_table.fillna(0)

    # Remove random headers
    player_table = player_table[~(player_table["Player"] == "Player")]

    # Remove random alphanumerics from end of string of Player
    player_table["Player"] = player_table["Player"].apply(
        lambda x: x[:-1] if x.endswith("*") else x
    )

    player_table = player_table.rename(columns={"Tm": "Team"})
    return player_table


def get_player_id_table(year: int) -> pd.DataFrame:
    """
    Gets data for all the players that played a given season.

    Args:
        year (int): The year in which the given season started.

    Returns:
        pd.DataFrame: All of the players who played that season.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_%(year)s_totals.html" % {
        "year": str(year)
    }
    html = urlopen(url)

    # Kicks you out if you request over 20 times over a minute
    time.sleep(3.5)

    soup = BeautifulSoup(html, features="html.parser")
    tables = soup.find_all("table", {"id": re.compile("totals_stats")})

    # Convert the tables into a string and wrap it with StringIO
    html_string = "\n".join(str(table) for table in tables)
    html_io = StringIO(html_string)

    # Use read_html with the StringIO object; remove 'Reserves' and 'Team Totals' rows, fill in NaN
    player_table = pd.read_html(html_io, header=0)[0]
    player_table = clean_player_table(player_table)

    # Formats into PostgreSQL format, adds Unique_ID
    player_table["Unique_ID"] = player_table.apply(generate_unique_player_id, axis=1)
    return player_table[["Player", "Team", "Unique_ID"]]


def insert_player_id_table(year: int) -> None:
    """
    Commits player_id_table to the basketball.player_id table. Run this to synthesize a year of player data.

    Args:
        year (int): Desired year to get player data for.
    """
    query = """
        INSERT INTO basketball.player_id(player, team, unique_id)
        VALUES %s
        ON CONFLICT DO NOTHING
    """

    db_config = config["database"]
    try:
        conn = psycopg2.connect(
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"],
        )
    except:
        print("Failure to connect to database.")

    player_table = get_player_id_table(year)

    # Split up each row value into tuples
    row_tuples = [tuple(row) for row in player_table.values]

    with conn.cursor() as cursor:
        try:
            execute_values(cursor, query, row_tuples)
            print("Successfully executed %(year)s player_id data" % {"year": year})
        except Exception as e:
            print(e)

    # Commit and close connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # python nba_data/scrape_player_data.py {year}
    insert_player_id_table(sys.argv[1])
