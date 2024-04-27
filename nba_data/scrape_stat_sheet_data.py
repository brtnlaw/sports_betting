from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
from io import StringIO
import datetime as dt
import pandas as pd
import psycopg2
import re
import sys
import time
from bball_utils import (
    generate_unique_game_id,
    generate_unique_player_id,
)
from bs4 import BeautifulSoup
from io import StringIO
from psycopg2.extras import execute_values
from urllib.request import urlopen
import psycopg2
import warnings


def get_all_games() -> pd.DataFrame:
    """
    Queries PostgreSQL and gets all the entire table of games.

    Returns:
        pd.DataFrame: All of the NBA games.
    """
    # Connect to postgres db
    try:
        conn = psycopg2.connect(
            dbname="sports_data",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432",
        )
    except:
        print("Failure to connect to database.")

    query = "SELECT * FROM basketball.all_games"
    with warnings.catch_warnings(action="ignore"):
        all_game_data = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()
    return all_game_data


def get_games_between(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Queries PostgreSQL and gets games between start and end times.

    Args:
        start (dt.date): Start of the window.
        end (dt.date): End of the window.

    Returns:
        pd.DataFrame: All of the NBA games.
    """
    # Connect to postgres db
    try:
        conn = psycopg2.connect(
            dbname="sports_data",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432",
        )
    except:
        print("Failure to connect to database.")

    query = """
            SELECT * FROM basketball.all_games ag WHERE ag.date < %(end)s and ag.date > %(start)s
            """
    with warnings.catch_warnings(action="ignore"):
        all_game_data = pd.read_sql_query(
            query, conn, params={"start": start, "end": end}
        )

    # Close the connection
    conn.close()
    return all_game_data


def clean_stat_sheet_table(stat_sheet_table: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the stat table thats concatenated.

    Args:
        stat_sheet_table (pd.DataFrame): The raw stat table.

    Returns:
        pd.DataFrame: Cleaned table.
    """
    # Remove header rows
    stat_sheet_table = stat_sheet_table[
        ~stat_sheet_table["Starters"].isin(["Reserves", "Team Totals"])
    ]

    # Ignore stats of DNP players
    stat_sheet_table = stat_sheet_table[
        (stat_sheet_table["MP"] != "Did Not Play")
        & (stat_sheet_table["MP"] != "Did Not Dress")
        & (stat_sheet_table["MP"] != "Not With Team")
        & (stat_sheet_table["MP"] != "Player Suspended")
    ]

    # Create a timedelta object representing the duration
    # Function to convert time string to timedelta
    def parse_time(time_str):
        minutes, seconds = map(int, time_str.split(":"))
        return dt.timedelta(minutes=minutes, seconds=seconds)

    stat_sheet_table["MP"] = stat_sheet_table["MP"].apply(parse_time)

    # Rename for unique ID
    stat_sheet_table = stat_sheet_table.rename(columns={"Starters": "Player"})
    stat_sheet_table = stat_sheet_table.fillna(0)
    return stat_sheet_table


def get_stat_sheet_table(date: dt.date, site: str) -> pd.DataFrame:
    """
    Get the whole stat_sheet_table of a given game, queried by date and home

    Args:
        date (dt.date): Desired date of game.
        site (str): Home team of game, in three letter code.

    Returns:
        pd.DataFrame: Gets the entire stat sheet for a given game.
    """
    url = f"https://www.basketball-reference.com/boxscores/%(date)s0%(site)s.html" % {
        "date": date.strftime("%Y%m%d"),
        "site": site,
    }
    try:
        html = urlopen(url)
    except Exception as e:
        print(e)

    # Kicks you out if you request over 20 times over a minute
    time.sleep(3.5)

    soup = BeautifulSoup(html, features="html.parser")
    tables = soup.find_all("table", {"id": re.compile("box-.*-game-basic")})

    # Extract three-letter codes from table IDs
    team_codes = [
        re.search(r"box-(\w+)-game-basic", table["id"]).group(1) for table in tables
    ]

    # Convert the tables into a string and wrap it with StringIO
    html_string = "\n".join(str(table) for table in tables)
    html_io = StringIO(html_string)
    stat_sheet_tables = pd.read_html(html_io)

    # Add Team, Opponent, Visitor, and Home teams
    for i in range(2):
        stat_sheet_tables[i] = stat_sheet_tables[i].droplevel(level=0, axis=1)
        stat_sheet_tables[i]["Team"] = team_codes[i]
        stat_sheet_tables[i]["Opponent"] = team_codes[1 - i]
        stat_sheet_tables[i]["Visitor"] = team_codes[0]
        stat_sheet_tables[i]["Home"] = team_codes[1]
        stat_sheet_tables[i]["Date"] = date

    # Put it together into one table
    stat_sheet_table = pd.concat(stat_sheet_tables)
    stat_sheet_table = clean_stat_sheet_table(stat_sheet_table)

    # Get unique ID columns
    stat_sheet_table["Player_ID"] = stat_sheet_table.apply(
        generate_unique_player_id, axis=1
    )
    stat_sheet_table["Game_ID"] = stat_sheet_table.apply(
        generate_unique_game_id, axis=1
    )
    return stat_sheet_table


def insert_stat_sheet_table(date: dt.date, site: str) -> None:
    """
    Inserts a given game, occurring on a date and at a site into PostgreSQL.

    Args:
        date (dt.date): The date of the game.
        site (str): The location of the game, three letter code.
    """
    query = """
        INSERT INTO basketball.stat_sheet(
            player, minutes_played, field_goals, field_goals_attempted, field_goal_percentage, three_pointers, three_pointers_attempted, three_pointer_percentage, 
            free_throws, free_throws_attempted, free_throw_percentage, offensive_rebounds, defensive_rebounds, total_rebounds, assists, steals, blocks, turnovers, 
            personal_fouls, points, plus_minus, team, opponent, visitor, home, date, player_id, game_id
            )
        VALUES %s
        ON CONFLICT DO NOTHING
    """

    # Connect to postgres db
    try:
        conn = psycopg2.connect(
            dbname="sports_data",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432",
        )
    except:
        print("Failure to connect to database.")

    stat_sheet_table = get_stat_sheet_table(date, site)

    row_tuples = [tuple(row) for row in stat_sheet_table.values]
    with conn.cursor() as cursor:
        try:
            execute_values(cursor, query, row_tuples)
            print(
                "Successfully executed %(date)s, %(site)s game stat_sheet data"
                % {"date": date.strftime("%Y-%m-%d"), "site": site}
            )
        except Exception as e:
            print(date, site)
            print(e)

    # Commit and close connection
    conn.commit()
    conn.close()


def insert_stat_sheet_table_all() -> None:
    """
    Inserts into basketball.stat_sheet for every game
    """
    game_df = get_all_games()
    game_df.apply(lambda x: insert_stat_sheet_table(x["date"], x["home"]), axis=1)


def insert_stat_sheet_table_period(start: str, end: str) -> None:
    """
    Inserts into basketball.stat_sheet between time period so as to not be overloaded.

    Args:
        start (str): Start in format YYYY-MM-DD
        end (str): End in format YYYY-MM-DD
    """
    start_dt = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = dt.datetime.strptime(end, "%Y-%m-%d").date()
    game_df = get_games_between(start_dt, end_dt)
    game_df.apply(lambda x: insert_stat_sheet_table(x["date"], x["home"]), axis=1)


if __name__ == "__main__":
    # python nba_data/scrape_stat_sheet_data.py {YYYY-MM-DD} {YYYY-MM-DD}
    insert_stat_sheet_table_period(sys.argv[1], sys.argv[2])
