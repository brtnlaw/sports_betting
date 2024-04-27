import pandas as pd
import datetime as dt
import pandas as pd
import psycopg2
import warnings
from scrape_player_data import get_player_id_table
from scrape_all_games_data import get_all_games_table
from scrape_stat_sheet_data import get_games_between
import contextlib
from bball_utils import group_contiguous_dates
import os


def get_missing_player_data(start: int, end: int) -> None:
    """
    Prints out missing players data between start and end years.

    Args:
        start (int): The starting year of the query.
        end (int): The ending year of the query, inclusive.
    """
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

    query = "SELECT * FROM basketball.player_id"
    with warnings.catch_warnings(action="ignore"):
        data = pd.read_sql_query(query, conn)

    # Close the cursor and connection
    conn.close()

    missing_df_list = []
    for year in range(start, end + 1):
        player_table = get_player_id_table(year).rename(
            columns={"Unique_ID": "unique_id"}
        )

        # Merge for any players that exist only in the online player table and not the database
        missing_df = player_table.merge(
            data.drop_duplicates(), on="unique_id", how="left", indicator=True
        )
        missing_df_list.append(missing_df[missing_df["_merge"] == "left_only"])

    total_missing_df = pd.concat(missing_df_list)
    total_missing_df.drop_duplicates(subset=["unique_id"], inplace=True)

    print("=" * 70)
    print(
        f"Missing {len(total_missing_df)} players from database from {start} to {end}:"
    )
    if len(total_missing_df) > 0:
        for index, row in total_missing_df.reset_index(drop=True).iterrows():
            print(f'\t{row["Player"]} on {row["Team"]}')
            if index > 10:
                print(f"\tAnd more!")
                break
    else:
        print("\tNone!")
    print("=" * 70)


def get_missing_game_data(start: int, end: int) -> None:
    """
    Prints out missing games data between start and end years.

    Args:
        start (int): The starting year of the query.
        end (int): The ending year of the query, inclusive.
    """
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
        data = pd.read_sql_query(query, conn)

    conn.close()

    missing_df_list = []
    months = [
        "october",
        "november",
        "december",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
    ]
    for year in (start, end):
        for month in months:
            try:
                with warnings.catch_warnings(action="ignore"), open(
                    os.devnull, "w"
                ) as f, contextlib.redirect_stdout(f):
                    games_table = get_all_games_table(month, year).rename(
                        columns={"Unique_ID": "unique_id"}
                    )
            except:
                continue

            # Merge for any players that exist only in the online player table and not the database
            missing_df = games_table.merge(
                data.drop_duplicates(), on="unique_id", how="left", indicator=True
            )
            missing_df_list.append(missing_df[missing_df["_merge"] == "left_only"])

    total_missing_df = pd.concat(missing_df_list)
    total_missing_df.drop_duplicates(subset=["unique_id"])

    print("=" * 70)
    print(f"Missing {len(total_missing_df)} games from database from {start} to {end}:")
    if len(total_missing_df) > 0:
        for index, row in total_missing_df.reset_index(drop=True).iterrows():
            print(f'\t{row["Visitor"]} at {row["Home"]} on {row["Date"]}')
            if index > 10:
                print(f"\tAnd more!")
                break
    else:
        print("\tNone!")
    print("=" * 70)


def get_missing_statsheet_data(start: int, end: int) -> None:
    """
    Prints out missing stat sheet data (that is, a player's performance in a given date) between start and end.

    Args:
        start (int): The starting year of the query.
        end (int): The ending year of the query, inclusive.
    """

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

    query = "SELECT * FROM basketball.stat_sheet"
    with warnings.catch_warnings(action="ignore"):
        data = pd.read_sql_query(query, conn)

    conn.close()

    with warnings.catch_warnings(action="ignore"), open(
        os.devnull, "w"
    ) as f, contextlib.redirect_stdout(f):
        game_data = get_games_between(
            dt.datetime(start, 1, 1), dt.datetime(end, 12, 31)
        )

    dates_missing_games = game_data[~game_data["unique_id"].isin(data["game_id"])][
        "date"
    ]
    dates_missing_games = group_contiguous_dates(dates_missing_games)

    print("=" * 70)
    print(
        f"Missing {len(dates_missing_games)} rows of stat sheet data on the following days:"
    )
    if len(dates_missing_games) > 0:
        for index, date_range in enumerate(dates_missing_games):
            print(date_range)
            if index > 10:
                print(f"\tAnd more!")
                break
    else:
        print("\tNone!")
    print("=" * 70)


if __name__ == "__main__":
    # python nba_data/check_missing_data.py
    get_missing_player_data(2020, 2024)
    get_missing_game_data(2020, 2024)
    get_missing_statsheet_data(2020, 2024)
