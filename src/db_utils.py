import hashlib
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import psycopg2
import warnings
import yaml
from typing import List, Optional


def load_config(config_path: str = "../../config/config.yaml") -> dict[str]:
    """
    Loads config file.

    Args:
        config_path: Path of the config

    Returns:
        dict[str]: Config from the file
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def execute_sql_script(sql_file_path: str) -> None:
    """
    Takes in a .sql file and creates the table or schema as desired.

    Args:
        sql_file_path (str): Sql file path of the desired query.
    """
    assert os.path.exists(sql_file_path), "Sql file path does not exist"
    config = load_config()
    db_config = config["database"]
    conn = psycopg2.connect(**db_config)
    try:
        with open(sql_file_path, 'r') as file:
                sql_commands = file.read()

        with conn.cursor() as cursor:
                cursor.execute(sql_commands)
                conn.commit()
    except Exception as e:
            conn.rollback()
            print(e)
    return None


def generate_unique_player_id(row: pd.Series) -> str:
    """
    Takes player and team and generates a unique hash.

    Args:
        row (pd.Series): Row of player data. Necessarily has a Player and Team column.

    Returns:
        str: Unique hash.
    """
    assert (
        "Player" in row.index and "Team" in row.index
    ), 'Row missing at least one of "Player", "Team"'
    combined_values = f'{row["Player"]}{row["Team"]}'
    hash_value = hashlib.md5(combined_values.encode()).hexdigest()
    return hash_value


def generate_unique_game_id(row: pd.Series) -> str:
    """
    Takes date, visitor team, and home team and generates a unique hash.

    Args:
        row (pd.Series): Row of player data. Necessarily has a Date, Visitor, and Home column.

    Returns:
        str: Unique hash.
    """
    assert (
        "Date" in row.index and "Visitor" in row.index and "Home" in row.index
    ), 'Row missing at least one of "Date", "Visitor", "Home"'
    combined_values = f'{row["Date"]}{row["Visitor"]}{row["Home"]}'
    hash_value = hashlib.md5(combined_values.encode()).hexdigest()
    return hash_value


def group_contiguous_dates(dates: List[str]) -> List[str]:
    """
    Groups list of dates into ranges.

    Args:
        dates (List[str]): List of dates to group.

    Returns:
        List[str]: List of ranges.
    """
    # Sort the dates
    sorted_dates = sorted(set(dates))

    # Group contiguous dates including weekends and ignoring two-day gaps
    groups = []
    group = [sorted_dates[0]]
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] - group[-1] <= dt.timedelta(days=2):
            group.append(sorted_dates[i])
        else:
            groups.append(group)
            group = [sorted_dates[i]]
    groups.append(group)

    # Format the groups
    formatted_groups = []
    for group in groups:
        if len(group) == 1:
            formatted_groups.append(group[0].strftime("%m/%d/%y"))
        else:
            formatted_groups.append(
                f"{group[0].strftime('%m/%d/%y')}-{group[-1].strftime('%m/%d/%y')}"
            )
    return formatted_groups


def retrieve_data(query: str, params: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """
    Wrapper for pulling from the database given a query

    Args:
        query (str): Query for the database
        params (dict): Params for query

    Returns:
        Optional[pd.DataFrame]: Data if available from database.
    """
    config = load_config()
    db_config = config["database"]
    try:
        conn = psycopg2.connect(**db_config)
    except psycopg2.OperationalError as e:
        print("Failure to connect to database:", e)
        return None

    data = None
    try:
        with warnings.catch_warnings(action="ignore"):
            data = pd.read_sql_query(query, conn, params)
    except psycopg2.Error as e:
        print("Error executing query:", e)

    # Close the cursor and connection
    conn.close()

    return data
