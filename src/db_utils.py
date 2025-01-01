import hashlib
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import psycopg2
import sqlparse
import warnings
import psycopg2.extras
import yaml
from typing import Callable, List, Optional


def load_config(config_path: str = "../../config/config.yaml") -> dict[str]:
    """
    Loads config file.

    Args:
        config_path(str): Path of the config

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
    Wrapper for pulling from the database given a query.

    Args:
        query (str): Query for the database
        params Optional[dict]: Params for query

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

def get_insert_table(query: str):
    """
    Given a query, gets the table intended to be inserted into.

    Args:
        query (str): SQL query
    """
    parsed = sqlparse.parse(query)

    # Extract table name from the first statement
    for stmt in parsed:
        # Find INSERT INTO statement and extract table name
        if stmt.get_type() == 'INSERT':
            for token in stmt.tokens:
                if isinstance(token, sqlparse.sql.Identifier):
                    table_name = token.get_real_name()  # This gets the table name without schema
                    schema_name = token.get_parent_name()  # This gets the schema name
                    return f"{schema_name}.{table_name}"
                
def insert_data(
        get_table_function: Callable, 
        params: dict,
        query: str,
        log_query: Optional[str] = None,
        log_cols: Optional[List] = None
    ) -> None: 
    """
    Framework for inserting clean data into Postgres. If no data and log_query is not None, then pastes a row with just the date.

    Args:
        get_table_function (Callable): Function which generates the clean dataframe.
        params (dict): Params that are input to the get_table_function, often date.
        query (str): Query to insert into the database.
        log_query (Optional[str], optional): Query to insert into the log database. Defaults to None.
        log_cols (Optional[List], optional): Additional columns that would be ticked True within the log. Defaults to None.
    """
    if not params["date"] and log_query:
        raise Exception("Params require date for logging.")
    
    config = load_config()
    db_config = config["database"]
    try:
        conn = psycopg2.connect(**db_config)
    except:
        print("Failure to connect to database.")

    all_data_table = get_table_function(**params)

    # If empty, insert audit placeholder for that date
    if all_data_table is None:
        if log_query:
            log_table = get_insert_table(log_query)
            empty_log_query = f"""
                INSERT INTO {log_table}(date)
                VALUES %s
                ON CONFLICT DO NOTHING
            """
            # Commits a row to the log table with only the date, Null unique_id
            with conn.cursor() as cursor:
                try:
                    psycopg2.extras.execute_values(cursor, empty_log_query, [(params["date"],)])    
                    conn.commit()
                    print("Successfully logged %(log_table)s with %(date)s - no data for this day"
                        % {"log_table": log_table, "date": params["date"]})
                except Exception as e:
                    print (e)
        conn.close()
        return
    # Split up each row value into tuples
    row_tuples = [tuple(row) for row in all_data_table.values]

    # Handle optional logging
    if log_query:
        all_log_table = all_data_table[["unique_id", "date"]]
        if log_cols:
            for log_col in log_cols:
                # Data passes the audit, i.e. column we want to log is true
                all_log_table[log_col] = True
        log_row_tuples = [tuple(row) for row in all_log_table.values]

    with conn.cursor() as cursor:
        try:
            psycopg2.extras.execute_values(cursor, query, row_tuples)
            print(
                "Successfully executed '%(name)s' with params %(params)s"
                % {"name": get_table_function.__name__, "params": params}
            )
            # Logging and data input are hand in hand
            psycopg2.extras.execute_values(cursor, log_query, log_row_tuples)
            if log_query:
                print(
                    "Successfully logged '%(name)s' with params %(params)s with log_cols %(log_cols)s"
                    % {"name": get_table_function.__name__, "params": params, "log_cols": log_cols}
                )
            else:
                print("No log query.")
        except Exception as e:
            print(e)
    conn.commit()
    conn.close()