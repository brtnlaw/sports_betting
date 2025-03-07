from dotenv import load_dotenv
import os
import pandas as pd
import psycopg2
import psycopg2.extras
from typing import Optional
import warnings
import yaml

load_dotenv
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())


def load_config(config_path: str = os.path.join(PROJECT_ROOT, "config/config.yaml")) -> dict[str]:
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


def retrieve_data(query: str, params: Optional[dict] = None) -> Optional[pd.DataFrame]:
    # TODO: Perhaps consider instead of passing in the query every time, doing something like "cfb" "games" just for ease of use.
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


def insert_data(query, data):
    config = load_config()
    db_config = config["database"]
    try:
        conn = psycopg2.connect(**db_config)
    except:
        print("Failure to connect to database.")

    row_tuples = [tuple(row) for row in data.values]
    with conn.cursor() as cursor:
        try:
            psycopg2.extras.execute_values(cursor, query, row_tuples)
            print("Successfully inserted data.")
        except Exception as e:
            print(e)
    conn.commit()
    conn.close()