import re
from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
from io import StringIO
from urllib.request import urlopen
from db_utils import retrieve_data, get_rank_from_row, generate_unique_game_id, insert_data
from typing import Optional
import time


def get_quarters_data_at_date_home(date: dt.date, home: str) -> pd.DataFrame:
    """
    Gets quarters data for a game given by its date and home team

    Args:
        date (str): Date of game.
        home (str): Home team.

    Returns:
        pd.DataFrame: Quarter score data. 
    """
    url_dict = {
        "UCF": "central-florida",
        "SMU": "southern-methodist",
        "TCU": "texas-christian",
        "Pitt": "pittsburgh",
        "LSU": "louisiana-state",
        "Ole Miss": "mississippi",
        "Louisiana": "louisiana-lafayette",
        "UAB": "alabama-birmingham",
        "USC": "southern-california",
        "UTSA": "texas-san-antonio",
        "UNLV": "nevada-las-vegas",
        "UTEP": "texas-el-paso"
    }
    adjust_home = lambda text: re.sub(r'[()&]', '', text).replace(" ", "-").lower()
    home_str = adjust_home(home) if home not in url_dict else url_dict[home]
    base_url = "https://www.sports-reference.com/cfb/boxscores/"
    url = f"{base_url}/{date.strftime('%Y-%m-%d')}-{home_str}.html"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="html.parser")
    divs = soup.find_all(attrs={"class": re.compile("linescore nohover")})
    html_io = StringIO(str(divs[0]))
    df = pd.read_html(html_io)[0]

    # Get visitor for the unique id, home already given
    visitor_row = df.iloc[0]
    visitor_row["Unnamed: 1"].replace(u'\xa0', ' ')
    visitor = get_rank_from_row(visitor_row, "Unnamed: 1")[0]

    # Special case mapping where key is diving into individual game, value is in the aggregate view
    # TODO: delete TCU
    naming_dict = {
        "Nevada-Las Vegas": "UNLV",
        "BYU": "Brigham Young",
        "Bowling Green": "Bowling Green State",
        "Texas Christian": "TCU"
        }
    standardized_home_id = home if home not in naming_dict else naming_dict[home]
    standardized_visitor_id = visitor if visitor not in naming_dict else naming_dict[visitor]
    # Generate unique_id to match to
    unique_id = generate_unique_game_id(pd.Series({'Date': date, 'Visitor': standardized_visitor_id, 'Home': standardized_home_id}))

    team = ["visitor", "home"]
    quarter = ["first_quarter", "second_quarter", "third_quarter", "fourth_quarter"]
    col_dict = {}
    for i in range(2):
        # Check that the final scores line up
        pt_total = 0
        for j in range(4):
            col = f"{team[i]}_{quarter[j]}"
            # Map the column to the data (indexed from 1)
            col_dict[col] = int(df[str(j+1)][i])
            pt_total += col_dict[col]

        # If we go into OT
        ot_col = f"{team[i]}_ot_total"
        if df.columns[-2] != "4":
            num_ot = df.columns[-2]
            # If there are 3 OT's, want to sum from OT1 to OT3
            ot_total = sum(df.loc[i, "OT1":num_ot])
            col_dict[ot_col] = ot_total
            pt_total += ot_total
            col_dict["OT"] = num_ot
        else:
            col_dict[ot_col] = 0
            col_dict["OT"] = None
        assert pt_total == int(df.iloc[i,-1:].iloc[0]), "Quarter points don't add up to final"

    col_dict["date"] = date
    col_dict["home"] = standardized_home_id
    col_dict["visitor"] = standardized_visitor_id
    col_dict["unique_id"] = unique_id
    
    # Kicks you out if you request over 20 times over a minute
    time.sleep(3.5)
    col_df = pd.DataFrame([col_dict])
    return col_df


def insert_quarters_data_at_date_home(date: dt.date, home: str) -> None:
    """
    Given a date, scrapes all CFB games on that date at once. Logs the final score and any rankings. Creates log entries.

    Args:
        date (dt.date): Date of interest.
        home (str): Home team.
    """
    get_table_function = get_quarters_data_at_date_home
    params = {
        'date': date,
        'home': home
    }
    query = """
        INSERT INTO cfb.all_games(visitor_first_quarter, visitor_second_quarter, visitor_third_quarter, visitor_fourth_quarter, visitor_ot_total,
                                OT, home_first_quarter, home_second_quarter, home_third_quarter, home_fourth_quarter, home_ot_total,
                                date, home, visitor, unique_id)
        VALUES %s
        ON CONFLICT (date, visitor, home)
        DO UPDATE SET
            visitor_first_quarter = EXCLUDED.visitor_first_quarter,
            visitor_second_quarter = EXCLUDED.visitor_second_quarter,
            visitor_third_quarter = EXCLUDED.visitor_third_quarter,
            visitor_fourth_quarter = EXCLUDED.visitor_fourth_quarter,
            visitor_ot_total = EXCLUDED.visitor_ot_total,
            OT = EXCLUDED.OT,
            home_first_quarter = EXCLUDED.home_first_quarter,
            home_second_quarter = EXCLUDED.home_second_quarter,
            home_third_quarter = EXCLUDED.home_third_quarter,
            home_fourth_quarter = EXCLUDED.home_fourth_quarter,
            home_ot_total = EXCLUDED.home_ot_total;
    """
    log_query = """
        INSERT INTO cfb.all_games_log(unique_id, date, all_quarters_scrape)
        VALUES %s
        ON CONFLICT (unique_id, date) 
        DO UPDATE SET
            all_quarters_scrape = EXCLUDED.all_quarters_scrape
    """
    insert_data(get_table_function=get_table_function, params=params, query=query, log_query=log_query, log_cols=["all_quarters_scrape"])


def backfill_data_for_table() -> None:
    """
    Generates quarters data for dates in CFB table.
    """
    date_query = """
    SELECT DISTINCT date, unique_id
    FROM cfb.all_games_log
    WHERE all_quarters_scrape IS NOT TRUE
    ORDER BY date ASC
    """
    # Generate ids for which quarters not genned
    ungenned_ids = list(retrieve_data(date_query)["unique_id"])
    # Filter out genned games
    # TODO: figure out why I can't use there WHERE, ANY clause properly, not important now
    home_query = """
    SELECT DISTINCT date, home, unique_id
    FROM cfb.all_games ag
    ORDER BY date ASC
    """
    all_games_df = retrieve_data(home_query)
    all_games_df = all_games_df[all_games_df["unique_id"].isin(ungenned_ids)]

    # Iterates through the games ordered by date, adds quarter data, then removes it from the ungenned_dates
    while ungenned_ids:
        id = ungenned_ids.pop(0)
        game = all_games_df[all_games_df["unique_id"] == id]
        date = game["date"].iloc[0]
        home = game["home"].iloc[0]
        print(f"Backfilling {date} at {home}")
        insert_quarters_data_at_date_home(date, home)

        # Rate limiting to avoid overloading the server
        time.sleep(3.5)

if __name__ == "__main__":
    # python src/cfb/data/scrape_quarters_data.py
    backfill_data_for_table()