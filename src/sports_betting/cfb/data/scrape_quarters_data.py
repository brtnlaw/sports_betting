import re
from bs4 import BeautifulSoup
import datetime as dt
import pandas as pd
from io import StringIO
from urllib.request import urlopen
from db_utils import retrieve_data, get_rank_from_row, generate_unique_game_id, insert_data
from typing import Optional


def get_quarters_data_at_date_home(date: dt.date, home: str) -> pd.DataFrame:
    """
    Gets quarters data for a game given by its date and home team

    Args:
        date (str): Date of game.
        home (str): Home team.

    Returns:
        pd.DataFrame: Quarter score data. 
    """
    home_str = home.replace(" ", "-").lower()
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

    # Generate unique_id to match to
    unique_id = generate_unique_game_id(pd.Series({'Date': date, 'Visitor': visitor, 'Home': home}))

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
        if df.columns[-2] != "4":
            num_ot = df.columns[-2]
            # If there are 3 OT's, want to sum from OT1 to OT3
            ot_col = f"{team[i]}_ot_total"
            ot_total = sum(df.loc[i, "OT1":num_ot])
            col_dict[ot_col] = ot_total
            pt_total += ot_total
            col_dict["OT"] = num_ot
        assert pt_total == int(df.iloc[i,-1:].iloc[0]), "Quarter points don't add up to final"

    col_dict["date"] = date
    col_dict["home"] = home
    col_dict["visitor"] = visitor
    col_dict["unique_id"] = unique_id

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
        INSERT INTO cfb.all_games_log(unique_id, date)
        VALUES %s
        ON CONFLICT (unique_id, date) 
        DO UPDATE SET
            all_quarters_scrape = EXCLUDED.all_quarters_scrape
    """
    insert_data(get_table_function=get_table_function, params=params, query=query, log_query=log_query)


def backfill_data_for_table() -> None:
    """
    Generates quarters data for dates in CFB table.
    """
    # Given that games are done a full day at a time, should not be possible to have partial slates logged
    date_query = """
    SELECT DISTINCT date, home
    FROM cfb.all_games_log
    WHERE all_quarters_scrape IS NOT TRUE
    ORDER BY date ASC
    """
    ungenned_games = retrieve_data(date_query)
    # Iterates through the games ordered by date, adds quarter data, then removes it from the ungenned_dates
    while ungenned_games:
        game = ungenned_games.iloc[0]
        date = game["date"]
        home = game["home"]
        print(f"Backfilling {date} at {home}")
        get_quarters_data_at_date_home(date, home)
        # Drop the row with game after everything done
        ungenned_games = ungenned_games[1:]
    pass

# TODO: redo ALL of the unique_ids...
if __name__ == "__main__":
    # python src/sports_betting/cfb/data/scrape_quarters_data.py
    insert_quarters_data_at_date_home(dt.date(2024, 9, 20), "Washington State")