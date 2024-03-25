import hashlib
import pandas as pd

TEAM_CODE_DICT = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

def generate_unique_player_id(row: pd.Series) -> str:
    '''
    Takes player and team and generates a unique hash.

    Args:
        row (pd.Series): Row of player data. Necessarily has a Player and Team column.

    Returns:
        str: Unique hash.
    '''
    assert('Player' in row.index and 'Team' in row.index), 'Row missing at least one of "Player", "Team"'
    combined_values = f'{row["Player"]}{row["Team"]}' 
    hash_value = hashlib.md5(combined_values.encode()).hexdigest()
    return hash_value


def generate_unique_game_id(row: pd.Series) -> str:
    '''
    Takes date, visitor team, and home team and generates a unique hash.

    Args:
        row (pd.Series): Row of player data. Necessarily has a Date, Visitor, and Home column.

    Returns:
        str: Unique hash.
    '''
    assert('Date' in row.index and 'Visitor' in row.index and 'Home' in row.index), 'Row missing at least one of "Date", "Visitor", "Home"'
    combined_values = f'{row["Date"]}{row["Visitor"]}{row["Home"]}' 
    hash_value = hashlib.md5(combined_values.encode()).hexdigest()
    return hash_value
