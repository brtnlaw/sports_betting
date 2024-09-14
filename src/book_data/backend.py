import api_utils
from dash import callback
import datetime as dt
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PlayerProp:
    name: str
    stat: str
    over: float = 0.0
    under: float = 0.0

    @property
    def over_implied(self) -> float:
        return 1.0/self.over
    
    @property
    def under_implied(self) -> float:
        return 1.0/self.under

    @property
    def hold(self) -> float:
        return (self.over_implied + self.under_implied - 1)
    
def amer_to_decimal(amer_odds: int) -> float:
    if amer_odds > 0:
        return np.round((float(amer_odds) / 100) + 1, 2)
    else:
        return np.round((100 / np.abs(float(amer_odds))) + 1, 2)
    
def build_player_prop_dict(outcomes: list, stat: str = 'block'):
    player_prop_dict = {}
    for outcome in outcomes:
        name = outcome['participant_name']
        handicap = outcome['handicap']
        amer_odds = outcome['odds']
        dec_odds = amer_to_decimal(amer_odds)

        # make an entry if it doesn't exist already
        if name not in player_prop_dict:
            player_prop_dict[name] = {}
        if handicap not in player_prop_dict[name]:
            player_prop_dict[name][handicap] = PlayerProp(name=name, stat=stat)

        # if the odds improve, then we replace
        if outcome['name'].startswith('Over') or outcome['name'].endswith('Over'):
            player_prop_dict[name][handicap].over = max(player_prop_dict[name][handicap].over, dec_odds)
        else:
            player_prop_dict[name][handicap].under = max(player_prop_dict[name][handicap].under, dec_odds)
    return player_prop_dict
    
def prop_df_from_dict(outcomes: list, stat: str = 'block'):
    prop_dict = build_player_prop_dict(outcomes, stat)

    data = []
    for player, handicaps in prop_dict.items():
        for handicap, props in handicaps.items():
            data.append({
                'player_name': player,
                'handicap': handicap,
                'over': props.over,
                'under': props.under,
                'hold': props.hold
            })

    # Convert to DataFrame
    prop_df = pd.DataFrame(data)
    prop_df.set_index(['player_name', 'handicap'])
    return prop_df

def build_prop_df(date, stat: str = 'blocks'):
    games = api_utils.get_games_at_date(date)['games']
    prop_df_list = []
    for game in games:
        game_id = game['game_id']
        odds = api_utils.get_most_recent_odds(game_id, 'player_blocks_over_under')
        for book in odds['sportsbooks']:
            prop_df_list.append(prop_df_from_dict(book['market']['outcomes'], stat))
    prop_df = pd.concat(prop_df_list)
    return prop_df



