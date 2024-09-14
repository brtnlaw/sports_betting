import requests
from requests.auth import HTTPBasicAuth
import urllib
from datetime import datetime
from dataclasses import dataclass
import numpy as np

BASE_URL = 'https://api.prop-odds.com'
API_KEY = 'S868xijsuW1dkuimxuR4sGKxsDt5aXTGe5EBnw8gZY'

def get_request(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    
    print('Request failed with status:', response.status_code)
    return {}

def get_nfl_games():
    now = datetime.now()
    query_params = {
        'date': now.strftime('%Y-%m-%d'),
        'tz': 'America/Chicago',
        'api_key': API_KEY,
    }
    params = urllib.parse.urlencode(query_params)
    url = BASE_URL + '/beta/games/nfl?' + params
    return get_request(url)

def get_nba_games():
    now = datetime.now()
    query_params = {
        'date': now.strftime('%Y-%m-%d'),
        'tz': 'America/Chicago',
        'api_key': API_KEY,
    }
    params = urllib.parse.urlencode(query_params)
    url = BASE_URL + '/beta/games/nba?' + params
    return get_request(url)


def get_game_info(game_id):
    query_params = {
        'api_key': API_KEY,
    }
    params = urllib.parse.urlencode(query_params)
    url = BASE_URL + '/beta/game/' + game_id + '?' + params
    return get_request(url)


def get_markets(game_id):
    query_params = {
        'api_key': API_KEY,
    }
    params = urllib.parse.urlencode(query_params)
    url = BASE_URL + '/beta/markets/' + game_id + '?' + params
    return get_request(url)


def get_most_recent_odds(game_id, market):
    query_params = {
        'api_key': API_KEY,
    }
    params = urllib.parse.urlencode(query_params)
    url = BASE_URL + '/beta/odds/' + game_id + '/' + market + '?' + params
    return get_request(url)

# I think I'm out of requests....
def get_games_at_date(date):
    query_params = {
        'date': date.strftime('%Y-%m-%d'),
        'tz': 'America/Chicago',
        'api_key': API_KEY,
    }
    params = urllib.parse.urlencode(query_params)
    url = BASE_URL + '/beta/games/nba?' + params
    return get_request(url)