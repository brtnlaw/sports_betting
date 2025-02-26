# Sports Betting Model

Looks for relative value within betting odds. Compares synthetic lines to get the lowest hold. Ideally, non-synthetic line is strong. Current focus on weaker market in CFB, although framework designed to be general.

## Features

- Visualization of lines, ordered by hold
- Recommendation of specific lines to trade on
- Ability to construct synthetic lines
    - Synthetic lines built using a model
- Various derivative predictions
    - Individual player stats, quarter stats

## Setup

- Download PostgreSQL
    - Optional: Download DBeaver
- Run the sql files
- Download the requirements.txt
```
pip install -r requirements.txt
```
- Set up a config.yaml file under the config folder
```
database:
  dbname: 
  user: 
  password: 
  host:
  port:
```
- Set up the anacron job

When editing, make sure to activate the venv. 

If not done yet,
```
python -m venv venv
./venv/Scripts/activate
pip install -e C:\Users\brtnl\OneDrive\Desktop\code\sports_betting
```
