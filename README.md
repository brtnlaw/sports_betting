# Sports Betting Model

Looks for relative value within betting odds within CFB.

## (Desired) Features

- Visualization of lines, ordered by hold
- Recommendation of specific lines to trade on
- Ability to construct synthetic lines
    - Synthetic lines built using a model
- Model prediction break down and predicted edge from theo
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
- When editing, make sure to activate the venv. If not done yet, run below, else only run the second line.
```
python -m venv venv
./venv/Scripts/activate
pip install -e C:\Users\brtnl\OneDrive\Desktop\code\sports_betting
```
- Set up a .env file
```
CFBD_API_KEY = ******
PROJECT_ROOT = ******
```

