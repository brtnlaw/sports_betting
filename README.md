# Sports Betting Model

Looks for relative value within betting odds within CFB. Simply a fun passion project whose primary goal is not to promote gambling or make loads of money, but rather to test my ability to predict random events and the sharpest lines are formed with people's wallets.

## (Desired) Features

- Visualization of lines, ordered by hold
- Recommendation of specific lines to trade on
- Ability to construct synthetic lines
    - Synthetic lines built using a model
- Model prediction break down and predicted edge from theo
- Various derivative predictions
    - Individual player stats, quarter stats
 
## Project Structure
The ethos behind this model is to utilize a more robust machine learning architecture that is enriched with fruitful and creative feature development. Namely, we have various pre-processing pipelines that transform various sources of public data, feed this into LightGBM with more intelligent hyperparameter sweeping and feature selection, and lastly output a prediction for the respective market.

## Setup

- Download PostgreSQL
    - Optional: Download DBeaver
- Run the sql files via each .py in src/cfb/data
- Download the requirements.txt 
    - May need to add pg_config to path, PATH="/Library/PostgreSQL/17/bin/:$PATH"
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
Windows
```
python -m venv venv
./venv/Scripts/activate
pip install -e .
```
Mac
```
python3 -m venv venv
source venv/bin/activate
```
- Set up a .env file
```
CFBD_API_KEY = ******
PROJECT_ROOT = ******
```
- backtest.py contains example usages depending on hyperparameter choice (betting function, etc.)
- saved models can be played with using tools in evaluation.py


