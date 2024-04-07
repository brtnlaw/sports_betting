# sports_betting

Personal page to start with sports betting modelling. Tasks that I want to crank out

Simple API call to get best top of book
Historical trends
  features to consider: historical trends by themselves, teams vs the same team, etc.
ELO model 
history agains tthe spread and the book... maybe certain books are motoxic in one direction? (unlikely but a hypo)
need data
then backtest

For this project, let's ensure at least 2020+

Want to store this in some PostgresSql DB



HERES THE IDEA


resample games from player 
blackbox resampling based on various factors (ML MOdel?)
Multivariate KDE to get a PDF

ideally i also have a health target of how much data i have...

Successfully executed 2020-12-26, POR game stat_sheet data
2020-12-26 SAC
insert or update on table "stat_sheet" violates foreign key constraint "stat_sheet_game_id_fkey"
DETAIL:  Key (game_id)=(c88b451e4d003776150a7fe278209933) is not present in table "all_games".