import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import xgboost as xgb
from model import *
from evaluation import *
from graphics import *

def get_top_seasons(n=10, rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
    
    seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    df = prepare_dataframe(seasons)
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season', 'touchdowns']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    df['EPAOE'] = df['epa'] - df['xEPA']
    df = df[['passer_player_id', 'passer_player_name', 'season', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'att', 'touchdowns', 'xCompPer', 'xEPA', 'MoV', 'EPAOE']].groupby(by=['passer_player_id', 'passer_player_name', 'season']).sum().reset_index()
    df.sort_values(by='xEPA', inplace=True, ascending=False)
    top_scores = df.head(n)
    return top_scores

def get_top_scores_by_season(seasons=[2021], n=10, min=10, rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
    if not type(seasons) == list:
        seasons = [seasons]
    df = prepare_dataframe(seasons)
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season', 'touchdowns']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    df['EPAOE'] = df['epa'] - df['xEPA']
    df = df[['passer_player_id', 'passer_player_name', 'season', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'att', 'touchdowns', 'xCompPer', 'xEPA', 'MoV', 'EPAOE']].groupby(by=['passer_player_id', 'passer_player_name', 'season']).sum().reset_index()
    df = df.query('att >= @min')
    df.sort_values(by='xEPA', inplace=True, ascending=False)
    top_scores = df.head(n)
    return top_scores

def get_scores_as_of_week(season=2021, week=17, min=10, rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
    df = prepare_dataframe([season])
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season', 'touchdowns']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    df['EPAOE'] = df['epa'] - df['xEPA']
    df = df.query('week <= @week')
    df = df.query('att >= @min')
    return df

def top_seasons(n=10):
    leaderboard = get_top_seasons(n)
    players = nfl.import_rosters(years=[2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
    leaderboard = leaderboard.merge(right=players[['player_id', 'season', 'headshot_url']], how='left', left_on=['passer_player_id', 'season'], right_on=['player_id', 'season'])
    build_top_seasons_leaderboard(leaderboard)
    return leaderboard

def top_scores_by_season(season=2021, week=17, n=32):
    min = 10 * week
    leaderboard = get_top_scores_by_season(season, n, min)
    players = nfl.import_rosters(years=[season])
    leaderboard = leaderboard.merge(right=players[['player_id', 'season', 'headshot_url']], how='left', left_on=['passer_player_id', 'season'], right_on=['player_id', 'season'])
    build_top_scores_by_season_leaderboard(leaderboard, season, week, min)
    return leaderboard

def scores_as_of_week(season=2021, week=17, min=20):
    scores = get_scores_as_of_week(season=season, week=week, min=min, rebuild_model=False)
    players = nfl.import_rosters(years=[season])
    team_desc = nfl.import_team_desc()
    team_colors = team_desc[['team_abbr', 'team_color', 'team_color2']]
    scores = scores.merge(right=players[['player_id', 'season', 'headshot_url', 'team']], how='left', left_on=['passer_player_id', 'season'], right_on=['player_id', 'season'])
    scores = scores.merge(right=team_colors, how='left', left_on='team', right_on='team_abbr')
    build_scores_as_of_week(scores, season, week, min)
    return scores
    

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    # top_seasons(n=25)
    top_scores_by_season(season=2022, week=1)
    #scores_as_of_week(week=1, season=2022)
    