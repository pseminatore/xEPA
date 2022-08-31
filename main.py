import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import xgboost as xgb
from model import *
from evaluation import *

def get_top_seasons(n=10, rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
    
    seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    df = prepare_dataframe(seasons)
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    df['EPAOE'] = df['epa'] - df['xEPA']
    df = df[['passer_player_id', 'passer_player_name', 'season', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'att', 'xCompPer', 'xEPA', 'MoV', 'EPAOE']].groupby(by=['passer_player_id', 'passer_player_name', 'season']).sum().reset_index()
    df.sort_values(by='xEPA', inplace=True, ascending=False)
    top_scores = df.head(n)
    return top_scores

def get_top_scores_by_season(seasons=[2021], n=10, rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
    if not type(seasons) == list:
        seasons = [seasons]
    df = prepare_dataframe(seasons)
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    df['EPAOE'] = df['epa'] - df['xEPA']
    df = df[['passer_player_id', 'passer_player_name', 'season', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'att', 'xCompPer', 'xEPA', 'MoV', 'EPAOE']].groupby(by=['passer_player_id', 'passer_player_name', 'season']).sum().reset_index()
    df.sort_values(by='xEPA', inplace=True, ascending=False)
    top_scores = df.head(n)
    return top_scores


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    get_top_scores_by_season(seasons=[2021], n=30, rebuild_model=False, verbose=True)