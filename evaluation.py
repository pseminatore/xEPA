import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import xgboost as xgb
from model import *




def mov_correlation(rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
    
        
    df = nfl.import_pbp_data([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], cache=True, downcast=False, columns=['passer_player_id', 'passer_player_name', 'season', 'game_id', 'pass', 'epa', 'qb_epa', 'wpa', 'cp', 'cpoe', 'complete_pass','air_yards', 'xyac_mean_yardage', 'receiver_player_name', 'pass_location', 'posteam_score_post', 'defteam_score_post'])
    df = df.query('`pass` == 1')
    df = df.dropna(subset=['passer_player_id', 'receiver_player_name', 'pass_location'])
    df[['cp', 'cpoe', 'air_yards', 'xyac_mean_yardage']] = df[['cp', 'cpoe', 'air_yards', 'xyac_mean_yardage']].fillna(0)
    df = df.query('air_yards > -15 and air_yards < 75')
    df['xCompletion'] = df.apply(lambda row: 1 if row['complete_pass'] == 1 or row['cp'] > 0.5 else 0, axis=1)
    df['xYards'] = df.apply(lambda row: row['air_yards'] + row['xyac_mean_yardage'] if row['xCompletion'] == 1 else 0, axis=1)
    df = df[['passer_player_id', 'passer_player_name', 'season', 'game_id', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'pass', 'posteam_score_post', 'defteam_score_post']].groupby(by=['passer_player_id', 'passer_player_name', 'season', 'game_id']).agg(epa=('epa', 'sum'), qb_epa=('qb_epa', 'sum'), wpa=('wpa', 'sum'), xCompletion=('xCompletion', 'sum'), xYards=('xYards', 'sum'), att=('pass', 'sum'), posteam_score=('posteam_score_post', 'max'), defteam_score=('defteam_score_post', 'max')).reset_index()
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    df['xCompPer'] = df['xCompletion'] / df['att']
    df['era1'] = df['season'].apply(lambda season: 1 if season <= 2013 else 0)
    df['era2'] = df['season'].apply(lambda season: 1 if season >= 2014 and season <= 2017 else 0)
    df['era3'] = df['season'].apply(lambda season: 1 if season >= 2018 else 0)
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'pass', 'era1', 'era2', 'era3']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    r2 = r2_score(df['MoV'], df['xEPA'])
    rmse = np.sqrt(mean_squared_error(df['MoV'], df['xEPA']))
    if verbose:
        print("R squared: %f" % (r2))
        print("RMSE: %f" % rmse)
        plt.scatter(x=df['MoV'], y=df['xEPA'])
        plt.ylabel('xEPA')
        plt.xlabel('Margin of Victory')
        plt.title(f"RMSE: {rmse} | R Squared: {r2}")
        plt.show()
    return 0    


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    mov_correlation(rebuild_model=True, verbose=True)