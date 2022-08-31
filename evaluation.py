import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import xgboost as xgb
from model import *

def stats_by_year(rebuild_model=False, verbose=False):
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
    df = df[['season', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'att', 'xCompPer', 'xEPA', 'MoV']].groupby(by=['season']).agg(np.mean).reset_index()
    return df
    

def mov_correlation(rebuild_model=False, verbose=False):
    if not exists('model.txt') or rebuild_model:
        model = build_model(verbose=verbose)
    else:
        model = xgb.XGBRegressor()
        model.load_model("model.txt")
       
        
    seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    df = prepare_dataframe(seasons)
    df['MoV'] = df['posteam_score'] - df['defteam_score']   
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'era']]
    xEPA_df = model.predict(X) 
    df['xEPA'] = xEPA_df.tolist()   
    df = df[['passer_player_id', 'passer_player_name', 'season', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'att', 'xCompPer', 'xEPA', 'MoV']].groupby(by=['passer_player_id', 'passer_player_name', 'season']).sum().reset_index()
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
    mov_correlation(rebuild_model=False, verbose=True)