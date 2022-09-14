from cgi import test
from pyexpat.errors import XML_ERROR_INCOMPLETE_PE
import nfl_data_py as nfl
import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

def tune_hyperparams(df):
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'pass']]
    y = df[['epa']]
    data_dmatrix = xgb.DMatrix(data=X, label=y)
    estimator = xgb.XGBRegressor(objective ='reg:squarederror', nthread=4, seed=42)
    parameters = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05],
        'eta': [0.0125, 0.025, 0.05],
        #'subsample': [0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.8, 1],
        #'gamma': [0, 1, 2, 3]
    }
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring = 'r2',
        n_jobs = 10,
        cv = 10,
        verbose=True
    )
    grid_search.fit(X, y)
    print('best estimator: %s' % grid_search.best_estimator_)
    
def prepare_dataframe(seasons):
    df = nfl.import_pbp_data(seasons, cache=True, downcast=False, columns=['passer_player_id', 'passer_player_name', 'season', 'game_id', 'week', 'pass', 'epa', 'qb_epa', 'wpa', 'cp', 'cpoe', 'complete_pass','air_yards', 'xyac_mean_yardage', 'receiver_player_name', 'pass_location', 'posteam_score_post', 'defteam_score_post', 'pass_touchdown'])
    df = df.query('`pass` == 1')
    df['passer_player_name'] = df['passer_player_name'].apply(lambda name: 'A.Rodgers' if name == 'Aa.Rodgers' else name)
    df['passer_player_name'] = df['passer_player_name'].apply(lambda name: 'J.Alen' if name == 'Jo.Allen' else name)
    df = df.dropna(subset=['passer_player_id', 'receiver_player_name', 'pass_location'])
    df[['cp', 'cpoe', 'air_yards', 'xyac_mean_yardage', 'pass_touchdown']] = df[['cp', 'cpoe', 'air_yards', 'xyac_mean_yardage', 'pass_touchdown']].fillna(0)
    df = df.query('air_yards > -15 and air_yards < 75')
    df['xCompletion'] = df.apply(lambda row: 1 if row['complete_pass'] == 1 or row['cp'] > 0.5 else 0, axis=1)
    df['xYards'] = df.apply(lambda row: row['air_yards'] + row['xyac_mean_yardage'] if row['xCompletion'] == 1 else 0, axis=1)
    df = df[['passer_player_id', 'passer_player_name', 'season', 'game_id', 'week', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'pass', 'posteam_score_post', 'defteam_score_post', 'pass_touchdown']].groupby(by=['passer_player_id', 'passer_player_name', 'season', 'game_id', 'week']).agg(epa=('epa', 'sum'), qb_epa=('qb_epa', 'sum'), wpa=('wpa', 'sum'), xCompletion=('xCompletion', 'sum'), xYards=('xYards', 'sum'), att=('pass', 'sum'), posteam_score=('posteam_score_post', 'max'), defteam_score=('defteam_score_post', 'max'), touchdowns=('pass_touchdown', 'sum')).reset_index()
    df['era'] = df['season'].apply(lambda season: 3 if season <= 2013 else (2 if season >= 2014 and season <= 2017 else (1 if season >= 2018 else 0)))
    df['xCompPer'] = df['xCompletion'] / df['att']
    # df['era'] = df['season'].apply(lambda season: 2 if season >= 2014 and season <= 2017 else 0)
    # df['era'] = df['season'].apply(lambda season: 1 if season >= 2018 else 0)
    return df

def build_model(verbose=False, file_path='model.txt'):
    #nfl.cache_pbp([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], downcast=False)
    seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    df = prepare_dataframe(seasons)
    
    
    ## LOSO (Leave One Season Out) training.  Append predictions to resulting df
    df = df[['passer_player_id', 'passer_player_name', 'season', 'game_id', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xCompPer', 'xYards', 'att', 'touchdowns']]
    res_df = pd.DataFrame(columns=['passer_player_id', 'passer_player_name', 'season', 'game_id', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xCompPer', 'xYards', 'att', 'touchdowns', 'xEPA'])
    for season in seasons:
        test_df = df.query('season == @season')
        train_df = df.query('season != @season')
        X_train = train_df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season', 'touchdowns']]
        y_train = train_df['epa']
        X_test = test_df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'att', 'season', 'touchdowns']]
        y_test = test_df['epa']
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                gamma=0, gpu_id=-1, importance_type=None,
                interaction_constraints='', learning_rate=0.05, max_delta_step=0,
                max_depth=3, min_child_weight=1,
                monotone_constraints='()', n_estimators=140, n_jobs=4, nthread=4,
                num_parallel_tree=1, predictor='auto', random_state=42,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,
                subsample=1, tree_method='exact', validate_parameters=1,
                verbosity=None)
        xg_reg.fit(X_train,y_train)
        preds = xg_reg.predict(X_test)
        test_df['xEPA'] = preds.tolist()
        res_df = res_df.append(test_df, ignore_index=True)
    
    

    
    r2 = r2_score(res_df['epa'], res_df['xEPA'])
    rmse = np.sqrt(mean_squared_error(res_df['epa'], res_df['xEPA']))
    if verbose:
        print("R squared: %f" % (r2))
        print("RMSE: %f" % rmse)
        plt.scatter(x=res_df['epa'], y=res_df['xEPA'])
        plt.ylabel('Predicted EPA')
        plt.xlabel('Actual EPA')
        plt.title(f"RMSE: {rmse} | R Squared: {r2}")
        plt.show()
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
    # params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
    #             'max_depth': 5, 'alpha': 10}

    # cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
    #                 num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    # rmse = (cv_results["test-rmse-mean"]).tail(1).values.tolist()[0]
    xg_reg.save_model(file_path)
    return xg_reg
    
    
    ### TODO -- Sum these by season and compare to next
    #df = df.groupby(by=['passer_player_id', 'passer_player_name', 'season']).agg(wpa=('wpa', 'sum'), xCompPer=('xCompPer', np.mean), xCompletion=('xCompletion', 'sum'), xYards=('xYards', 'sum'), passes=('pass', 'sum'), qb_epa=)
    # prev_act_y = df.query('season == 2020')['qb_epa']
    # next_act_y = df.query('season == 2021')['qb_epa']
    # prev_act_x = df.query('season == 2020')[['passer_player_id', 'passer_player_name', 'wpa', 'xCompPer', 'xCompletion', 'xYards', 'pass', 'qb_epa']]
    # next_act_x = df.query('season == 2021')[['passer_player_id', 'passer_player_name', 'wpa', 'xCompPer', 'xCompletion', 'xYards', 'pass']]
    # next_pred_y = xg_reg.predict(prev_act_x[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'pass']])
    # prev_act_x['pred_qb_epa'] = next_pred_y
    # next_act_x['act_qb_epa'] = next_act_y
    # res_df = next_act_x.merge(prev_act_x[['passer_player_id', 'qb_epa', 'pred_qb_epa']], how='left', left_on='passer_player_id', right_on='passer_player_id')
    
    
    
    
    



if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    build_model(verbose=True)