from pyexpat.errors import XML_ERROR_INCOMPLETE_PE
import nfl_data_py as nfl
import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


## TODO - Add era to model
def build_model(verbose=False, file_path='model.txt'):
    #nfl.cache_pbp([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], downcast=False)
    df = nfl.import_pbp_data([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], cache=True, downcast=False, columns=['passer_player_id', 'passer_player_name', 'season', 'game_id', 'pass', 'epa', 'qb_epa', 'wpa', 'cp', 'cpoe', 'complete_pass','air_yards', 'xyac_mean_yardage', 'receiver_player_name', 'pass_location'])
    df = df.query('`pass` == 1')
    df = df.dropna(subset=['passer_player_id', 'receiver_player_name', 'pass_location'])
    df[['cp', 'cpoe', 'air_yards', 'xyac_mean_yardage']] = df[['cp', 'cpoe', 'air_yards', 'xyac_mean_yardage']].fillna(0)
    df = df.query('air_yards > -15 and air_yards < 75')
    df['xCompletion'] = df.apply(lambda row: 1 if row['complete_pass'] == 1 or row['cp'] > 0.5 else 0, axis=1)
    df['xYards'] = df.apply(lambda row: row['air_yards'] + row['xyac_mean_yardage'] if row['xCompletion'] == 1 else 0, axis=1)
    df = df[['passer_player_id', 'passer_player_name', 'season', 'game_id', 'epa', 'qb_epa', 'wpa', 'xCompletion', 'xYards', 'pass']].groupby(by=['passer_player_id', 'passer_player_name', 'season', 'game_id']).sum().reset_index()
    
    y = df['qb_epa']
    df['xCompPer'] = df['xCompletion'] / df['pass']
    X = df[['wpa', 'xCompPer', 'xCompletion', 'xYards', 'pass']]
    data_dmatrix = xgb.DMatrix(data=X, label=y)
    

    # estimator = xgb.XGBRegressor(objective ='reg:squarederror', nthread=4, seed=42)
    # parameters = {
    #     'max_depth': range (2, 10, 1),
    #     'n_estimators': range(60, 220, 40),
    #     'learning_rate': [0.1, 0.01, 0.05]
    # }
    # grid_search = GridSearchCV(
    #     estimator=estimator,
    #     param_grid=parameters,
    #     scoring = 'r2',
    #     n_jobs = 10,
    #     cv = 10,
    #     verbose=True
    # )
    # grid_search.fit(X, y)
    # print('best estimator: %s' % grid_search.best_estimator_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
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
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    if verbose:
        print("R squared: %f" % (r2))
        print("RMSE: %f" % rmse)
        plt.scatter(x=y_test, y=preds)
        plt.ylabel('Predicted EPA')
        plt.xlabel('Actual EPA')
        plt.title(f"RMSE: {rmse} | R Squared: {r2}")
        plt.show()
    #xgb.plot_importance(xg_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
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