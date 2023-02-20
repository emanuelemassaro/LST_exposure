### Libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
import libpysal
from spreg import GM_Lag, OLS
import pickle


## some functions
def w_adj(X):
    w_ = libpysal.weights.Queen.from_dataframe(X)
    w_.transform = 'r'
    return w_

def XY2(gdf_, Yvar, XVars):
    x_ = gdf_[XVars].values
    y_ = gdf_[Yvar].values.reshape(-1, 1)
    return x_, y_

def run_model_validation(x_, w, model):
    val = sm.add_constant(np.hstack((x_, np.array(w).reshape(-1, 1))))
    if len(model.betas)==4:
        val = np.delete(val, -1, axis=1)
    return np.sum(val * model.betas.T, axis=1).reshape((-1, 1))

def norm01(df_min_max_scaled, column):
    return (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())


### Load city shapefile
dfc = gpd.read_file('../data_revision/cities/all/gdfCities.shp')

### Split cities for k-fold cross validation
df_cities = dfc[['UC_NM_MN']]
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
cities_ = []
for train_index, test_index in kf.split(df_cities.index):
    tmp = df_cities[df_cities.index.isin(test_index)]
    cities_.append(tmp['UC_NM_MN'].values)

## save cities
with open('../data_revision/cities.pkl', 'wb') as f:
    pickle.dump(cities_, f)

### Create Unique DataFrame that contains all the data
gdf = gpd.GeoDataFrame()
for idx in range(0, len(dfc)):
    fout = '../data_revision/dataframes/gdf_%d_all.shp'%idx
    df = gpd.read_file(fout)
    df['UC_NM_MN'] = dfc['UC_NM_MN'][idx]
    gdf = gdf.append(df, ignore_index=True)

## Clean the data and define right variables
gdf = gdf[gdf['dist']>0].reset_index(drop=True)
gdf['dist_n'] = 1/(gdf['dist'])
gdf['dist_n'] = norm01(gdf, 'dist_n')
#gdf['dist_n'] = np.sqrt(gdf['dist'])
#gdf = gdf[gdf['NDBI']<=1].reset_index(drop=True)
#gdf = gdf[gdf['NDBI']>=-1].dropna().reset_index(drop=True)

target_vars = ['hot_days', 'hot_nights']
predictor_vars = ['NDVI', 'NDBI', 'dist_n']
cols = ['Phase', 'R2_train_slm', 'R2_val_slm', 'R2_train_ols', 'R2_val_ols', 'MAE_val_slm', 'MAE_val_ols', 'SLM', 'OLS']


for target_var in target_vars:
    ## Run Training validation
    df_tv = pd.DataFrame(columns=cols)
    for phase, c in enumerate(cities_):
        ## remove the test set
        gdfF = df_cities[~df_cities['UC_NM_MN'].isin(c)]
        count = 0
        for train_index, val_index in kf.split(gdf.index):
            ## Split in training and validation set
            cities_train = gdfF[gdfF.index.isin(train_index)]
            cities_val = gdfF[gdfF.index.isin(val_index)]

            ## 1. training phase
            gdf_train = gdf[gdf['UC_NM_MN'].isin(cities_train['UC_NM_MN'])]
            x_train, y_train = XY2(gdf_train, target_var, predictor_vars)
            w_train = w_adj(gdf_train)
            ols = OLS(y_train, x_train, w=w_train, name_y=target_var, name_x=predictor_vars, spat_diag=True)
            slm = GM_Lag(y_train, x_train, w=w_train, name_y=target_var, name_x=predictor_vars, spat_diag=True)
            print('training done', phase, target_var, len(df_tv))

            ## 2. validation phase
            gdf_val = gdf[gdf['UC_NM_MN'].isin(cities_val['UC_NM_MN'])]
            x_val, y_val = XY2(gdf_val, target_var, predictor_vars)
            w_val = libpysal.weights.lag_spatial(w_adj(gdf_val), y_val)
            print('validation done', phase, target_var, len(df_tv))

            ## 2.1 OLS
            yOls = run_model_validation(x_val, w_val, ols)
            r2_ols = r2_score(yOls, y_val)
            mae_ols = mean_absolute_error(yOls, gdf_val[target_var])

            ## 2.2 SLM
            ySLM = run_model_validation(x_val, w_val, slm)
            r2_slm = r2_score(ySLM, y_val)
            mae_slm = mean_absolute_error(ySLM, gdf_val[target_var])


            ## 3. Save results
            data = [phase, slm.pr2, r2_slm, ols.r2, r2_ols, mae_slm, mae_ols, slm, ols]
            tmp = pd.DataFrame(columns=cols, data=[data])
            count += 1
            tmp.to_pickle("../data_revision/coefficients/df_tv_%s_%d_%d.pkl"%(target_var, phase, count))


            #df_tv = df_tv.append(pd.DataFrame(columns=cols, data=[data]), ignore_index=True)
            #print('dataframe updated', phase, target_var, len(df_tv))

    #df_tv.to_pickle("../data_revision/coefficients/df_tv1_%s.pkl"%target_var)