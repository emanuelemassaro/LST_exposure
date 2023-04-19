########## file that contains all the functions used in the notebooks
import rasterio as rio
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point

import pickle
#####################################################################
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
#import libpysal
#from spreg import GM_Lag, OLS, MoranRes
import seaborn as sns
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from matplotlib.gridspec import GridSpec
from matplotlib import colors

from rasterio.plot import plotting_extent
import earthpy.plot as ep
import rioxarray as rxr


def createDataFrame(idx):
    lst = '../data_revision/LST_hot_days/total_%d_1000_new.tif'%idx
    with rio.open(lst) as dataset:
        val = dataset.read(1) # band 5
        no_data=dataset.nodata
        geometry = [Point(dataset.xy(x,y)[0],dataset.xy(x,y)[1]) for x,y in np.ndindex(val.shape)]
        df = gpd.GeoDataFrame({'geometry':geometry})
        df.crs = dataset.crs
        val = dataset.read(1) # band 1
        no_data=dataset.nodata
        v1 = [val[x,y]/10 for x,y in np.ndindex(val.shape)]
        
        val = dataset.read(2) # band 1
        no_data=dataset.nodata
        v2 = [val[x,y]/10 for x,y in np.ndindex(val.shape)]
        
        
        df['hot_days']=v1
        df['hot_nights']=v2
        return df

def addPop(fout):
    with rio.open(fout) as dataset:
        val = dataset.read(1) # band 5
        v = [val[x,y] for x,y in np.ndindex(val.shape)]
        return v

def returnAverage(path, idx):
    years = np.arange(2010,2021,1)
    count = 0
    for year in years:
        im = path%(idx,year)
        if not os.path.exists(im):continue
        #print(im)
        with rio.open(im) as src:
            array0 = src.read()
        #array0[np.isnan(array0)] = 0
        if count == 0:
            val = array0
            count+=1
        else:
            val = np.nanmean((val, array0), axis=0)
    val = val[0]
    v = [val[x,y] for x,y in np.ndindex(val.shape)]
    
    return v

def distToDF(path):
    with rio.open(path) as src:
        val = -src.read()
    val[np.where( val < 0 )] = 0
    val = val[0]
    v = [val[x,y] for x,y in np.ndindex(val.shape)]
    return v

def norm01(df_min_max_scaled, column):
    return (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    


def valSet(gdfF, cc, XVars, Yvars):
    gdf_val = gdfF[gdfF['UC_NM_MN'].isin(cc)].reset_index(drop=True)
    gdf_val = gdf_val[gdf_val['dist']>=0].reset_index(drop=True)
    y_day_val = gdf_val[Yvars[0]].values.reshape(-1, 1)
    y_night_val= gdf_val[Yvars[1]].values.reshape(-1, 1)
    x_val = gdf_val[XVars].values
    #w_val = libpysal.weights.KNN.from_dataframe(gdf_val, k=4)
    w_val = libpysal.weights.Queen.from_dataframe(gdf_val)
    w_val.transform = 'r'
    w_day_val = libpysal.weights.lag_spatial(w_val, y_day_val)
    w_night_val = libpysal.weights.lag_spatial(w_val, y_day_val)
    
    return w_day_val, w_night_val, x_val, y_day_val, y_night_val


def trainSet(gdfF, cc, XVars, Yvars):
    gdf_train = gdfF[~gdfF['UC_NM_MN'].isin(cc)].reset_index(drop=True)
    #gdf_train = gdf_train[gdf_train['dist']>=0].reset_index(drop=True)
    #w_train = libpysal.weights.KNN.from_dataframe(gdf_train, k=4)
    w_train = libpysal.weights.Queen.from_dataframe(gdf_train)
    w_train.transform = 'r'
    x_train = gdf_train[XVars].values
    y_day_train = gdf_train[Yvars[0]].values.reshape(-1, 1)
    y_night_train = gdf_train[Yvars[1]].values.reshape(-1, 1)
    return w_train, x_train, y_day_train, y_night_train



def run_model_validation_s(x_, w, model):
    val = sm.add_constant(np.hstack((x_, np.array(w).reshape(-1, 1))))
    return np.sum(val * model.betas.T, axis=1).reshape((-1, 1))

def run_model_validation_o(x_, model):
    val = sm.add_constant(x_)
    return np.sum(val* model.betas.T, axis=1).reshape((-1, 1))
    
    
    
#################### plot #################################################
def billions(x, pos):
    'The two args are the value and tick position'
    return '%d' % (x * 1e-9)

def millions(x, pos):
    'The two args are the value and tick position'
    return '%d' % (x * 1e-6)

def setFont(ax, font, size):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font)
        label.set_fontsize(size)
    return ax    
    
    

