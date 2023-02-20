import warnings
warnings.filterwarnings('ignore')

import ee

import geopandas as gpd
from calendar import monthrange
import os
from shapely.geometry import Polygon, Point
from math import radians, cos, sin, asin, sqrt, degrees
import geopy
import geopy.distance
import pandas as pd
import datetime
import rasterio as rio
from rasterio.plot import show

global lst90

def urbanBoundaries1(gdf,city,crs):
    b = 5000
    geo = gdf[gdf['UC_NM_MN']==city].reset_index(drop=True)
    bounds = geo.total_bounds
    lon_point_list = [bounds[0]-b, bounds[0]-b, bounds[2]+b, bounds[2]+b]
    lat_point_list = [bounds[1]-b, bounds[3]+b, bounds[3]+b, bounds[1]-b]
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    return geo, gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])

def returnGeometry(gdf):
    bounds = gdf.total_bounds
    return ee.Geometry.Polygon(
            [[[bounds[0], bounds[1]],
            [bounds[0], bounds[3]],
            [bounds[2], bounds[3]],
            [bounds[2], bounds[1]]]])

def millSec(date):
    return ee.Date(date).millis()

def toCelciusDay(image):
    lst = image.select('LST_Day_1km').multiply(0.02).subtract(273.15);
    overwrite = True;
    result = image.addBands(lst, ['LST_Day_1km'], overwrite);
    return result;

def toCelciusNight(image):
    lst = image.select('LST_Night_1km').multiply(0.02).subtract(273.15);
    overwrite = True;
    result = image.addBands(lst, ['LST_Night_1km'], overwrite);
    return result;



def bitwiseExtract(value, fromBit, toBit):
    if not toBit in locals():
        toBit = fromBit;
    maskSize = ee.Number(1).add(toBit).subtract(fromBit);
    mask = ee.Number(1).leftShift(maskSize).subtract(1);
    return value.rightShift(fromBit).bitwiseAnd(mask);

def QC_Day_mask(image2):
    return image2.updateMask(bitwiseExtract(image2.select('QC_Day'), 0, 1));


def QC_Night_mask(image3):
    return image3.updateMask(bitwiseExtract(image3.select('QC_Night'), 0, 1));

def exportImage(img, geometry, folder, dscr):
    task = ee.batch.Export.image.toDrive(image=img,  # an ee.Image object.
                                         region=geometry,  # an ee.Geometry object.
                                         description=dscr,
                                         folder=folder,
                                         fileNamePrefix=dscr,
                                         scale=1000)
    return task

def exportImageMoll(img, geometry, folder, dscr, proj):
    task = ee.batch.Export.image.toDrive(image=img,  # an ee.Image object.
                                         region=geometry,  # an ee.Geometry object.
                                         description=dscr,
                                         folder=folder,
                                         fileNamePrefix=dscr,
                                         scale=1000,
                                         crs=proj)
    return task

### Mollweide


def returnSquare(gdf):
    bounds = gdf.total_bounds
    lon_point_list = [bounds[0], bounds[0], bounds[2], bounds[2]]
    lat_point_list = [bounds[1], bounds[3], bounds[3], bounds[1]]
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    crs = {'init': 'epsg:4326'}
    square = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
    square = square.to_crs(crs)
    #xc = square['geometry'].centroid.x[0]
    #yc = square['geometry'].centroid.y[0]
    #xl = bounds[0]
    #yl = yc
    ## create a buffer of the square of 5 km
    ##
    # = 5 ## kilometers
    distance = sqrt(5*5 + 5*5)  #Pitagora

    ## corner low left
    start = geopy.Point(bounds[1], bounds[0])
    d = geopy.distance.geodesic(kilometers = distance)
    lond1 = d.destination(point=start, bearing=225)[1]
    latd1 = d.destination(point=start, bearing=225)[0]

    ## corner up right
    start = geopy.Point(bounds[3], bounds[2])
    d = geopy.distance.geodesic(kilometers = distance)
    lond2 = d.destination(point=start, bearing=45)[1]
    latd2 = d.destination(point=start, bearing=45)[0]

    ## create buffered square
    lon_point_list = [lond1, lond1, lond2, lond2]
    lat_point_list = [latd1, latd2, latd2, latd1]
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    square1 = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])

    return square, square1

def returnCityBoundary(urban_gdf_cities):
    S = []
    for i in urban_gdf_cities.index:
        original_df = urban_gdf_cities[urban_gdf_cities.index==i]
        exploded = original_df.explode()
        #x = exploded['geometry'][i].centroid.x
        #y = exploded['geometry'][i].centroid.y
        #city = urban_gdf_cities['UC_NM_NN'][i]
        #polygon = exploded.geometry[i][0]
        #p = [x[0], y[0]]
        #lp = list(polygon.exterior.coords)
        s,s1 = returnSquare(exploded)
        S.append(s1)
    return S


######## Cities file ##############################################
def hotdays(image):
    hot = image.gt(lst90);
    return image.addBands(hot.rename('hotdays').set('system:time_start', image.get('system:time_start')));


def listHotDays(dfT, years_str):
    # Define warmest months in the year
    dfT['valid_time'] = pd.to_datetime(dfT['valid_time'])
    df_filtered = dfT[dfT['valid_time'].dt.strftime('%Y').isin(years_str)].reset_index(drop=True)

    ################ get list of days in the three warmest month of the year
    df_filtered = df_filtered.groupby('valid_time')['t2m'].mean().reset_index()
    df_filtered['year'] = df_filtered['valid_time'].dt.year
    df_filtered['month'] = df_filtered['valid_time'].dt.month
    grouped = df_filtered.sort_values('t2m', ascending=False).groupby(['year'])
    result = grouped.head(3).sort_values('year').reset_index()

    dates = []
    delta = datetime.timedelta(days=1)

    dates = []
    for vd in result.valid_time:
        year = vd.year
        month = vd.month
        my_date = datetime.date(year, month, 1)
        while my_date.month == month:
            dates.append((my_date).strftime('%Y-%m-%d'))
            my_date += delta

    return ee.List(dates).map(millSec)