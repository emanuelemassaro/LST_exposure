import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import libpysal
import geopandas as gpd



dfc = gpd.read_file('../data_revision/cities/all/gdfCities.shp')
dfc = dfc.to_crs("EPSG:4326")
idx = 34
dfc = dfc[dfc.index==idx]