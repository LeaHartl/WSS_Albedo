# %%
import geemap
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
# import seaborn as sns
import geopandas as gpd


# load shapefile with positions
stakes = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
stakes_r = stakes.to_crs(4326)

print(stakes_r)


stakes_r['x'] = stakes_r.geometry.x
stakes_r['y'] = stakes_r.geometry.y

Map = geemap.Map()

# function to add albedo band:
def addAlbedo_Liang(image):
    albedo2 = image.expression(
         '0.356 * Blue+0.130* Red+0.373*NIR+0.085*SWIR1+0.072*SWIR2-0.0018',
        {
            'Blue': image.select('Blue'),
            'Green': image.select('Green'),
            'Red': image.select('Red'),
            'NIR': image.select('NIR'),
            'SWIR1': image.select('SWIR1'),
            'SWIR2': image.select('SWIR2')
        }
    ).rename('liangAlbedo')
    return image.addBands(albedo2).copyProperties(image, ['system:time_start'])


# Function to get and rename bands of interest from Sentinel 2.
def renameS2(img):
  return img.select(
    ['B2',   'B3',    'B4',  'B8',  'B11', 'B12' ,'QA60', 'SCL'],
    ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'QA60', 'SCL']
  )


# function to mask bad pixels:
def maskS2sr(image):
    # 1 is saturated or defective pixel
    not_saturated = image.select('SCL').neq(1)
    return image.updateMask(image.select(QA_BAND).gte(CLEAR_THRESHOLD)).updateMask(not_saturated).divide(10000).copyProperties(image, ['system:time_start'])


# function to reduce area:
def reducePoint(image):
    reduced = image.reduceRegion(
            reducer=ee.Reducer.mean().unweighted(),
            geometry=aoi,
            scale=10,  # meters
            crs='EPSG:4326'
            )
    return ee.Feature(None, reduced).copyProperties(image, ['system:time_start'])


# %%
date_start = '2017-05-01'
date_end = '2024-12-31'

csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
# Use 'cs' or 'cs_cdf', depending on your use case; see docs for guidance.
QA_BAND = 'cs'
# The threshold for masking; values between 0.50 and 0.65 generally work well.
# Higher values will remove thin clouds, haze & cirrus shadows.
CLEAR_THRESHOLD = 0.5
# buffer around the point in meters
BUFFER = 10


for xy in stakes_r.geometry:
    st = stakes_r.loc[stakes_r.geometry==xy]['name'].values[0]
    print(st)
    # if st == 'F':
        # pt = ee.Geometry.Point(xy.x, xy.y)
        # print(xy.x, xy.y)
        # print(xy.x.round(decimals=6), xy.y.round(decimals=6))
    pt = ee.Geometry.Point(xy.x.round(decimals=6), xy.y.round(decimals=6))
    aoi = pt.buffer(distance=BUFFER)

    s2colFilter = ee.Filter.And(
        ee.Filter.bounds(aoi), # filterbounds not available on python api https://github.com/google/earthengine-api/issues/83
        ee.Filter.date(date_start, date_end),
    )

    s2Col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").linkCollection(csPlus, [QA_BAND])\
                            .map(maskS2sr)\
                            .filter(s2colFilter)\
                            .map(renameS2)\
                            .map(addAlbedo_Liang)\
                            .select(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'liangAlbedo'])\
                            .map(reducePoint)
            
    time = s2Col.aggregate_array('system:time_start').getInfo()
    # convert to dataframe:
    dfpoint = ee.data.computeFeatures({
    'expression': s2Col,
    'fileFormat': 'PANDAS_DATAFRAME'
    })
    # reality check print
    print(dfpoint)  
    dfpoint['name'] = st
    dfpoint['time'] = time
    dfpoint['dttime'] = pd.to_datetime(dfpoint['time'], unit='ms')
    dfpoint.dropna(subset =['liangAlbedo'] , inplace=True)
    dfpoint.drop(columns =['geo'] , inplace=True)

    dfpoint.to_csv('/Users/leahartl/Desktop/WSS/WSS_Albedo/output_GEE/extractpoint_'+st+'_allyear_'+str(BUFFER)+'.csv')

