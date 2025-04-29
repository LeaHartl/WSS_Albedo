# ! /usr/bin/env python3
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import scipy.stats as stats
from scipy.stats import circmean
from scipy.fft import fft, ifft

import geopandas as gpd

from pytz import timezone
from solartime import SolarTime
import ephem
from pysolar.solar import *

from pvlib import location
from pvlib import irradiance
from pvlib.iotools import read_tmy3
from pvlib.solarposition import get_solarposition
import pvlib

import solar_helpers as sol

# supress copy warning - careful
pd.options.mode.chained_assignment = None  # default='warn'


# get AWS coordinates from shapefile:
#UPDATE CENTRIODS WITH 2024 COORDINDATES - GET MARTIN TO ENTER COORDS IN EXCEL.
AWS = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
AWS = AWS.loc[AWS.name == 'AWS']
AWS.to_crs(epsg=4326, inplace=True)

# load AWS data:
file = '/Users/leahartl/Desktop/WSS/AWS_wss/out/AWS_WSS_proc2025.csv'
data = pd.read_csv(file, parse_dates=True, skiprows=[0,2,3])
data.index = pd.to_datetime(data['TIMESTAMP'])

flags = ['Batt_Min_flag', 'Tair_Avg_flag', 'Hum_Avg_flag', 'Press_Avg_flag', 'Wdir_flag', 'Wspeed_flag',
         'Wspeed_Max_flag', 'SWin_Avg_flag', 'SWout_Avg_flag', 'LWin_Cor_flag', 'LWout_Cor_flag',
         'Snow_flag']


data['albedo'] = data.SWout_Avg/data.SWin_Avg
mask = (data.index.hour >= 10) & (data.index.hour <= 13)

data1 = data[mask]


# day = '2020-11-23'
# day1 = '2020-12-01'

# day = '2021-06-12'
# day1 = '2021-06-22'

datasub = data

# 2020-01-06
# 2020-06-23
# 2020-12-18
#2020-11-24 bis 2020-11-30
#2021-05-31
#2021-06-14


# https://pvlib-python.readthedocs.io/en/stable/gallery/irradiance-decomposition/plot_diffuse_fraction.html#sphx-glr-gallery-irradiance-decomposition-plot-diffuse-fraction-py
# Set location and time zone:
tz = 'UTC'
lat, lon = AWS.geometry.y.values[0], AWS.geometry.x.values[0]

# Create pvlib location object to store lat, lon, timezone
site = location.Location(lat, lon, tz=tz, altitude= 3491.84)

solpos = get_solarposition(
    datasub.index, latitude=lat,
    longitude=lon, altitude=3491.84,
    pressure=datasub.Press_Avg*100,  # convert from millibar to Pa
    temperature=datasub.Tair_Avg)

# Calculate clear-sky GHI and transpose to plane of array
# Define a function so that we can re-use the sequence of operations with
# different locations
def get_irradiance(site_location, tilt, surface_azimuth, datasub, lat, lon):

    # Create 10 min intervals
    times = datasub.index#pd.date_range(date, freq='10min', periods=6*24*9,
                          #tz=site_location.tz)
    # Generate clearsky data using the Ineichen model, which is the default
    # The get_clearsky method returns a dataframe with values for GHI, DNI,
    # and DHI
    clearsky = site_location.get_clearsky(times)
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(
                    times=times,
                    # latitude=lat,
                    # longitude=lon, 
                    pressure=datasub.Press_Avg*100,  # convert from millibar to Pa
                    temperature=datasub.Tair_Avg)
    # Use the get_total_irradiance function to transpose the GHI to POA
    # POA_irradiance = irradiance.get_total_irradiance(
    #     surface_tilt=tilt,
    #     surface_azimuth=surface_azimuth,
    #     dni=clearsky['dni'],
    #     ghi=clearsky['ghi'],
    #     dhi=clearsky['dhi'],
    #     solar_zenith=solar_position['apparent_zenith'],
    #     solar_azimuth=solar_position['azimuth'])

    out_boland = irradiance.boland(datasub.SWin_Avg, solar_position.zenith, datasub.index)

    POA_irradiance = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=out_boland['dni'],
        ghi=datasub['SWin_Avg'],
        dhi=out_boland['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        albedo=0.5)
    # Return DataFrame with only GHI and POA
    return pd.DataFrame({'GHI': clearsky['ghi'],
                         'poa_global': POA_irradiance['poa_global'],
                         'poa_direct': POA_irradiance['poa_direct'],
                         'poa_diffuse': POA_irradiance['poa_diffuse'],
                         'poa_sky_diffuse': POA_irradiance['poa_sky_diffuse'],
                         'poa_ground_diffuse': POA_irradiance['poa_ground_diffuse']})


# Get irradiance data for summer and winter solstice
# summer_irradiance = get_irradiance(site, '06-20-2020', 6, 73)
# winter_irradiance = get_irradiance(site, '12-21-2020', 6, 73)

irradiance = get_irradiance(site, 6, 73, datasub, lat, lon)#73

# Convert Dataframe Indexes to Hour:Minute format to make plotting easier
# summer_irradiance.index = summer_irradiance.index.strftime("%H:%M")
# winter_irradiance.index = winter_irradiance.index.strftime("%H:%M")
print(irradiance)

both = pd.merge(datasub[['SWin_Avg', 'SWout_Avg']], irradiance.tz_localize(None), left_index=True, right_index=True)

print(both)
both.to_csv('out/albedo_cor2025.csv')
