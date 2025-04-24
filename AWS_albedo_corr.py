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
AWS = AWS.loc[AWS.stakename == 'AWS']
AWS.to_crs(epsg=4326, inplace=True)

# load AWS data:
file = '/Users/leahartl/Desktop/WSS/AWS_wss/out/AWS_WSS_proc2025.csv'
data = pd.read_csv(file, parse_dates=True, skiprows=[0,2,3])
data.index = pd.to_datetime(data['TIMESTAMP'])

flags = ['Batt_Min_flag', 'Tair_Avg_flag', 'Hum_Avg_flag', 'Press_Avg_flag', 'Wdir_flag', 'Wspeed_flag',
         'Wspeed_Max_flag', 'SWin_Avg_flag', 'SWout_Avg_flag', 'LWin_Cor_flag', 'LWout_Cor_flag',
         'Snow_flag']

# set data with flags to np.nan before passing to plotting functions:
# for flag in flags:
#     data.loc[(data[flag] != 0), flag[:-5]] = np.nan

data['albedo'] = data.SWout_Avg/data.SWin_Avg
mask = (data.index.hour >= 10) & (data.index.hour <= 13)
# data.loc[data['Wdir_flag']!=0, 'albedo'] = np.nan
data1 = data[mask]



# day = '2020-11-23'
# day1 = '2020-12-01'

day = '2021-06-12'
day1 = '2021-06-22'
# datasub = data.loc['2020-12-18 00:00' :'2020-12-18 23:50']

datasub = data#data.loc[day+' 00:00' : day1+' 23:50']

# 2020-01-06
# 2020-06-23
# 2020-12-18
#2020-11-24 bis 2020-11-30
#2021-05-31
#2021-06-14


# https://pvlib-python.readthedocs.io/en/stable/gallery/irradiance-decomposition/plot_diffuse_fraction.html#sphx-glr-gallery-irradiance-decomposition-plot-diffuse-fraction-py
# Set location and time zone:
tz = 'UTC'
lat, lon = AWS.geometry.y[0], AWS.geometry.x[0],

# Create pvlib location object to store lat, lon, timezone
site = location.Location(lat, lon, tz=tz, altitude= 3491.84)


solpos = get_solarposition(
    datasub.index, latitude=lat,
    longitude=lon, altitude= 3491.84,
    pressure=datasub.Press_Avg*100,  # convert from millibar to Pa
    temperature=datasub.Tair_Avg)
#solpos.index = greensboro.index  # reset index to end of the hour

# print(solpos)
# out_boland = irradiance.boland(datasub.SWin_Avg, solpos.zenith, datasub.index)
# out_boland = out_boland.rename(
#     columns={'dni': 'dni_boland', 'dhi': 'dhi_boland'})
# print(out_boland)



# fig, ax = plt.subplots(1,1, figsize=(8,6))
# ax.plot(out_boland.index, out_boland.dni_boland, label='dni, normal horz. irr.') 
# ax.plot(out_boland.index, out_boland.dhi_boland, label='dhi, diffuse horz. irr.')
# ax.plot(out_boland.index, out_boland.dhi_boland + out_boland.dni_boland*np.cos(solpos.zenith*np.pi/180), label='..')
# ax.plot(datasub.index, datasub.SWin_Avg, label='AWS')
# ax.legend()
# plt.show()
# stop
# Calculate clear-sky GHI and transpose to plane of array
# Define a function so that we can re-use the sequence of operations with
# different locations
def get_irradiance(site_location, date, tilt, surface_azimuth, datasub, lat, lon):

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

winter_irradiance = get_irradiance(site, day, 6, 73, datasub, lat, lon)#73


# Convert Dataframe Indexes to Hour:Minute format to make plotting easier
# summer_irradiance.index = summer_irradiance.index.strftime("%H:%M")
# winter_irradiance.index = winter_irradiance.index.strftime("%H:%M")
print(winter_irradiance)

# Plot GHI vs. POA for winter and summer

# summer_irradiance['GHI'].plot(ax=ax1, label='GHI')
# # summer_irradiance['POA'].plot(ax=ax1, label='POA')
# winter_irradiance['GHI'].plot(ax=ax2, label='GHI')
# winter_irradiance['POA'].plot(ax=ax2, label='POA')



both = pd.merge(datasub[['SWin_Avg', 'SWout_Avg']], winter_irradiance.tz_localize(None), left_index=True, right_index=True)



# both['difrad1'] = both['GHI']-both['POA']
# both['difrad'] = both['SWin_Avg']-both['POA']

# both['albedo'] = both['SWout_Avg']/both['SWin_Avg']
# both['SWout_Cor'] = both['SWout_Avg']+both['difrad']

# both['albedo_Cor'] = both['SWout_Cor']/both['SWin_Avg']
print(both)
both.to_csv('out/albedo_cor2025.csv')

# stop

# fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
# ax = ax.flatten()
# ax[0].plot(both.index, both.SWin_Avg, label='AWS in', color='k')
# ax[0].plot(both.index, both.SWout_Avg, label='AWS out', color='k', linestyle='--')

# # ax[0].plot(winter_irradiance.index, winter_irradiance['POA']*0.7, label='POA*0.7')

# ax[0].plot(both.index, both['POA'], label='POA', color='red')
# ax[0].plot(both.index, both['GHI'], label='GHI', color='orange')

# ax[0].plot(both.index, both['SWout_Cor'], color='magenta', label='AWS out + diff')

# # ax1.set_xlabel('Time of day (Summer)')
# ax[0].set_xlabel('Time of day (Winter)')
# ax[0].set_ylabel('Irradiance ($W/m^2$)')
# # ax1.legend()
# ax[0].legend()

# ax[1].plot(both.index, both['difrad'])

# ax[2].plot(both.index, both['albedo'], label='albedo, raw')
# ax[2].plot(both.index, both['albedo_Cor'], label='albedo, cor')

# # ax[2].plot(both.index, np.sin(both['albedo']), label='cos')

# ax[2].set_ylim(0,1.8)




# mask = (both.index.hour >= 10) & (both.index.hour <= 12)
# # data.loc[data['Wdir_flag']!=0, 'albedo'] = np.nan
# masked = both[mask]
# print(masked)
# dailyalbedo = masked.resample('d').mean()

# ax[2].scatter(dailyalbedo.index, dailyalbedo['albedo'], label='daily albedo, raw')
# ax[2].scatter(dailyalbedo.index, dailyalbedo['albedo_Cor'], label='daily albedo, cor')
# ax[2].legend()


# # ax[3].plot(both.index, both['albedo'].diff(), label='albedo, shift')
# # ax[3].plot(both.index, fft(both['albedo'].values), label='albedo, shift')
# # # ax[3].set_ylim(-0.2,0.2)
# plt.show()

# # f1, a1 = plt.subplots(1,1)

# # for d in both.index.dayofyear:
# #     # tmp = masked.loc[masked.index.dayofyear==d]
# #     tmp = both.loc[both.index.dayofyear==d]
# #     fd = fft(tmp['albedo'].values)
# #     a1.plot(fd)
# # plt.show()
# # stop


# # # # check how far min and max are shifted against each other
# # checkoffset = data1.resample("d")[['SWin_Avg', 'SWout_Avg']].agg(lambda x: np.nan if x.isna().all() else x.idxmax())
# # diff_ixmx = checkoffset.SWin_Avg - checkoffset.SWout_Avg
# # fig, ax = plt.subplots(2, 1, figsize=(8, 6))
# # ax = ax.flatten()
# # # ax[0].plot(checkoffset.index, diff_ixmx.dt.total_seconds()/3600)
# # ax[1].plot(datasub.index, datasub.SWin_Avg)
# # ax[1].plot(datasub.index, datasub.SWout_Avg)
# # # print(diff_ixmx.min())



# stop

# start='00:00'
# end='23:50'
# # date = '2023-10-11'
# date = '2022-07-06'


# # get subset (pick a day)
# sub = data.loc[(data.index>= pd.to_datetime(date +' '+ start)) & (data.index<= pd.to_datetime(date+' '+end))]


# # subest again for relevant parameters
# st_in = sub[['albedo', 'SWin_Avg', 'SWout_Avg', 'TIMESTAMP']]

# ### olar time from UTC:
# londeg = AWS.geometry.x[0] # longitude of AWS position

# # compute equaton of time
# st_in['ET'] = st_in.index.map(lambda x: sol.eot(x.dayofyear))
# # compute solar time
# st_in['soltime_in'] = st_in.apply(
#         lambda x: pd.to_datetime(x["TIMESTAMP"])
#         + pd.DateOffset(minutes= 4* londeg + x["ET"]),
#         axis=1,
#     )

# # compute solar hour from solar time (=decimal solar hours)
# st_in['solhour'] = pd.to_datetime(st_in['soltime_in']).dt.hour + pd.to_datetime(st_in['soltime_in']).dt.minute/60 + pd.to_datetime(st_in['soltime_in']).dt.second/(60*60)


# # compute solar angles for a horizontal and tilted case, for the given day (start), location (AWS), and solar hour:
# slope = 6
# solnoon, df2 = sol.getSolar(pd.to_datetime(date + ' ' + start), AWS, st_in['solhour'], slope)

# _, df2_9 = sol.getSolar(pd.to_datetime(date + ' ' + start), AWS, st_in['solhour'], 9)
# _, df2_3 = sol.getSolar(pd.to_datetime(date + ' ' + start), AWS, st_in['solhour'], 3)
# _, df2_12 = sol.getSolar(pd.to_datetime(date + ' ' + start), AWS, st_in['solhour'], 12)

# df2 = df2[['solhours', 'inc_tilt', 'inc_flat']]


# df2['ang_dif'] = df2['inc_tilt'] - df2['inc_flat']

# merged = pd.merge(st_in, df2, left_on='solhour', right_on='solhours', how='outer')
# merged.index = pd.to_datetime(merged['TIMESTAMP'])
# merged = merged.sort_index()

# merged_0 = merged.loc[merged.solhour>0]
# print(merged_0)

# # data['albedo'].loc[~mask] = np.nan
# data = data.drop(['TIMESTAMP'], axis=1)

# # plot sunangles for flat and tilted surface on given day of year and slope

# sol.plot_tilt(slope, df2, date)
# sol.plot_measured_shifted(merged_0, date)
# sol.compare_angles(df2, df2_3, df2_9, df2_12, date, slope)

# plt.show()
