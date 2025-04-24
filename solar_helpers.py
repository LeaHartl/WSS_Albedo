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
import geopandas as gpd

from pytz import timezone
from solartime import SolarTime

import ephem
from pysolar.solar import *

# calculate solar angle for an inclined surface
# https://www.researchgate.net/publication/339687753_A_Solar_Altitude_Angle_Model_for_Efficient_Solar_Energy_Predictions#pf7
def incidenceangle(decl, latrad, beta, az, hour_ang):
    # incidence angle for inclined surface:
    sangle_incl = np.arcsin(np.sin(decl)*np.sin(latrad)*np.cos(beta) +
                            np.sin(decl)*np.cos(latrad)*np.sin(beta)*np.cos(az) +
                            np.cos(decl)*np.cos(latrad)*np.cos(hour_ang)*np.cos(beta) -
                            np.cos(decl)*np.sin(latrad)*np.cos(hour_ang)*np.sin(beta)*np.cos(az) -
                            np.cos(decl)*np.sin(hour_ang)*np.sin(beta)*np.sin(az)
                            )
    return(sangle_incl)


# claculate solar time, normal
def solartime(sub, AWS, sun=ephem.Sun()):
    o = ephem.Observer()
    lat, lon = AWS.geometry.y.values[0], AWS.geometry.x.values[0] 
    o.lat, o.long = str(AWS.geometry.y.values[0]), str(AWS.geometry.x.values[0]) 
    sun.compute(o)
    lst = []
    for d in sub.index:
        o.date = d
        hour_angle = o.sidereal_time() - sun.ra

        #print(ephem.hours(hour_angle))# + ephem.hours('12:00')).norm)

        if ephem.hours(hour_angle) < 0:
            lst.append(str('00:00:00.0'))
        else:
            lst.append(str(ephem.hours(hour_angle)))

    sub['soltime'] = lst
    return (sub)


# calculate solar time with option to have a tilted surface
def getSolar(some_date, AWS, solhours, slope):
    o = ephem.Observer()
    lat, lon = AWS.geometry.y.values[0], AWS.geometry.x.values[0] 
    latrad = lat *np.pi/180

    o.lat, o.long = str(AWS.geometry.y.values[0]), str(AWS.geometry.x.values[0]) 

    sun = ephem.Sun()

    yr = some_date.year
    mn = some_date.month
    dy = some_date.day

    start = dt.datetime(yr, mn, dy, 23, 0)

    sunrise = o.previous_rising(sun, start=start)
    noon = o.next_transit(sun, start=sunrise)
    sunset = o.next_setting(sun, start=noon)

    d = some_date.dayofyear

    # declination angle:
    decl = -0.40928 * np.cos(2* np.pi * 365 * (d+10))
    # surface azimuth:
    az = 72 * np.pi/180
    # slope:
    beta = slope * np.pi/180

    inc_tilt_angle = []
    inc_flat_angle = []
    # dif_noon = []
    hrs = solhours.values
    for h in hrs:
        # solar hour:
        t = h
        # hour angle:
        hour_ang = np.pi/12 * (t-12)

        inc_tilt = incidenceangle(decl, latrad, beta, az, hour_ang)
        inc_flat = incidenceangle(decl, latrad, 0, az, hour_ang)

        inc_tilt_angle.append(inc_tilt)
        inc_flat_angle.append(inc_flat)


    df2 = pd.DataFrame(columns=['dt', 'solhours'])
    df2['solhours'] = hrs
    df2['dt'] = some_date.strftime(format='%Y%m%d')
    df2['inc_tilt'] = inc_tilt_angle
    df2['inc_flat'] = inc_flat_angle

    df2['date'] = pd.to_datetime(df2['dt'], format='%Y%m%d')
    df2['datetime'] = df2.apply(
        lambda x: x["date"]
        + pd.DateOffset(hours=x["solhours"]),
        axis=1,
    )
    df2.index = df2['datetime']

    return(noon.datetime(), df2)


# Equation of time
# https://susdesign.com/popups/sunangle/time-basis.php#:~:text=%22Local%20solar%20time%22%20(or,or%20north)%20at%20exactly%20noon.
# https://faculty.eng.ufl.edu/jonathan-scheffe/wp-content/uploads/sites/100/2020/08/Solar-Time1419.pdf
def eot(n):
    B = (n-1)*360/365 * np.pi/180 
    E = 229.2*(0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) - 0.014615 * np.cos(2*B) - 0.04089 * np.sin(2*B))
    return(E)


# re-index
def reindex_up_down(up, down):
# resample in up data wavelength index to match down data, linear interpolation
    Xresampled = down.index
    up_resampled = up.reindex(up.index.union(Xresampled)).interpolate(method='spline', order=2).loc[Xresampled]
    return(up_resampled)


def makenewdf(df, what):
    df['inc_'+what+'_deg'] = df['inc_'+what] * 180/np.pi
    df['inc_'+what+'_deg'] = df['inc_'+what+'_deg']#.round(1)
    df=df.loc[df['inc_'+what+'_deg']>0]
    df.index = df['inc_'+what+'_deg']
    #df = df.sort_index()

    return(df)
    # df=df.loc[df_flat.inc_flat>0].sort_index()




def plot_tilt(slope, df2, date):
    print(df2.head())
    offset = df2.inc_tilt.idxmax() - df2.inc_flat.idxmax()

    zcross_flat = df2[np.sign(df2['inc_flat']).diff().fillna(0).ne(0)].copy()
    zcross_tilt = df2[np.sign(df2['inc_tilt']).diff().fillna(0).ne(0)].copy()
    # print(offset)
    offsun = zcross_tilt.index - zcross_flat.index

    fig, ax = plt.subplots(1, 1, figsize= (12,8))
    ax.scatter(df2.index, df2.inc_flat*180/np.pi, label='sun angle, flat', s=4, color='k')
    ax.scatter(df2.index, df2.inc_tilt*180/np.pi, label='sun angle, tilt: '+str(slope), s=4, color='grey')

    ax.scatter(df2.inc_flat.idxmax(), df2.inc_flat.max()*180/np.pi, label='flat max', marker='s', color='blue')
    ax.scatter(df2.inc_tilt.idxmax(), df2.inc_tilt.max()*180/np.pi, label='tilt max', marker='s', color='orange')
    ax.axhline(y=0, linestyle='-', color='k', linewidth=1)

    df2.inc_tilt.idxmax() - df2.inc_flat.idxmax()
    ax.axvline(x=df2.inc_flat.idxmax(), label='', linestyle='-', color='blue', linewidth=0.8)
    ax.axvline(x=df2.inc_tilt.idxmax(), label='', linestyle='-', color='orange', linewidth=0.8)

    ax.text(pd.to_datetime(date+ ' 08:00',format='%Y-%m-%d %H:%M'), -20, 'offset sunrise, hours: '+ str(round(offsun[0].total_seconds()/3600, 2)),
             bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))
    ax.text(pd.to_datetime(date+ ' 08:00',format='%Y-%m-%d %H:%M'), -29, 'offset solar noon, hours: '+ str(round(offset.total_seconds()/3600, 2)),
             bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))
    ax.text(pd.to_datetime(date+ ' 08:00',format='%Y-%m-%d %H:%M'), -38, 'offset sunset, hours: '+ str(round(offsun[1].total_seconds()/3600, 2)),
             bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    xlims = pd.to_datetime([date+ ' 06:00', date+ ' 19:00'], format='%Y-%m-%d %H:%M')
    ylims = [-40, 25]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_ylabel('Sun angle (°)')
    ax.set_xlabel('Local solar time')
    ax.set_title('Sun angle for flat and tilted surface at WSS AWS location, date:' +date+ ', tilt in deg.:' +str(slope))
    ax.grid('both')
    ax.legend()

    fig.savefig('figs/sunangle_'+date+'_'+str(slope)+'.png', bbox_inches='tight')


def plot_measured_shifted(merged_0, date):
    # offset = merged_0.inc_tilt.idxmax() - merged_0.inc_flat.idxmax()
    # print(offset)
    fig, ax = plt.subplots(1, 1)
    maxin = merged_0.loc[merged_0.SWin_Avg == merged_0.SWin_Avg.max()]
    maxout = merged_0.loc[merged_0.SWout_Avg == merged_0.SWout_Avg.max()]

    merged_soltime = merged_0
    merged_soltime.index = merged_soltime['soltime_in']

    off = merged_soltime.SWout_Avg.idxmax()-merged_soltime.SWin_Avg.idxmax()

    xlims2 = pd.to_datetime([date+ ' 06:00', date+ ' 19:00'], format='%Y-%m-%d %H:%M')
    ax.set_xlim(xlims2)
    ax.plot(merged_soltime.index, merged_0.SWin_Avg, color='k', label='SW in', linestyle='-')
    ax.plot(merged_soltime.index, merged_0.SWout_Avg, color='r', label='SW out', linestyle='-')
    ax.plot(merged_soltime.index +off, merged_0.SWout_Avg, color='r', label='SW out, shifted by offset', linestyle='--')

    ax.grid('both')
    ax.set_xlabel('Local solar time')
    ax.legend()
    ax.text(pd.to_datetime(date+ ' 08:00',format='%Y-%m-%d %H:%M'), 50, 'offset max in - out, hours: '+ str(round((merged_soltime.SWout_Avg.idxmax()-merged_soltime.SWin_Avg.idxmax()).total_seconds()/3600, 2)),
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    merged_0['shifted_time'] = merged_0.apply(
            lambda x: pd.to_datetime(x['TIMESTAMP'])
            + pd.Timedelta(hours=x['ang_dif']*180/np.pi/15),
            axis=1,
        )
    ax.set_ylabel('SW rad (W/m2)')
    # ax[1].set_xlabel('Local solar time')
    ax.set_title('Measured SW radiation')
    plt.tight_layout()
    fig.savefig('figs/AWS_radiation_angles.png', dpi=200, bbox_inches='tight')


def compare_angles(df2, df2_3, df2_9, df2_12, date, slope):
    fig2, ax2 = plt.subplots(1, 1, figsize= (12,8))

    ax2.scatter(df2.index, df2.inc_flat*180/np.pi, label='sun angle, flat', s=4, color='k')
    ax2.scatter(df2_3.index, df2_3.inc_tilt*180/np.pi, label='sun angle, tilt: 3', s=4, color='magenta')
    ax2.scatter(df2.index, df2.inc_tilt*180/np.pi, label='sun angle, tilt: '+str(slope), s=4, color='grey')
    ax2.scatter(df2_9.index, df2_9.inc_tilt*180/np.pi, label='sun angle, tilt: 9', s=4, color='green')
    ax2.scatter(df2_12.index, df2_12.inc_tilt*180/np.pi, label='sun angle, tilt: 12', s=4, color='blue')
    xlims = pd.to_datetime([date+ ' 06:00', date+ ' 19:00'], format='%Y-%m-%d %H:%M')
    ax2.set_xlim(xlims)
    ylims = [-40, 25]
    ax2.set_ylim(ylims)
    ax2.axhline(y=0, linestyle='-', color='k', linewidth=1)
    ax2.grid('both')
    ax2.legend()
    ax2.set_title('Sun angle for flat and tilted surfaces at WSS AWS location, date:' +date)
    fig2.savefig('figs/AWS_radiation_angles_compare.png', dpi=200, bbox_inches='tight')




def getSolarNoon(some_date):
    AWS = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
    AWS = AWS.loc[AWS.stakename == 'AWS']
    AWS.to_crs(epsg=4326, inplace=True)

    o = ephem.Observer()
    lat, lon = AWS.geometry.y.values[0], AWS.geometry.x.values[0] 

    o.lat, o.long = str(AWS.geometry.y.values[0]), str(AWS.geometry.x.values[0]) 

    sun = ephem.Sun()

    yr = some_date.year
    mn = some_date.month
    dy = some_date.day
    # print(yr)
    # start=dt.datetime(yr, mn, dy, 23, 0)

    start=dt.datetime(2024, 12, 22, 23, 0) #2024/12/22 11:15:54
    start=dt.datetime(2024, 6, 22, 23, 0) #2024/6/22 11:19:16


    sunrise = o.previous_rising(sun, start=start)
    noon = o.next_transit(sun, start=sunrise)
    sunset = o.next_setting(sun, start=noon)

    print(noon)
    return(pd.to_datetime(noon))

# getSolarNoon(pd.to_datetime('2022-02-02'))


def getRadiation(some_date):
    AWS = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
    AWS = AWS.loc[AWS.stakename == 'AWS']
    AWS.to_crs(epsg=4326, inplace=True)

    lat, lon = AWS.geometry.y.values[0], AWS.geometry.x.values[0] 

    yr = some_date.year
    mn = some_date.month
    dy = some_date.day
    
    hh = some_date.hour
    mm = some_date.minute
    
    date=dt.datetime(yr, mn, dy, hh, mm, tzinfo=dt.timezone.utc)

    altitude_deg = get_altitude(lat, lon, date)
    rad = radiation.get_radiation_direct(date, altitude_deg)
    print(rad)
    # stop
    return(rad)
