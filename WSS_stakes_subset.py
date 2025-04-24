# ! /usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import geopandas as gpd
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
# from shapely.geometry import Point
import glob
# import contextily as cx
import fontawesome as fa
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep
import rasterio as rio
from rasterio.plot import show
from matplotlib_scalebar.scalebar import ScaleBar
from rio_color.operations import sigmoidal

from shapely.geometry import box
from shapely.geometry import Polygon

import rioxarray
import rasterio

# supress copy warning - careful 
pd.options.mode.chained_assignment = None  # default='warn'




# list of dates with known issues in the S2 data
nogood = pd.to_datetime(['2018-08-02', '2018-09-29', '2018-11-25', '2018-12-08',
                         '2019-01-27', '2019-04-02', '2019-04-04', '2019-08-22', '2019-08-30', '2019-12-15',
                         '2020-02-06', '2020-11-19', '2020-12-04',
                         '2021-01-31', '2021-12-14', '2021-11-19',
                         '2022-01-03', '2022-02-07', '2022-08-26', '2022-10-20', '2022-12-27', '2022-12-29',
                         '2023-01-13', '2023-02-20', '2023-12-29',
                             '2024-01-08', '2024-01-28', '2024-02-15', '2024-12-11'])

# # read excel file with OG stake data
# def ReadExcl(fname):
#     stakes = ['A', 'B', 'C', 'D', 'E', 'F', 'BL0319', 'Gams']
#     data = []
#     clrs = cm.tab10(np.linspace(0, 1, len(stakes)))
#     clrs_yr = cm.tab10(np.linspace(0, 1, len([2017, 2018, 2019, 2020, 2021, 2022, 2023])))

#     for j, st in enumerate(stakes):
#         temp = pd.read_excel(fname, st, header=1)
#         temp = temp[['Datum', 'Diff', 'SH']]
#         temp['name'] = st
#         temp['date0'] = temp['Datum'].shift()
#         temp = temp.loc[~temp.Datum.isnull()]
#         temp['color'] = mcolors.rgb2hex(clrs[j], keep_alpha=True)
    
#         data.append(temp)
    
#     stake_data = pd.concat(data)
#     stake_data['Datum'] = pd.to_datetime(stake_data['Datum'], format='%d/%m/%y')
#     stake_data['date0'] = pd.to_datetime(stake_data['date0']) 
#     stake_data['name'] = stake_data['name'].str.replace('BL0319','H')
#     stake_data['name'] = stake_data['name'].str.replace('Gams','G')
#     stake_data['period'] = (stake_data['Datum'] - stake_data['date0']).dt.days
#     stake_data['daily'] = stake_data['Diff'] / stake_data['period'] 
#     stake_data['color_yr'] = 'blue'

#     for js, yr in enumerate(stake_data.Datum.dt.year.unique()):
#         stake_data['color_yr'].loc[stake_data.Datum.dt.year==yr] = mcolors.rgb2hex(clrs_yr[js], keep_alpha=True)

#     return(stake_data, clrs_yr)



def StakesPlotsSub18(stake_data, dd, dh, nogood):

    # set file paths etc
    fldr = 'output_GEE/'

    clrs = ['blue', 'orange', 'green', 'purple', 'cyan', 'pink','red']#,'olive'
    pts2 = ['A', 'B', 'C', 'D', 'E', 'F', 'AWS']#,'g'

    start = '2018-07-30'
    end = '2018-08-31'
    yr = '2018'

    dd = dd.loc[(dd.index>=pd.to_datetime(start)) & (dd.index<=pd.to_datetime(end))]
 
    stakes_sub = stake_data.loc[stake_data.date1.dt.year == int(yr)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
    fs = 14
    myFmt2 = mdates.DateFormatter('%Y-%m-%d')

    #plot AWS albedo
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax.scatter(dd.index, dd.Albedo, c='red', marker='*', label='Albedo, in situ')
    ax.fill_between(dd.index, dd.Albedo-dd.Albedo*0.14, dd.Albedo+dd.Albedo*0.14, color='red', alpha=0.2, label='± 14%')
    lowalb=dd.loc[dd.Albedo<0.4]
    print(lowalb)
    print(dd)
    ax.scatter(lowalb.index, lowalb.Albedo, s=60 ,c='darkred', marker='*', label='Low albedo day, observation')
    ax.axvspan(pd.to_datetime('2018-08-17 00:00'),pd.to_datetime('2018-08-24 23:00'), alpha=0.2, color='grey', zorder=0, label='Low albedo days, estimate')

    ax.set_ylabel('Albedo', fontsize=fs)
    ax.set_ylim([0, 1])

    # loop through csv with sat albedo at the points and plot:
    for i, pt in enumerate(pts2):

        df = pd.read_csv(fldr+'extractpoint_'+pt+'_allyear_5.csv', index_col='dttime')
        df.index = pd.to_datetime(df.index)
        for no in nogood: 
            df.loc[(df.index.day==no.day) & (df.index.month==no.month) & (df.index.year==no.year)] = np.nan

        ax.scatter(df.index, df.liangAlbedo, c=clrs[i], marker='o', edgecolor='k', label='Albedo S2, '+pt)
        ax.errorbar(df.index, df.liangAlbedo, yerr=df.liangAlbedo*0.16, linestyle="None", color='k', linewidth=0.4, zorder=0)
        

    ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
    ax.grid('both')


    fig.savefig('figs/stakedata_albedo_subset_'+yr+'.png', dpi=200, bbox_inches='tight')



def StakesRGB18_Albedo(stake_data, stakes, AWS):
    fig, ax = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    stakes = stakes.loc[stakes.stakename.isin(['a', 'b', 'c', 'd', 'f'])]

    fs=14
    ax = ax.flatten()
    myFmt2 = mdates.DateFormatter('%Y-%m-%d')

    start = '2018-06-20'
    end = '2018-09-31'
    yr = '2018'

    satdates = ['180713', '180718', '180731', '180805', '180815', '180817', '180827']
    strings = pd.to_datetime(satdates, format='%y%m%d').strftime('%Y-%m-%d')

    for i, dt in enumerate(satdates):
        fn = '/Users/leahartl/Desktop/WSS/examples/S2_10px_WSS2018_subset_JulyAugust/'+dt+'.tif'
        with rio.open(fn) as src:
            sat = src.read()
            show(sat, transform=src.transform, ax=ax[i])

    for i, dt in enumerate(satdates):
        fn = '/Users/leahartl/Desktop/WSS/examples/S2_10px_WSS2018_subset_JulyAugust_multiband/'+dt+'.tif'
        # m = fl[-10:-4]
        # print(m)
        # dt = '20'+m
        S2 = rioxarray.open_rasterio(fn, masked=True)
        print(S2)

        S2 = S2.to_dataset('band')
        S2 = S2.rename({i + 1: name for i, name in enumerate(S2.attrs['long_name'])})

        # ND = S2['NIR'] / S2['SWIR1']
        # ND2 = S2['Red'] / S2['SWIR1']

        # S2_masked1 = S2.where(ND >= 1)
        S2_masked = S2#.where(ND2 >= 2)

        albedo= 0.356 * S2_masked['Blue'] + 0.130 * S2_masked['Red'] +0.373 * S2_masked['NIR'] + 0.085*S2_masked['SWIR1'] + 0.072*S2_masked['SWIR2'] -0.0018
       
        # albedo_clip = albedo.rio.clip(clip.geometry.values, clip.crs)
        levels=np.arange(0, 1, 0.1)
        cmap='plasma'
        im = albedo.plot.contour(ax=ax[i], levels=levels, cmap=cmap, alpha=1, add_colorbar=False, linewidths=1)
        albedo.plot.contour(ax=ax[i], levels=[0.4], colors='darkred', alpha=1, add_colorbar=False, linewidths=2)
        #ax[i].clabel(im, im.levels, inline=True, fontsize=10)

        # plt.show()

    ax[0].set_xlim([630750, 631100])
    ax[0].set_ylim([5.1894e6, 5.1896e6])

    #fig.subplots_adjust(right=0.8)

    # cbar_ax = fig.add_axes([0.2, -0.002, 0.4, 0.03])
    # fig.colorbar(im, cax=cbar_ax, cmap = cmap, label = 'S2 derived albedo', orientation='horizontal')

    for j, a in enumerate(ax[:-1]):
        AWS.plot(ax=a, alpha=1, color='red', marker='*', markersize=18)
        stakes.plot(ax=a, alpha=1, color='grey', marker='o', markersize=10, edgecolor='k', zorder=200)
        a.add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 12}))
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_ylabel('')
        a.set_xlabel('')
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(strings[j])
        for x, y, label in zip(stakes.geometry.x, stakes.geometry.y, stakes.stakename):
            a.annotate(label.upper(), xy=(x, y), xytext=(-4, 4), textcoords="offset points", fontsize=8)

    ax[-1].set_axis_off()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    cbar_ax = fig.add_axes([0.75, 0.14, 0.02, 0.3])
    fig.colorbar(im, cax=cbar_ax, cmap = cmap, label = 'S2 derived albedo')

    lns = []
    patch = Line2D([0], [0], linestyle='-', label='0.4 contour', color='darkred', linewidth=2)
    lns.append(patch)
    
    patch1 = Line2D([0], [0], marker='*', linestyle='None', label='AWS', color='red', markersize=4, zorder=10)
    lns.append(patch1)

    patch2 = Line2D([0], [0], marker='o', linestyle='None', label='Stakes', color='grey', markeredgecolor='k', markersize=4, zorder=10)
    lns.append(patch2)

    handles = (lns)
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.2), ncol=1, fontsize=12)

    fig.savefig('figs/stakedata_RGB_Albedo_subset_'+yr+'.png', dpi=200, bbox_inches='tight')



# make figure showing AWS albedo, AWS params, stake readings, S2 albedo at the stakes in summer 2022
# No surface elevation here bc SR50 data is too noisy.
def StakesPlotsSub22(stake_data, dd, dh, nogood):

    fldr = '/Users/leahartl/Desktop/WSS/satAlbedo/output/'
    pts2 = ['a', 'b', 'c', 'f', 'AWS']
    clrs = ['blue', 'orange', 'green', 'purple', 'red']

    fs = 14
    start = '2022-07-05'
    end = '2022-09-25'
    yr ='2022'

    dd = dd.loc[(dd.index>=pd.to_datetime(start)) & (dd.index<=pd.to_datetime(end))]

    snow1 = ['2022-07-28 12:00', '2022-07-30 12:00']
    snow2 = ['2022-08-20 08:00', '2022-08-21 04:00']
    snow3 = ['2022-08-27 00:00', '2022-08-27 23:00']
    snow4 = ['2022-08-31 12:00', '2022-09-01 12:00']
    snow5 = ['2022-09-03 12:00', '2022-09-04 12:00']
    snow6 = ['2022-09-08 12:00', '2022-09-11 12:00']
    snow7 = ['2022-09-14 12:00', '2022-09-19 12:00']
    snow8 = ['2022-09-24 13:00', '2022-09-25 23:00']
    # snow4 = ['2021-08-27 00:00', '2021-08-31 23:00']
    snow = [snow1, snow2, snow3, snow4, snow5, snow6, snow7, snow8]
    
    print(stake_data)
    #stop
    stakes_sub = stake_data.loc[stake_data.date1.dt.year == int(yr)]
    readings = stakes_sub.date1.unique() # dates of stake readings
    print(readings)
    dt1 = readings[0]
    dt2 = readings[1]

    ticks = pd.to_datetime(['2022-07-15', '2022-08-01', '2022-08-15', '2022-09-01', '2022-09-15'])
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = ax.flatten()
    myFmt2 = mdates.DateFormatter('%Y-%m-%d')
    ax[0].set_xlim(pd.to_datetime(start), pd.to_datetime(end))

    #plot AWS albedo
    ax[0].set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax[0].set_ylabel('Albedo', fontsize=16)
    ax[0].set_ylim([0, 1])

    # loop through csv with sat albedo at the points and plot:
    for i, pt in enumerate(pts2):
        df = pd.read_csv(fldr+'extractpoint_'+pt+'_allyear_5.csv', index_col='dttime', parse_dates=True)
        # print(df.head())
        # stop
        df.index = pd.to_datetime(df.index)
        for no in nogood: 
            df.loc[(df.index.day==no.day) & (df.index.month==no.month) & (df.index.year==no.year)] = np.nan

        # one bad value for Stake A in the file
        df = df.loc[(df.index < pd.to_datetime('2022-08-26 00:00')) | (df.index > pd.to_datetime('2022-08-27 00:00'))]
        if pt == 'BL0319':
            ax[0].scatter(df.index, df.liangAlbedo, c=clrs[i], marker='o', edgecolor='k', label='Albedo S2, H')
        else:
            ax[0].scatter(df.index, df.liangAlbedo, c=clrs[i], marker='o', edgecolor='k', label='Albedo S2, '+pt.upper())
    ax[0].scatter(dd.index, dd.Albedo, c='k', marker='*', label='Albedo, in situ')

    ax[0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize=16)
    ax[0].grid('both')


    ax[0].text(pd.to_datetime('2022-07-10'), 0.9, 'a', fontsize=12,
               bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))


    ax[1].text(pd.to_datetime('2022-07-10'), 7, 'b', fontsize=12,
               bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    ax1 = ax[1].twinx()
    ax[1].set_ylim(-8, 10)
    ax[1].plot(dh.index, dh.Tair_Avg, c='red', linestyle='-', linewidth=0.5, label='Air temperature')
    ax[1].set_ylabel('Temperature (°C)', fontsize=16)
    ax[1].axhline(y=0, linewidth=1, color='k', zorder=10)

    ax[1].spines["left"].set_edgecolor('red')
    #ax[0].spines["right"].set_edgecolor('red')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.2, 1.2), fontsize=14)
    ax[1].grid('both')
    ax[1].set_xticks(ticks)
    ax[1].xaxis.set_major_formatter(myFmt2)

    # ax1.plot(dh.index, dh.SWin_Avg, c='k', linestyle='-', linewidth=0.5, label='SW incoming')
    # ax1.plot(dh.index, dh.SWout_Avg, c='grey', linestyle='--', linewidth=0.5, label='SW reflected')

    ax1.plot(dh.index, dh.SWin_Avg - dh.SWout_Avg, c='k', linestyle='-', linewidth=0.5, label='SW incoming - SW reflected')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.7, 1.2), fontsize=14)
    ax1.set_ylabel('Net SW radiation ($W m^{-2}$)', fontsize=16)
    ax1.set_ylim(0, 1000)

    ax[0].axvline(x=dt1, linewidth=1, color='k', zorder=10)
    ax[0].axvline(x=dt2, linewidth=1, color='k', zorder=10)

    ax[0].text(dt1-pd.to_timedelta(8, unit='d'), 1.05,'Stake reading Aug. 4', fontsize=14)
    ax[0].text(dt2-pd.to_timedelta(8, unit='d'), 1.05,'Stake reading Sep. 20', fontsize=14)

    for a in ax:
        a.tick_params(axis='x', labelsize=14)
        a.tick_params(axis='y', labelsize=16)

    ax1.tick_params(axis='y', labelsize=16)


    fig.savefig('figs/stakedata_albedo_subset_'+yr+'.png', dpi=200, bbox_inches='tight')



def StakesRGB22_Albedo(stake_data, stakes, AWS):
    fig, ax = plt.subplots(3, 4, figsize=(10, 5), sharex=True, sharey=True)
    fs=14
    ax = ax.flatten()
    myFmt2 = mdates.DateFormatter('%Y-%m-%d')

    yr = '2022'

    stakes = stakes.loc[stakes.stakename.isin(['a', 'b', 'c', 'f'])]
    # GI5g = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/mergedGI5_3.shp')
    # GI5g.to_crs(epsg=32632, inplace=True)

    satdates = ['220712', '220717', '220722', '220804', '220809', '220814', '220816',
                '220821', '220824', '220829', '220913', '220923']
    strings = pd.to_datetime(satdates, format='%y%m%d').strftime('%Y-%m-%d')

    for i, dt in enumerate(satdates):
        fn = '/Users/leahartl/Desktop/WSS/examples/S2_10px_WSS2022_subset_JulyAugust/'+dt+'.tif'
        with rio.open(fn) as src:
            sat = src.read()
            show(sat, transform=src.transform, ax=ax[i])

    for i, dt in enumerate(satdates):
        fn = '/Users/leahartl/Desktop/WSS/examples/S2_10px_WSS2022_subset_JulyAugust_multiband/'+dt+'.tif'

        S2 = rioxarray.open_rasterio(fn, masked=True)
        print(S2)

        S2 = S2.to_dataset('band')
        S2 = S2.rename({i + 1: name for i, name in enumerate(S2.attrs['long_name'])})

        # ND = S2['NIR'] / S2['SWIR1']
        # ND2 = S2['Red'] / S2['SWIR1']

        # S2_masked1 = S2.where(ND >= 1)
        S2_masked = S2#.where(ND2 >= 2)

        albedo= 0.356 * S2_masked['Blue'] + 0.130 * S2_masked['Red'] +0.373 * S2_masked['NIR'] + 0.085*S2_masked['SWIR1'] + 0.072*S2_masked['SWIR2'] -0.0018
       
        # albedo_clip = albedo.rio.clip(clip.geometry.values, clip.crs)
        levels=np.arange(0, 1, 0.1)
        cmap='plasma'
        im = albedo.plot.contour(ax=ax[i], levels=levels, cmap=cmap, alpha=1, add_colorbar=False, linewidth=0.02)
        albedo.plot.contour(ax=ax[i], levels=[0.4], cmap='Reds', alpha=1, add_colorbar=False, linewidth=0.08, inline=True)

    ax[0].set_xlim([630750, 631100])
    ax[0].set_ylim([5.1894e6, 5.1896e6])

    #fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.2, -0.002, 0.4, 0.03])
    fig.colorbar(im, cax=cbar_ax, cmap = cmap, label = 'S2 derived albedo', orientation='horizontal')



    ax[0].set_xlim([630750, 631100])
    ax[0].set_ylim([5.1894e6, 5.1896e6])
    ax[0].add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 12}))

    for j, a in enumerate(ax):
        AWS.plot(ax=a, alpha=1, color='red', marker='*', markersize=10)
        stakes.plot(ax=a, alpha=1, color='grey', marker='o', markersize=8, edgecolor='k', linewidth=0.1, zorder=200)
        # GI5g.boundary.plot(color='lime', ax=a)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylabel('')
        a.set_xlabel('')
        a.set_title(strings[j])
        for x, y, label in zip(stakes.geometry.x, stakes.geometry.y, stakes.stakename):
            a.annotate(label, xy=(x, y), xytext=(-4, 4), textcoords="offset points", fontsize=8)

    #ax[-1].set_axis_off()
    plt.subplots_adjust(wspace=0.0, hspace=0.1)

    lns = []
    patch = Line2D([0], [0], linestyle='-', label='0.4 contour', color='darkred', linewidth=2)
    lns.append(patch)

    # patchOutline = Line2D([0], [0], linestyle='-', label='Glacier boundary', color='lime', linewidth=2)
    # lns.append(patchOutline)

    patch1 = Line2D([0], [0], marker='*', linestyle='None', label='AWS', color='red', markersize=4, zorder=10)
    lns.append(patch1)

    patch2 = Line2D([0], [0], marker='o', linestyle='None', label='Stakes', color='grey', markersize=4, markeredgecolor='k', zorder=10)
    lns.append(patch2)

    handles = (lns)
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.9, -0.15), ncol=1, fontsize=20)



    fig.savefig('figs/stakedata_RGB_Albedo_subset_'+yr+'.png', dpi=200, bbox_inches='tight')


def StakesRGB22_Albedo_2(stake_data, stakes, AWS):
    fig, ax = plt.subplots(4, 3, figsize=(8, 9), sharex=True, sharey=True)
    fs=14
    ax = ax.flatten()
    myFmt2 = mdates.DateFormatter('%Y-%m-%d')

    yr = '2022'
    print(stakes)
    stakes = stakes.loc[stakes.name.isin(['A', 'B', 'C', 'F'])]
    # GI5g = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/mergedGI5_3.shp')
    # GI5g.to_crs(epsg=32632, inplace=True)

    satdates = ['220712', '220717', '220722', '220804', '220809', '220814', '220816',
                '220821', '220824', '220829', '220913', '220923']
    strings = pd.to_datetime(satdates, format='%y%m%d').strftime('%Y-%m-%d')

    for i, dt in enumerate(satdates):
        fn = '/Users/leahartl/Desktop/WSS/examples/S2_10px_WSS2022_subset_JulyAugust/'+dt+'.tif'
        with rio.open(fn) as src:
            sat = src.read()
            show(sat, transform=src.transform, ax=ax[i])

    for i, dt in enumerate(satdates):
        fn = '/Users/leahartl/Desktop/WSS/examples/S2_10px_WSS2022_subset_JulyAugust_multiband/'+dt+'.tif'

        S2 = rioxarray.open_rasterio(fn, masked=True)
        print(S2)

        S2 = S2.to_dataset('band')
        S2 = S2.rename({i + 1: name for i, name in enumerate(S2.attrs['long_name'])})

        # ND = S2['NIR'] / S2['SWIR1']
        # ND2 = S2['Red'] / S2['SWIR1']

        # S2_masked1 = S2.where(ND >= 1)
        S2_masked = S2#.where(ND2 >= 2)

        albedo = 0.356 * S2_masked['Blue'] + 0.130 * S2_masked['Red'] +0.373 * S2_masked['NIR'] + 0.085*S2_masked['SWIR1'] + 0.072*S2_masked['SWIR2'] -0.0018
       
        # albedo_clip = albedo.rio.clip(clip.geometry.values, clip.crs)
        levels=np.arange(0, 1, 0.1)
        cmap='plasma'
        im = albedo.plot.contour(ax=ax[i], levels=levels, cmap=cmap, alpha=1, add_colorbar=False, linewidth=0.02)
        albedo.plot.contour(ax=ax[i], levels=[0.4], cmap='Reds', alpha=1, add_colorbar=False, linewidth=0.08, inline=True)

    ax[0].set_xlim([630750, 631100])
    ax[0].set_ylim([5.1894e6, 5.1896e6])

    #fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.02, 0.2, 0.04, 0.3])
    fig.colorbar(im, cax=cbar_ax, cmap = cmap, label = 'S2 derived albedo', orientation='vertical')


    minx=630750
    maxx=631100
    miny=5.1894e6
    maxy=5.1896e6
    ax[0].set_xlim([630750, 631100])
    ax[0].set_ylim([5.1894e6, 5.1896e6])
    ax[0].add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 12}))

    for j, a in enumerate(ax):
        AWS.plot(ax=a, alpha=1, color='red', marker='*', markersize=16)
        stakes.plot(ax=a, alpha=1, color='skyblue', marker='o', markersize=18, edgecolor='k', linewidth=0.1, zorder=200)
        # GI5g.boundary.plot(color='lime', ax=a)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylabel('')
        a.set_xlabel('')
        a.set_title(strings[j])
        for x, y, label in zip(stakes.geometry.x, stakes.geometry.y, stakes.name):
            a.annotate(label, xy=(x, y), xytext=(-4, 4), textcoords="offset points", fontsize=8)

    #ax[-1].set_axis_off()
    plt.subplots_adjust(wspace=0.0, hspace=0.1)

    lns = []
    patch = Line2D([0], [0], linestyle='-', label='0.4 contour', color='darkred', linewidth=2)
    lns.append(patch)

    # patchOutline = Line2D([0], [0], linestyle='-', label='Glacier boundary', color='lime', linewidth=2)
    # lns.append(patchOutline)

    patch1 = Line2D([0], [0], marker='*', linestyle='None', label='AWS', color='red', markersize=6, zorder=10)
    lns.append(patch1)

    patch2 = Line2D([0], [0], marker='o', linestyle='None', label='Stakes', color='skyblue', markersize=4, markeredgecolor='k', zorder=10)
    lns.append(patch2)

    handles = (lns)
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.25, 0.5), ncol=1, fontsize=18)


    axins = ax[2].inset_axes([1.15, -0.3, 1.2, 1.3])
    satrasterLarge = '/Users/leahartl/Desktop/WSS/examples/2022_07_17.tif'
    with rio.open(satrasterLarge) as src3:
        satL = src3.read()

        show(sigmoidal(satL, 6, 0.25), transform=src3.transform, ax=axins)
        cr = src3.crs
        # bounds  = src3.bounds
    lon_lat_list = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]

    polygon_geom = Polygon(lon_lat_list)
    box1 = gpd.GeoDataFrame(index=[0], crs='epsg:32632', geometry=[polygon_geom])    
    box1.to_crs(crs=cr, inplace=True)   
    box1.boundary.plot(ax=axins, alpha=1, color='red', linewidth=2)
    axins.set_ylim(5.189e6, 5.1907e6)
    axins.set_xlim(630500, 633000)
    axins.add_artist(ScaleBar(dx=1, location="upper right", font_properties={"size": 12}))
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([])
    axins.set_yticks([])

    

    fig.savefig('figs/stakedata_RGB_Albedo_subset_'+yr+'_2.png', dpi=200, bbox_inches='tight')


# set some file names, load shapefiles
# stakesfile = '/Users/leahartl/Desktop/WSS/Pegel_bea2.xlsx'

stakes = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
stakes.to_crs(epsg=32632, inplace=True)
# rename borehole stake:
#stakes['stakename'].loc[stakes.stakename=='BL0319'] = 'h'

# separate AWS from stakes:
AWS = stakes.loc[stakes.name == 'AWS']
stakes = stakes.loc[stakes.name != 'AWS']



# # # read AWS file and compute albedo:
# file1 = '/Users/leahartl/Desktop/WSS/AWS_wss/out/AWS_WSS_proc.csv'
# data1 = pd.read_csv(file1, parse_dates=True, skiprows=[0,2,3], index_col=0)
# data1 = data1[['Tair_Avg', 'SWin_Avg', 'SWout_Avg']]
# dh = data1.resample('H').mean()

# data_daily = data1.copy()
# data_daily[data_daily['SWin_Avg'] < 2] = np.nan
# data_daily[data_daily['SWout_Avg']< 2] = np.nan
# data_daily['albedo_sum'] = data_daily['SWout_Avg'].resample('D').sum()/data_daily['SWin_Avg'].resample('D').sum()

def procforfunc(data):
    # file contains hourly data produced as cosipy input, i.e. "daily" value for every hour of the day 
    dd = data.resample('d').mean()
    # set time so the marker is placed correctly in the plot
    dd['hour'] = 11
    dd.hour = pd.to_timedelta(dd.hour, unit='h')
    dd.index = dd.index + dd.hour
    return(dd)

# load stake file and assign colors to be used in plotting functions
def ReadStakes(fname):
    stakes = pd.read_csv(fname, parse_dates=True)
    stakes['date1'] = pd.to_datetime(stakes['date1'])
    sts = stakes['name'].unique()
    stakes['daily'] = stakes['ice_ablation_mmwe'] / stakes['period'] 

    clrs = cm.tab10(np.linspace(0, 1, len(sts)))#np.linspace(0.7, 0.9, len(fls)))
    clrs_yr = cm.tab10(np.linspace(0, 1, len([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])))
    stakes['color'] = ''
    stakes['color_yr'] = ''

    for j, st in enumerate(sts):
        stakes['color'].loc[stakes['name']==st] = mcolors.rgb2hex(clrs[j], keep_alpha=True)
   
    for js, yr in enumerate(stakes.date1.dt.year.unique()):
        stakes['color_yr'].loc[stakes.date1.dt.year==yr] = mcolors.rgb2hex(clrs_yr[js], keep_alpha=True)

    return(stakes, clrs_yr)




# file21 = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2021_summer_sub4plot.csv'#August_full.csv'
# data21 = pd.read_csv(file21, parse_dates=True, index_col=0)

# dd21 = procforfunc(data21)

file22 = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2022_summer_sub4plot.csv'#JulAugust_full.csv'
data22 = pd.read_csv(file22, parse_dates=True, index_col=0)

dd22 = procforfunc(data22)


file18 = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2018_summer_sub4plot.csv'
data18 = pd.read_csv(file18, parse_dates=True, index_col=0)

dd18 = procforfunc(data18)


# read excel file
stakesfile  = '/Users/leahartl/Desktop/WSS/process_stakes/WSS_stakes_point_mass_balance.csv'
# read csv file with stake data:
stake_data, clrs_yr =ReadStakes(stakesfile)



# paper fig: Summer 2022 temp, rad, albedo
# StakesPlotsSub22(stake_data, dd22, data22, nogood)
# plt.show()
# stop
# # paper fig: Albedo maps in WSS summit area summer 2022:
# StakesRGB22_Albedo_2(stake_data, stakes, AWS)

# sup fig - August 2018: 
StakesPlotsSub18(stake_data, dd18, data18, nogood)





# StakesPlotsSub21(stake_data, dd21, data21, nogood)
# StakesPlotsSub22(stake_data, dd22, data22, nogood)

# StakesRGB18_Albedo(stake_data, stakes, AWS)
# StakesRGB21_Albedo(stake_data, stakes, AWS)



plt.show()

