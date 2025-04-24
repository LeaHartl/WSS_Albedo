import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import netCDF4 as nc
import time
import dateutil
from itertools import product
import metpy.calc
from metpy.units import units
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

# '/Users/leahartl/Desktop/WSS/figs/averagemelt_model2025.csv'

Y = 2021

if Y == 2021:
    inp = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/input/WSS/AWS_WSS_4cosipy_2021_summer_Aug05_new.nc')
    inp_csv = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2021_summer_Aug05_new.csv', parse_dates=True, index_col=0)
    ds = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2021_summer_Aug05_new_out.nc')

    inp_csv_5dark = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2021_summer_Aug05_new5daysdark.csv', parse_dates=True, index_col=0)
    ds_5dark = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2021_summer_Aug05_new5daysdark_out.nc')

    inp_csv_0dark = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2021_summer_Aug05_new0daysdark.csv', parse_dates=True, index_col=0)
    ds_0dark = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2021_summer_Aug05_new0daysdark_out.nc')

    time_start = '2021-08-05T00:00'   # '2021-08-01T00:00' # '2017-11-01T00:00'
    time_end   = '2021-08-31T23:00' 


if Y == 2022:
    inp = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/input/WSS/AWS_WSS_4cosipy_2022_summer.nc')
    inp_csv = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_2022_summer.csv', parse_dates=True, index_col=0)
    ds = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2022_summer_out.nc')
    time_start = '2022-07-05T00:00'   # '2021-08-01T00:00' # '2017-11-01T00:00'
    time_end   = '2022-09-30T23:00' 

print(inp.variables)

print(ds.variables)

inp_csv = inp_csv[time_start:time_end]
print(inp_csv.head())


def fig1_1(inp_csv, ds):
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = ax.flatten()
    ax1 = ax[0].twinx()
    inp_csv_D = inp_csv.resample('D').sum()
    print(inp_csv_D.Snowfall)
    ax1.bar(inp_csv_D.index, inp_csv_D.Snowfall*100, zorder=1, alpha=0.2, label='Daily snow fall (cm)')
    
    bias = pd.DataFrame(columns=['obs', 'mod'])
    bias['obs'] = inp_csv.resample('H').mean().Surf
    bias['mod'] = ds.TOTALHEIGHT.squeeze()-6

    daily = bias.resample('D').mean()
    # print(daily)

    ax1.step(inp_csv.index, inp_csv.Albedo*100, label='albedo as measured at AWS (%)', color='red')
    ax1.set_ylabel('Albedo (%), daily snow fall (cm)')
    ax[0].plot(daily.index, daily['obs'], label='daily mean surface height, AWS', color='k', zorder=100)
    ax[0].plot(daily.index, daily['mod'], label='daily mean surface height, model', color='grey',zorder=100)
    # ax.plot(inp_csv.index, inp_csv.Surf, label='surface height, AWS', color='k', zorder=100)
    # ax.plot(ds.time, ds.TOTALHEIGHT.squeeze()-6, label='surface height, model', color='grey',zorder=100)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.4, 0.98))
    ax[0].axhline(y=0, color='k')


    daily['dif_cm'] = (daily['obs'] - daily['mod'])*100
    daily['rate_obs_cm'] = (daily['obs'].diff())*100
    daily['rate_mod_cm'] = (daily['mod'].diff())*100
    daily['dif_rates'] = daily['rate_obs_cm'] - daily['rate_mod_cm']
    daily['cumsum_obs'] = daily['rate_obs_cm'].cumsum()
    daily['cumsum_mod'] = daily['rate_mod_cm'].cumsum()
    print(daily)

    # daily_sub = daily.loc[daily['rate_obs_cm']<0]
    
    rmse = mean_squared_error(daily['rate_obs_cm'].values[1:], daily['rate_mod_cm'].values[1:])
    r2 = r2_score(daily['rate_obs_cm'].values[1:], daily['rate_mod_cm'].values[1:])
    absbias = (daily['rate_obs_cm'].abs() - daily['rate_mod_cm'].abs()).mean()
    print('rmse',rmse)
    print('r2',r2)
    print('abs',absbias)

    # rmse_melt = mean_squared_error(daily_sub['rate_obs_cm'].values[1:], daily_sub['rate_mod_cm'].values[1:])
    # r2_melt = r2_score(daily_sub['rate_obs_cm'].values[1:], daily_sub['rate_mod_cm'].values[1:])
    # absbias_melt = (daily_sub['rate_obs_cm'].abs() - daily_sub['rate_mod_cm'].abs()).mean()
    # print('rmse_melt',rmse_melt)
    # print('r2_melt',r2_melt)
    # print('abs_melt',absbias_melt)



    ax1.legend()
    ax[0].set_ylabel('Surface height (m)')

    ax[1].plot(daily.index, daily['rate_obs_cm'], label='daily change rate, AWS', color='k', zorder=100)
    ax[1].plot(daily.index, daily['rate_mod_cm'], label='daily change rate, model', color='grey', zorder=100)
    ax[1].grid('both')
    ax[1].legend()
    ax[1].set_ylabel('Change rate (cm day$^{-1}$)')

    ticks = pd.to_datetime(['2021-08-05', '2021-08-10', '2021-08-15', '2021-08-20', '2021-08-25', '2021-08-30',])
    ax[1].set_xticks(ticks)
    ax[1].set_xlim(pd.to_datetime('2021-08-05'), pd.to_datetime('2021-09-01'))

  
    fig.savefig('figs/model_August2021_SIMPLE.png', bbox_inches='tight', dpi=200)



def fig2():
    minalb = 0
    maxalb = 1

    df = pd.DataFrame(columns=np.arange(minalb, maxalb, 0.05).round(decimals=2))

    for alb in np.arange(minalb, maxalb, 0.05).round(decimals=2):
        fn = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_'+str(alb.round(decimals=2))+'.nc'
        ds = xr.open_dataset(fn)

        ds_d = ds.surfMB.squeeze().to_pandas()
        ds_d.index = pd.to_datetime(ds_d.index)
        # print(ds_d)
        ds_d = ds_d.resample('D').sum()

        df[alb] = ds_d
    print(df)
    df = df*1000

    ds_real = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2018_2024.nc')
    ds_d_real = ds_real.surfMB.squeeze().to_pandas()
    ds_d_real.index = pd.to_datetime(ds_d_real.index)
    ds_d_real = ds_d_real.resample('D').sum()

    df_real = pd.DataFrame(columns=['surfMB', 'albedo'])
    df_real['surfMB'] = ds_d_real
    df_real['albedo'] = ds_real.ALBEDO.squeeze().to_pandas().resample('D').mean()
    # print(df_real)
    # print(df_real.loc[df_real.index.month==7])

    mean15D = df.resample('15D').mean()
    # print(mean15D[[0.10, 0.15, 0.20, 0.30, 0.40, 0.60]])
    sub4table = mean15D[[0.10, 0.15, 0.20, 0.30, 0.40, 0.60]].round(decimals=0)
    print(df.head())
    print('Jul mean:',df[df.index.month==7].resample('D').mean().mean())
    print('Aug mean:',df[df.index.month==8].resample('D').mean().mean())
    print('Sep mean:',df[df.index.month==9].resample('D').mean().mean())
    print('All mean:',df.resample('D').mean().mean())

    #stop
    sub4table.to_csv('out/averagemelt_model2025.csv')
    # print(df.resample('15D').mean())

    x, y = np.meshgrid(df.index.values, df.columns.values) 
    x_r, y_r = np.meshgrid(df_real.index.values, df_real.columns.values) 

    # fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharex=False)
    # ax = ax.flatten()
    fig = plt.figure(figsize=(12, 8))
    widths = [2, 4]
    heights = [2, 2]
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=widths, height_ratios=heights, wspace=0.32, hspace=0.3)
    ax_0 = fig.add_subplot(spec[0, 0])
    ax_1 = fig.add_subplot(spec[1, 0])
    # ax_2 = fig.add_subplot(spec[2, 0])

    ax_3 = fig.add_subplot(spec[:, 1])

    
    inp = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_mean_2018_2024.csv', parse_dates=True, index_col=0)
    inp_d = inp.resample('D').mean()
    # inp_d = inp_d['1900-07-01':'1900-10-01']
    ax_0.plot(inp_d.index, inp_d.Tair_Avg, color='k', label='Air temperature', linewidth=0.5)
    ax0 = ax_0.twinx()
    ax0.plot(inp_d.index, inp_d.Albedo, color='red', label='Albedo', linewidth=0.5)
    # ax_0.legend()
    # ax0.legend()
    ax_0.set_ylabel('Temperature ($°C$)', fontsize=16)
    ax0.set_ylabel('Albedo', fontsize=16)

    ax_0.set_xlim(pd.to_datetime('1900-01-01'), pd.to_datetime('1900-12-31'))
    ax_0.xaxis.set_tick_params(labelbottom=False)
    ax0.xaxis.set_tick_params(labelbottom=False)

    ax = [ax_0, ax_1, ax_3, ax0]
    for a in ax:
        a.tick_params(axis='x', labelsize=14)
        a.tick_params(axis='y', labelsize=14)

    lnAlb = Line2D([0], [0], label='Albedo', color='r', linewidth=1)
    lnTmp = Line2D([0], [0], label='Air temperature', color='k', linewidth=1)
    leg_1 = [lnTmp, lnAlb]
    plt.legend(handles=leg_1, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True, fontsize=14)


    ax_1.set_xlim(pd.to_datetime('1900-01-01'), pd.to_datetime('1900-12-31'))
    ticks = pd.to_datetime(['1900-03-01', '1900-06-01', '1900-09-01', '1900-12-01'])
    ax_0.set_xticks(ticks)
    ax_1.set_xticks(ticks)

    ax_1.plot(inp_d.index, inp_d.SWin_Avg, color='k', label='SW in', linewidth=0.5)
    ax_1.plot(inp_d.index, inp_d.SWout_Avg, color='orange', label='SW out', linestyle='-', linewidth=0.5)
    ax_1.plot(inp_d.index, inp_d.LWin_Cor, color='blue', label='LW in', linewidth=1)
    ax_1.plot(inp_d.index, inp_d.LWout_Cor, color='cyan', label='LW out', linestyle='-', linewidth=1)

    lnSWin = Line2D([0], [0], label='SW in', color='k', linewidth=1)
    lnSWout = Line2D([0], [0], label='SW out', color='orange', linewidth=1, linestyle='-')
    lnLWin = Line2D([0], [0], label='LW in', color='blue', linewidth=1)
    lnLWout = Line2D([0], [0], label='LW out', color='cyan', linewidth=1, linestyle='-')


    ax_1.legend(handles=[lnSWin, lnSWout, lnLWin, lnLWout], loc='upper center', bbox_to_anchor=(0.5,1.3), ncol=2, frameon=True, fontsize=14)

    # ax_1.legend()
    ax_1.set_ylabel('Radiation ($W m^{-2}$)', fontsize=16)


    ax_0.axvspan(pd.to_datetime('1900-07-01'), pd.to_datetime('1900-10-01'), alpha=0.4, color='lightgrey')
    ax_1.axvspan(pd.to_datetime('1900-07-01'), pd.to_datetime('1900-10-01'), alpha=0.4, color='lightgrey')

   
    cmap = 'plasma'
    levels = np.arange(-65, 10, 5)
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=256, extend='both')
    # levels=levels
    im = ax_3.pcolormesh(x, y, df.T.values, cmap=cmap, norm=norm)#,shading='flat')
    ax_3.set_ylim(0, 0.95)
    ax_3.scatter(df_real.index, df_real.albedo, c=df_real.surfMB*1000, cmap=cmap, norm=norm, edgecolor='k')
    ax_3.set_xticklabels(ax_3.get_xticklabels(), rotation=45, ha='right')

    cbar = fig.colorbar(im, extend='both', label='daily surface MB (mm w.e.)')#, ticks=[-1, 0, 1])
    # cbar.ax.tick_params(labelsize=12) 
    cbar.set_label('daily surface MB (mm w.e.)', size=14)
    cbar.ax.tick_params(labelsize=14)

    myFmt = mdates.DateFormatter('%b-%d')
    myFmt2 = mdates.DateFormatter('%b')
    ax_0.xaxis.set_major_formatter(myFmt2)
    ax_1.xaxis.set_major_formatter(myFmt2)
    # ax_2.xaxis.set_major_formatter(myFmt)
    ax_3.xaxis.set_major_formatter(myFmt)
    ax_0.grid('both')
    ax_1.grid('both')
    # ax_2.grid('both')
    ax_3.grid('both')
    ax_3.set_ylabel('Albedo', fontsize=16)

    ax_0.text(pd.to_datetime('1900-01-12'), 2, 'a', 
             bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))
    ax_1.text(pd.to_datetime('1900-01-12'), 510, 'b', 
             bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))
    ax_3.text(pd.to_datetime('1900-07-04'), 0.9, 'c', 
             bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))



    fig.savefig('figs/model_albedo_meanclim2025.png', bbox_inches='tight', dpi=200)




def fig2_forposter():
    minalb = 0
    maxalb = 1

    df = pd.DataFrame(columns=np.arange(minalb, maxalb, 0.05).round(decimals=2))

    for alb in np.arange(minalb, maxalb, 0.05).round(decimals=2):
        fn = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_'+str(alb.round(decimals=2))+'.nc'
        ds = xr.open_dataset(fn)

        ds_d = ds.surfMB.squeeze().to_pandas()
        ds_d.index = pd.to_datetime(ds_d.index)
        # print(ds_d)
        ds_d = ds_d.resample('D').sum()

        df[alb] = ds_d
    print(df)
    df = df*1000

    ds_real = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2018_2024.nc')
    ds_d_real = ds_real.surfMB.squeeze().to_pandas()
    ds_d_real.index = pd.to_datetime(ds_d_real.index)
    ds_d_real = ds_d_real.resample('D').sum()

    df_real = pd.DataFrame(columns=['surfMB', 'albedo'])
    df_real['surfMB'] = ds_d_real
    df_real['albedo'] = ds_real.ALBEDO.squeeze().to_pandas().resample('D').mean()
    # print(df_real)
    # print(df_real.loc[df_real.index.month==7])



    x, y = np.meshgrid(df.index.values, df.columns.values) 
    x_r, y_r = np.meshgrid(df_real.index.values, df_real.columns.values) 

    fig, ax_3 = plt.subplots(1, 1, figsize=(8, 8), sharex=False)


    cmap = 'plasma'
    levels = np.arange(-65, 10, 5)
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=256, extend='both')
    # levels=levels
    im = ax_3.pcolormesh(x, y, df.T.values, cmap=cmap, norm=norm)#,shading='flat')
    ax_3.set_ylim(0, 0.95)
    ax_3.scatter(df_real.index, df_real.albedo, c=df_real.surfMB*1000, cmap=cmap, norm=norm, edgecolor='k', label='2018-2024 mean albedo at AWS')
    ax_3.set_xticklabels(ax_3.get_xticklabels(), rotation=45, ha='right')

    cbar = fig.colorbar(im, extend='both', label='daily surface MB (mm w.e.)')#, ticks=[-1, 0, 1])
    # cbar.ax.tick_params(labelsize=12) 
    cbar.set_label('daily surface MB (mm w.e.)', size=20)
    cbar.ax.tick_params(labelsize=18)
    ax_3.xaxis.set_tick_params(labelsize=20)
    ax_3.yaxis.set_tick_params(labelsize=20)
    ax_3.legend()

    myFmt = mdates.DateFormatter('%b-%d')
    myFmt2 = mdates.DateFormatter('%b')

    ax_3.xaxis.set_major_formatter(myFmt)
    # ax_2.grid('both')
    ax_3.grid('both')
    ax_3.set_ylabel('Albedo', fontsize=20)




    fig.savefig('figs/model_albedo_meanclim_poster.png', bbox_inches='tight', dpi=400, transparent=True)







def Avg_vs_2022():


    minalb = 0
    maxalb = 1

    df = pd.DataFrame(columns=np.arange(minalb, maxalb, 0.05).round(decimals=2))

    for alb in np.arange(minalb, maxalb, 0.05).round(decimals=2):
        fn = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_'+str(alb.round(decimals=2))+'.nc'
        ds = xr.open_dataset(fn)

        ds_d = ds.surfMB.squeeze().to_pandas()
        ds_d.index = pd.to_datetime(ds_d.index)
        # print(ds_d)
        ds_d = ds_d.resample('D').sum()

        df[alb] = ds_d
    print(df)
    df = df*1000

    ds_real = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_2018_2024.nc')
    ds_d_real = ds_real.surfMB.squeeze().to_pandas()
    ds_d_real.index = pd.to_datetime(ds_d_real.index)
    ds_d_real = ds_d_real.resample('D').sum()

    df_real = pd.DataFrame(columns=['surfMB', 'albedo'])
    df_real['surfMB'] = ds_d_real
    df_real['albedo'] = ds_real.ALBEDO.squeeze().to_pandas().resample('D').mean()
    # print(df_real)
    # print(df_real.loc[df_real.index.month==7])

    # ax = ax.flatten()
    fig = plt.figure(figsize=(12, 8))
    widths = [2, 2]
    heights = [2, 2]
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=widths, height_ratios=heights, wspace=0.32, hspace=0.3)
    ax_0 = fig.add_subplot(spec[0, 0])
    ax_1 = fig.add_subplot(spec[1, 0])
    # ax_2 = fig.add_subplot(spec[2, 0])

    ax_3 = fig.add_subplot(spec[:, 1])

    ax = [ax_0, ax_1, ax_3]

    for a in ax:
        a.tick_params(axis='x', labelsize=14)
        a.tick_params(axis='y', labelsize=14)

    inp = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_mean_2018_2024.csv', parse_dates=True, index_col=0)
    inp = inp.loc[pd.to_datetime('1900-07-15'): pd.to_datetime('1900-07-23')]
    inp_d = inp.resample('D').mean()
    print(inp.head())

    inp_csv = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_heatwave2022.csv', parse_dates=True, index_col=0)
    inp_csv.index = inp_csv.index.map(lambda t: t.replace(year=1900))
    inp_csv_d = inp_csv.resample('D').mean()

    # plot temperature for mean and 2022 heatwave
    ax_0.plot(inp.index, inp.Tair_Avg, color='k', label='avg. temp.', linewidth=0.5, linestyle='--')
    ax_0.plot(inp_csv.index, inp_csv.Tair_Avg, label='2022 temp.', color='k')
    ax0 = ax_0.twinx()
    ax0.tick_params(axis='y', labelsize=14)

    # ax_0.legend()
    # ax0.legend()
    # plot albedo for mean and 2022 heatwave
    ax0.scatter(inp_d.index.shift(freq='12h'), inp_d.Albedo, label='avg. albedo', color='orange' )
    ax0.scatter(inp_csv_d.index.shift(freq='12h'), inp_csv_d.Albedo, label='2022 albedo', color='r')

    ax_0.set_ylabel('Temperature ($°C$)', fontsize=16)
    ax0.set_ylabel('Albedo', fontsize=16)
    ax_0.grid('both')
    ax_0.set_ylim(-2, 10)

    ax_0.set_xlim(pd.to_datetime('1900-07-15'), pd.to_datetime('1900-07-23'))
    #ax_0.set_xticklabels(ax_0.get_xticklabels(), rotation=45, ha='right')
    ax_0.set_xticklabels([])
    ax0.set_xticklabels([])
    ax_0.xaxis.set_tick_params(labelbottom=False)
    ax0.xaxis.set_tick_params(labelbottom=False)
    # ax_0.set_xticks([])
    # ax0.set_xticks([])

    myFmt = mdates.DateFormatter('%b-%d')
    # myFmt2 = mdates.DateFormatter('%b')
    ax_0.xaxis.set_major_formatter(myFmt)

    ax_0.legend(loc='upper left', bbox_to_anchor=(-.02, 1.34), ncol=2, fontsize=14)
    ax0.legend(loc='upper left', bbox_to_anchor=(-.02, 1.2), ncol=2, fontsize=14)


    ax_1.plot(inp.index, inp.SWin_Avg-inp.SWout_Avg, label='avg. SW net', color='k', linewidth=0.5, linestyle='--')
    ax_1.plot(inp_csv.index, inp_csv.SWin_Avg-inp_csv.SWout_Avg, label='2022 SW net', color='k')
    ax_1.legend(loc='upper left', bbox_to_anchor=(-.02, 1.2), ncol=2, fontsize=14)
    ax_1.grid('both')
    ax_1.set_xlim(pd.to_datetime('1900-07-15'), pd.to_datetime('1900-07-23'))
    ax_1.set_xticklabels(ax_1.get_xticklabels(), rotation=45, ha='right')

    ax_1.xaxis.set_major_formatter(myFmt)
    ax_1.set_ylabel('SW net (W m$^{-2}$)', fontsize=16)



    ds_real22 = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_heatwave2022.nc')
    print(ds_real22.variables)

    ds_mean_G_H = extractNC(ds_real, 'G')
    ds_mean_alb_H = extractNC(ds_real, 'ALBEDO')
    ds_mean_LWin_H = extractNC(ds_real, 'LWin')
    ds_mean_LWout_H = extractNC(ds_real, 'LWout')
    ds_mean_ME_H = extractNC(ds_real, 'ME')
    ds_mean_sens_H = extractNC(ds_real, 'H')
    ds_mean_LE_H = extractNC(ds_real, 'LE')
    ds_mean_Ground_H = extractNC(ds_real, 'B')


    ds_22_G_H = extractNC(ds_real22, 'G')
    ds_22_alb_H = extractNC(ds_real22, 'ALBEDO')
    ds_22_ME_H = extractNC(ds_real22, 'ME')
    ds_22_sens_H = extractNC(ds_real22, 'H')
    ds_22_LE_H = extractNC(ds_real22, 'LE')
    ds_22_LWin_H = extractNC(ds_real22, 'LWin')
    ds_22_LWout_H = extractNC(ds_real22, 'LWout')
    ds_22_Ground_H = extractNC(ds_real22, 'B')


    dfweek_major = pd.DataFrame(columns=['Q_ground', 'Q_latent', 'Q_sens', 'LWin', 'LWout', 'LWnet','SWin', 'SWout', 'SWnet','ME'], index=['2022 fluxes', 'avg. fluxes'])
    dfweek_major.loc['2022 fluxes', 'ME'] = ds_22_ME_H.sum()
    dfweek_major.loc['2022 fluxes', 'SWin'] = ds_22_G_H.sum()
    dfweek_major.loc['2022 fluxes', 'SWout'] = -1*(ds_22_G_H*ds_22_alb_H).sum()
    dfweek_major.loc['2022 fluxes', 'SWnet'] = dfweek_major.loc['2022 fluxes', 'SWin'] + dfweek_major.loc['2022 fluxes', 'SWout']
    dfweek_major.loc['2022 fluxes', 'LWin'] = ds_22_LWin_H.sum()
    dfweek_major.loc['2022 fluxes', 'LWout'] = ds_22_LWout_H.sum()
    dfweek_major.loc['2022 fluxes', 'LWnet'] = dfweek_major.loc['2022 fluxes', 'LWout'] + dfweek_major.loc['2022 fluxes', 'LWin']
    dfweek_major.loc['2022 fluxes', 'Q_sens'] = ds_22_sens_H.sum()
    dfweek_major.loc['2022 fluxes', 'Q_latent'] = ds_22_LE_H.sum()
    dfweek_major.loc['2022 fluxes', 'Q_ground'] = ds_22_Ground_H.sum()


    dfweek_major.loc['avg. fluxes', 'ME'] = ds_mean_ME_H.sum()
    dfweek_major.loc['avg. fluxes', 'SWin'] = ds_mean_G_H.sum()
    dfweek_major.loc['avg. fluxes', 'SWout'] = -1*(ds_mean_G_H*ds_mean_alb_H).sum()
    dfweek_major.loc['avg. fluxes', 'SWnet'] = dfweek_major.loc['avg. fluxes', 'SWin'] + dfweek_major.loc['avg. fluxes', 'SWout']
    dfweek_major.loc['avg. fluxes', 'LWin'] = ds_mean_LWin_H.sum()
    dfweek_major.loc['avg. fluxes', 'LWout'] = ds_mean_LWout_H.sum()
    dfweek_major.loc['avg. fluxes', 'LWnet'] = dfweek_major.loc['avg. fluxes', 'LWout'] + dfweek_major.loc['avg. fluxes', 'LWin']
    dfweek_major.loc['avg. fluxes', 'Q_sens'] = ds_mean_sens_H.sum()
    dfweek_major.loc['avg. fluxes', 'Q_latent'] = ds_mean_LE_H.sum()
    dfweek_major.loc['avg. fluxes', 'Q_ground'] = ds_mean_Ground_H.sum()

    print(dfweek_major)

    dif = dfweek_major.loc['2022 fluxes', :] - dfweek_major.loc['avg. fluxes', :]
    print(dif)
    
    percdif = 100*(dfweek_major.loc['2022 fluxes', :]) / dfweek_major.loc['avg. fluxes', :]
    print(percdif)
    # dfweek_major = dfweek_major / (8)

    # print(dfweek_major)

    dfweek_major.T.plot.barh(ax=ax_3, color=['black', 'grey'])
    ax_3.yaxis.tick_right()
    ax_3.grid('x')
    ax_3.set_xlabel('Total energy fluxes 15.-22. Jul. (W m$^{-2}$)', fontsize=16) 
    ax_3.legend(fontsize=14)


    ax_0.text(pd.to_datetime('1900-07-15 06:00'), 7, 'a', 
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    ax_1.text(pd.to_datetime('1900-07-15 06:00'), 600, 'b', 
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    ax_3.text(-51000, 0.4, 'c', 
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    fig.savefig('figs/heatwave_vs_meanclim.png', bbox_inches='tight', dpi=200)

    # plt.show()



def extractNC(ds, par):
    ds_par = ds[par].squeeze().to_pandas()
    ds_par.index = ds_par.index.map(lambda t: t.replace(year=1900))
    ds_par = ds_par.loc[pd.to_datetime('1900-07-15'): pd.to_datetime('1900-07-23')]
    # ds_par_H = ds_par.groupby(ds_par.index.hour).mean()
    return(ds_par)






def fig3():
    minalb = 0.1
    maxalb = 0.9
    step=0.1

    df = pd.DataFrame(columns=np.arange(minalb, maxalb, step))


    for alb in np.arange(minalb, maxalb, step):
        fn = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_heatwave2022_'+str(alb.round(decimals=2))+'.nc'
        ds = xr.open_dataset(fn)

        ds_d = ds.surfMB.squeeze().to_pandas()
        ds_d.index = pd.to_datetime(ds_d.index)
        # print(ds_d)
        ds_d = ds_d#.resample('D').sum()

        df[alb] = ds_d
    print(df)
    df = df
    df_d=df#.resample('D').sum()

    ds_real = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_heatwave2022.nc')
    ds_d_real = ds_real.surfMB.squeeze().to_pandas()
    ds_d_real.index = pd.to_datetime(ds_d_real.index)
    # ds_d_real = ds_d_real.resample('D').sum()

    df_real = pd.DataFrame(columns=['surfMB', 'albedo'])
    df_real['surfMB'] = ds_d_real
    df_real['albedo'] = ds_real.ALBEDO.squeeze().to_pandas().resample('D').mean()
    df_real['LE'] = ds_real.LE.squeeze().to_pandas()#.resample('D').mean()
    print(df_real)
    print(df.resample('D').sum()*1000)
    print(df_real.resample('D').sum()*1000)
    print(df.sum()*1000)
    print(df_real.sum()*1000)
    #print(df_real.loc[df_real.index.month==8])


    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax = ax.flatten()
    clrs = cm.plasma(np.linspace(0, 1, len(df.columns)))
    # ax.plot(ds.time, ds.T2.squeeze())
    inp_csv = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_heatwave2022.csv', parse_dates=True, index_col=0)
    ax[0].plot(inp_csv.index, inp_csv.Tair_Avg, label='Air temperature', color='k', linestyle='--')
    ax[0].plot(inp_csv.index, inp_csv.SWin_Avg/100, label='SWin', color='k')
    ax[0].plot(inp_csv.index, inp_csv.LWin_Cor/100, label='LWin', color='grey')
    # ax[0].plot(df_real.index, df_real['LE']/100, label='B', color='blue')
    ax[0].grid('both')
    ax0 = ax[0].twinx()
    #ax0.plot(inp_csv.index, inp_csv.Surf, label='surface height, AWS', color='k')
    ax0.scatter(df_real.index.shift(freq='12h'), df_real.albedo, label='albedo, AWS', color='r')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.4, 1.28), ncol=3, fontsize=14)
    ax0.legend(loc='upper center', bbox_to_anchor=(0.88, 1.28), fontsize=14)

    ax[0].set_ylabel('$°C$ ; $100*W/m^2$', fontsize=14)
    ax0.set_ylabel('Albedo', fontsize=14)


    ax[1].scatter(df_real.index, df_real.surfMB*1000, label='albedo as measured', color='grey', zorder=100, s=12)
    # for i, c in enumerate(np.arange(0.1, 0.95, 0.5)):
    for i, c in enumerate(df.columns):
        ax[1].plot(df_d.index, df_d[c]*1000, label='albedo: '+str(round(c, 2)), color=clrs[i], linewidth=0.8)#, s=4)
    ax[1].legend(loc='center left', bbox_to_anchor=(1.01, 0.0), fontsize=14)
    ax[1].grid('both')


    ax[2].scatter(df_real.index, df_real.surfMB.cumsum()*1000, label='albedo as measured', color='grey', s=10)
    # for i, c in enumerate(np.arange(0.1, 0.95, 0.5)):
    for i, c in enumerate(df.columns):
        ax[2].plot(df_d.index, df_d[c].cumsum()*1000, label='albedo: '+str(round(c, 2)), color=clrs[i], linewidth=1)#, color='pink')
    # ax[2].legend(loc='lower left')
    ax[2].grid('both')

    ax[1].set_ylabel('SMB. (mm w.e.)', fontsize=16)
    # ax[1].set_title('Surface mass balance')
    # ax[2].set_ylabel('mm w.e.')
    ax[2].set_ylabel('Cumul. SMB (mm w.e.)', fontsize=16)
    ax[2].set_xlim(pd.to_datetime('2022-07-15 00:00'), pd.to_datetime('2022-07-23 00:00'))
    ax[2].set_ylim(-500, 100)

    for a in ax:
        a.tick_params(axis='x', labelsize=16, rotation=45)
        a.tick_params(axis='y', labelsize=16)
    ax0.tick_params(axis='y', labelsize=16)


    ax[0].text(pd.to_datetime('2022-07-15 06:00'), 9, 'a', 
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    ax[1].text(pd.to_datetime('2022-07-15 06:00'), -6.5, 'b', 
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    ax[2].text(pd.to_datetime('2022-07-15 06:00'), -250, 'c', 
                 bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))



    surfheight = pd.read_csv('out/cleaned_surface_hourly.csv', index_col='TIMESTAMP', parse_dates=True)
    
    surfheight=surfheight.loc['2022-07-15 00:00': '2022-07-23 00:00']
    print(surfheight)
    # ax[3].plot(surfheight.index, surfheight.Surf_step1*1000, linewidth=1, color='k',zorder=200)
    # ax[3].set_ylim(-400, 100)
    snow = surfheight.loc[surfheight.Surf_step1>0]
    ice = surfheight.loc[surfheight.Surf_step1<=0]


    snow_we = (snow.diff()*400).cumsum()
    ice_we = (ice.diff()*900).cumsum()

    print(surfheight.Surf_step1.diff().cumsum())

    print((ice.diff()*900).resample('D').sum())

   
    # ax[3].fill_between(snow.index, snow.Surf_step1*1000, color='lightgrey', zorder=100, label='snow ('+str(snow_we.values[-1][0].round(decimals=0).astype(int))+' mm w.e.)')
    # ax[3].fill_between(surfheight.index, surfheight.Surf_step1*1000, -600, color='lightblue', label='ice ('+str(ice_we.values[-1][0].round(decimals=0).astype(int))+') mm w.e.)')
    # ax[3].grid('both')
    # ax[3].legend()
    # ax[3].set_ylabel('Surface height (mm)')

    # ax[3].text(pd.to_datetime('2022-07-15 06:00'), -350, 'd', 
    #              bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    fig.savefig('figs/model_Heatwave2022.png', bbox_inches='tight', dpi=200)




# def fig3_rev():
#     minalb = 0.1
#     maxalb = 0.9
#     step=0.1

#     df = pd.DataFrame(columns=np.arange(minalb, maxalb, step))

#     df_ME = pd.DataFrame(columns=np.arange(minalb, maxalb, step))


#     for alb in np.arange(minalb, maxalb, step):
#         fn = '/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_heatwave2022_'+str(alb.round(decimals=2))+'.nc'
#         ds = xr.open_dataset(fn)

#         ds_d = ds.surfMB.squeeze().to_pandas()
#         ds_d.index = pd.to_datetime(ds_d.index)
#         ds_d = ds_d#.resample('D').sum()
#         df[alb] = ds_d


#         ds_me = ds.ME.squeeze().to_pandas()
#         ds_me.index = pd.to_datetime(ds_me.index)
#         df_ME[alb] = ds_me

#     # print(df)
#     # print(df_ME)
#     # stop
#     df = df
#     df_d=df#.resample('D').sum()

#     ds_real = xr.open_dataset('/Users/leahartl/Desktop/WSS/AWS2COSIPY/cospiy_July2024/data/output/AWS_WSS_4cosipy_heatwave2022.nc')
#     ds_d_real = ds_real.surfMB.squeeze().to_pandas()
#     ds_d_real.index = pd.to_datetime(ds_d_real.index)
#     # ds_d_real = ds_d_real.resample('D').sum()

#     print(ds_real.variables)

#     # ds_real_SWin = ds_real.G.squeeze().to_pandas()
#     # ds_real_LWin = ds_real.LWin.squeeze().to_pandas()
#     # ds_real_LWout = ds_real.LWout.squeeze().to_pandas()
#     # ds_real_sens = ds_real.H.squeeze().to_pandas()
#     # ds_real_late = ds_real.LE.squeeze().to_pandas()
#     # ds_real_ME = ds_real.ME.squeeze().to_pandas()

#     # print(ds_real.ME)


#     df_real = pd.DataFrame(columns=['surfMB', 'albedo'])
#     df_real['surfMB'] = ds_d_real
#     df_real['albedo'] = ds_real.ALBEDO.squeeze().to_pandas().resample('D').mean()
#     df_real['LE'] = ds_real.LE.squeeze().to_pandas()#.resample('D').mean()
#     df_real['SWin'] = ds_real.G.squeeze().to_pandas()
#     df_real['H'] = ds_real.H.squeeze().to_pandas()
#     df_real['ME'] = ds_real.ME.squeeze().to_pandas()
#     df_real['Ground'] = ds_real.B.squeeze().to_pandas()
#     print(df_real)
#     print(df.resample('D').sum()*1000)
#     print(df_real.resample('D').sum()*1000)
#     print(df.sum()*1000)
#     print(df_real.sum()*1000)
#     #print(df_real.loc[df_real.index.month==8])


#     fig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
#     ax = ax.flatten()
#     clrs = cm.plasma(np.linspace(0, 1, len(df.columns)))
#     # ax.plot(ds.time, ds.T2.squeeze())
#     inp_csv = pd.read_csv('/Users/leahartl/Desktop/WSS/AWS2COSIPY/AWS/AWS_WSS_4cosipy_heatwave2022.csv', parse_dates=True, index_col=0)
#     ax[0].plot(inp_csv.index, inp_csv.Tair_Avg, label='Air temperature', color='k', linestyle='--')
#     ax[0].plot(inp_csv.index, inp_csv.SWin_Avg/100, label='SWin', color='k')
#     ax[0].plot(inp_csv.index, inp_csv.LWin_Cor/100, label='LWin', color='grey')
#     # ax[0].plot(df_real.index, df_real['LE']/100, label='B', color='blue')
#     ax[0].grid('both')
#     ax0 = ax[0].twinx()
#     #ax0.plot(inp_csv.index, inp_csv.Surf, label='surface height, AWS', color='k')
#     ax0.scatter(df_real.index.shift(freq='12h'), df_real.albedo, label='albedo, AWS', color='r')
#     ax[0].legend(loc='upper center', bbox_to_anchor=(0.4, 1.28), ncol=3)
#     ax0.legend(loc='upper center', bbox_to_anchor=(0.8, 1.28))

#     ax[0].set_ylabel('$°C$ ; $100*W/m^2$', fontsize=16)
#     ax0.set_ylabel('Albedo', fontsize=16)




#     ax[1].scatter(df_real.index, df_real.surfMB*1000, label='albedo as measured', color='grey', zorder=100, s=12)
#     # for i, c in enumerate(np.arange(0.1, 0.95, 0.5)):
#     for i, c in enumerate(df.columns):
#         ax[1].plot(df_d.index, df_d[c]*1000, label='albedo: '+str(round(c, 2)), color=clrs[i], linewidth=0.8)#, s=4)
#     ax[1].legend(loc='center left', bbox_to_anchor=(1.01, 0.0), fontsize=14)
#     ax[1].grid('both')


#     ax[2].scatter(df_real.index, df_real.surfMB.cumsum()*1000, label='albedo as measured', color='grey', s=10)
#     # for i, c in enumerate(np.arange(0.1, 0.95, 0.5)):
#     for i, c in enumerate(df.columns):
#         ax[2].plot(df_d.index, df_d[c].cumsum()*1000, label='albedo: '+str(round(c, 2)), color=clrs[i], linewidth=1)#, color='pink')
#     # ax[2].legend(loc='lower left')
#     ax[2].grid('both')

#     ax[1].set_ylabel('SMB. (mm w.e.)', fontsize=16)
#     # ax[1].set_title('Surface mass balance')
#     # ax[2].set_ylabel('mm w.e.')
#     ax[2].set_ylabel('Cumul. SMB (mm w.e.)',  fontsize=16)
#     ax[2].set_xlim(pd.to_datetime('2022-07-15 00:00'), pd.to_datetime('2022-07-23 00:00'))



#     ax[3].scatter(df_real.index, df_real.LE, label='LE', color='grey', zorder=100, s=12)
#     ax[3].scatter(df_real.index, df_real.H, label='H', color='k', zorder=100, s=12)
#     ax[3].scatter(df_real.index, df_real.ME, label='SW', color='red', zorder=100, s=12)
#     ax[3].set_xlim(pd.to_datetime('2022-07-15 00:00'), pd.to_datetime('2022-07-23 00:00'))


#     for a in ax:
#         a.tick_params(axis='x', labelsize=16, rotation=45)
#         a.tick_params(axis='y', labelsize=16)
#     ax0.tick_params(axis='y', labelsize=16)

#     ax[0].text(pd.to_datetime('2022-07-15 06:00'), 9, 'a', 
#                  bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

#     ax[1].text(pd.to_datetime('2022-07-15 06:00'), -6.5, 'b', 
#                  bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

#     ax[2].text(pd.to_datetime('2022-07-15 06:00'), -150, 'c', 
#                  bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))



#     fig.savefig('figs/model_Heatwave2022_2.png', bbox_inches='tight', dpi=200)




def fig4():
    dat = pd.read_csv('out/averagemelt_model2025.csv')
    dat.index = dat.time
    
    dat = dat[['0.1', '0.2', '0.3', '0.4', '0.6']]
    dat = dat.loc[dat.index<'1900-09-29']
    print(dat)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['DimGrey', 'DarkGrey', 'Silver', 'LightGrey', 'WhiteSmoke']
    dat.plot.bar(grid=True, ax = ax, color=colors, edgecolor='k')
    ax.set_ylabel('Daily SMB (mm w.e.)', fontsize=16)
    ax.set_xlabel('Time period', fontsize=16)
    ax.legend(title='Albedo', title_fontsize='large', fontsize=14)

    ticks = ['Jul. 1 - 15', 'Jul. 16 - 30', 'Jul. 31 - Aug. 14', 'Aug. 15 - 29',
             'Aug. 30 - Sep. 13', 'Sep. 14 - 29']
    ax.set_xticklabels(ticks, rotation=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    fig.savefig('figs/modeloutput_average.png', dpi=200, bbox_inches='tight')


# paper fig: average summer climatology:
# fig2()

#fig2_forposter()

# figure comparing average conditions vs 2022 during July subperiod
Avg_vs_2022()

# sup fig: model evalulation
# fig1_1(inp_csv, ds)

# paper fig: 2022 heat wave
# fig3()


# paper fig: bar plot SMB for average conditions (15 day averages)
# fig4()


    
plt.show()

stop

