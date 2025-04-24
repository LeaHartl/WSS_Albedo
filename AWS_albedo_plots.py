# ! /usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
# import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as seaborn
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from windrose import WindroseAxes, plot_windrose
import seaborn as sns
import scipy.stats as stats
from scipy.stats import circmean
import geopandas as gpd
from sklearn.metrics import mean_squared_error
from datetime import datetime


# load AWS data
file = '/Users/leahartl/Desktop/WSS/AWS_wss/out/AWS_WSS_proc2025.csv'
data = pd.read_csv(file, parse_dates=True, skiprows=[0, 2, 3])
data.index = pd.to_datetime(data['TIMESTAMP'])

# load file with POA (plane of array) direct and diffuse components computed in AWS_albedo_corr.py
file2 = '/Users/leahartl/Desktop/WSS/WSS_Albedo/out/albedo_cor2025.csv'
data2 = pd.read_csv(file2, parse_dates=True, index_col=0)

# plane of arra (=estimate of incoming SW on tilted plane)
data2['POA'] = data2['poa_direct']+data2['poa_sky_diffuse']
# albedo estimate with simple correction for the tilt
data2['albedo_Cor'] = data2['SWout_Avg']/data2['POA']
# difference between SW in at AWS sensor and SW in for tilted plane
data2['difrad'] = data2['SWin_Avg'] - data2['POA']
# albedo as ratio of SW out to SW in at sensor
data2['albedo'] = data2['SWout_Avg']/data2['SWin_Avg']





# # list of flags
# flags = ['Batt_Min_flag', 'Tair_Avg_flag', 'Hum_Avg_flag', 'Press_Avg_flag', 'Wdir_flag', 'Wspeed_flag',
#          'Wspeed_Max_flag', 'SWin_Avg_flag', 'SWout_Avg_flag', 'LWin_Cor_flag', 'LWout_Cor_flag',
#          'Snow_flag']

# set data with flags to np.nan before passing to plotting functions:
# for flag in flags:
#     #print(data.loc[(data[flag] != 0), flag[:-5]])
#     data2.loc[(data[flag] != 0), flag[:-5]] = np.nan


# data2['albedo'] = data2.SWout_Avg/data2.SWin_Avg
#data2.loc[data['Wdir_flag']!=0, 'albedo'] = np.nan

# bothWinter[maskWinterSWinout & maskSolarNoonWinter]

# make new DF with most relevant parameters:
dfAlbedo = data2[['albedo', 'albedo_Cor', 'GHI', 'POA', 'SWin_Avg', 'SWout_Avg']]
# add day of year
dfAlbedo['doy'] = dfAlbedo.index.dayofyear

# maskNoon = (dfAlbedo.index.hour >= 10) & (dfAlbedo.index.hour <= 12)
# maskSWinout= (dfAlbedo.SWout_Avg < dfAlbedo.SWin_Avg)

# set albedo values below 0 and above 1 to nan
dfAlbedo['albedo'].loc[dfAlbedo.albedo < 0] = np.nan
dfAlbedo['albedo'].loc[dfAlbedo.albedo > 1] = np.nan
# do the same for "corrected albedo"
dfAlbedo['albedo_Cor'].loc[dfAlbedo.albedo_Cor < 0] = np.nan
dfAlbedo['albedo_Cor'].loc[dfAlbedo.albedo_Cor > 1] = np.nan

# start steps to generate a DF of daily mean values (drop nans first)
df_mean1 = dfAlbedo[['albedo', 'albedo_Cor', 'POA', 'SWin_Avg', 'SWout_Avg']].dropna(subset=['albedo', 'albedo_Cor'])
df_mean1['cor_dfif'] = df_mean1['albedo_Cor']-df_mean1['albedo']

# At this point, df_mean contains:
# albedo: daily mean albedo computed as ratio of SWout to SWin_Avg, then averaged for each day
# albedo_cor :albedo estimate corrected for tilted surface
df_mean = df_mean1.resample('d').mean()

# now add albedo computed as daily sum of outgoing / dail ysum of incoming SW:
# first remove values below 2 W/m2 to account for sensor issues (based on comments in the sensor manual):
data['SWout_Avg'].loc[data['SWout_Avg'] < 2] = np.nan
data['SWin_Avg'].loc[data['SWout_Avg'] < 2] = np.nan
# albedo sum: daily SWout / daily SWin
df_mean['albedo_sum'] = data.resample('d').sum()['SWout_Avg'] / data.resample('d').sum()['SWin_Avg']
# set values below 0 and above 1 to nan
df_mean['albedo_sum'].loc[df_mean.albedo_sum < 0] = np.nan
df_mean['albedo_sum'].loc[df_mean.albedo_sum > 1] = np.nan
# albedo_Cor_sum: ratio of daily sum but with tilted incoming estimate instead of SWin measured at sensor
df_mean['albedo_Cor_sum'] = data2.resample('d').sum()['SWout_Avg'] / data2.resample('d').sum()['POA']


# WRITE CSV (optional)
# print(df_mean.head())
#df_mean.to_csv('out/albedo_cor_daily2025.csv')


# make a DF of monthly median values from the DF with the daily values:
monthly = df_mean.groupby(df_mean.index.month).median()

## -------- This section deals with time periods with faulty data and removes them from the DF
# these dates have riming/faulty values in the AWS data:
skip2019_Jan = ['2019-01-08', '2019-01-18']
skip2019_Aug = ['2019-08-25', '2019-10-01']
# skip2020_Sep = ['2020-09-30', '2020-09-30']
skip2020_Oct = ['2020-10-13', '2020-10-13']
skip2023_Nov = ['2023-10-27', '2023-11-07']

skip2018Nov = ['2018-11-12', '2018-11-29']
skip2023May = ['2023-05-15', '2023-06-09']

skip2024June =['2024-06-09', '2024-06-21']
skip2024August =['2024-07-29', '2024-08-23']

# combine to make a list
skiplist = [skip2018Nov, skip2019_Jan, skip2019_Aug, skip2020_Oct, skip2023_Nov, skip2023May, skip2024June, skip2024August]

# loop through the list and set the data to nan:
for s in skiplist:
    df_mean.loc[(df_mean.index >= pd.to_datetime(s[0])) & (df_mean.index <= pd.to_datetime(s[1])), 'albedo'] = np.nan
    df_mean.loc[(df_mean.index >= pd.to_datetime(s[0])) & (df_mean.index <= pd.to_datetime(s[1])), 'albedo_Cor'] = np.nan
    df_mean.loc[(df_mean.index >= pd.to_datetime(s[0])) & (df_mean.index <= pd.to_datetime(s[1])), 'albedo_sum'] = np.nan


## -------- This section aplies thresholds to set unrealistic values to nan (applies to low albedo
# in winter, most likely from riming issues not caught in the previous step)

# set to nan if albedo is less than 0.4 between November and May.
df_mean['albedo'].loc[((df_mean.index.month>=11) | (df_mean.index.month<=5)) & (df_mean['albedo']<0.4)] = np.nan
df_mean['albedo_Cor'].loc[((df_mean.index.month>=11) | (df_mean.index.month<=5)) & (df_mean.albedo_Cor<0.4)] = np.nan
df_mean['albedo_sum'].loc[((df_mean.index.month>=11) | (df_mean.index.month<=5)) & (df_mean.albedo_sum<0.4)] = np.nan

# set to nan if albedo is less than 0.6 between December and March.
df_mean['albedo'].loc[((df_mean.index.month>=12) | (df_mean.index.month<=3)) & (df_mean.albedo<0.6)] = np.nan
df_mean['albedo_Cor'].loc[((df_mean.index.month>=12) | (df_mean.index.month<=3)) & (df_mean.albedo_Cor<0.6)] = np.nan
df_mean['albedo_sum'].loc[((df_mean.index.month>=12) | (df_mean.index.month<=3)) & (df_mean.albedo_sum<0.6)] = np.nan
df_mean['doy'] = df_mean.index.dayofyear


## This is a list of dates with faulty data in the S2 time series, this is passed to some of the
# plotting functions later to exclude these days.
#UPDATE FOR 2024
baddates_sat = pd.to_datetime(['2018-08-02',
                             '2018-09-29', '2018-11-25','2018-12-08',
                             '2019-01-27', '2019-04-02','2019-04-04', '2019-08-22', '2019-08-30','2019-12-15',
                             '2020-02-06', '2020-11-19', '2020-12-04', 
                             '2021-01-31', '2021-12-14', '2021-11-19',
                             '2022-01-03', '2022-02-07', '2022-08-26', '2022-10-20', '2022-12-27', '2022-12-29', 
                             '2023-01-13', '2023-02-20', '2023-12-29',
                             '2024-01-08', '2024-01-28', '2024-02-15', '2024-12-11'])





## ----- HERE ARE THE PLOTTING FUNCTIONS ----- ##



def albedoAllYears(df_mean, what):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    # ax = ax.flatten()

    # clrs = cm.tab20(range(7))
    clrs = cm.Set2(range(7))
    # axins = ax.inset_axes([0.72, -0.30, 0.4, 0.5])
    for i, y in enumerate([2018, 2019, 2020, 2021, 2022, 2023, 2024]):
        temp = df_mean.loc[df_mean.index.year == y]
        # print(temp['albedo'].head(20))
        ax.plot(temp['doy'].values, temp[what].rolling('5d', center=True, closed='both').mean(), c=clrs[i], label=str(y))
        ax.scatter(temp['doy'], temp[what], color=clrs[i], s=2)

        # if y >= 2022:
        # axins.plot(temp['doy'].values, temp[what].rolling('5d', center=True, closed='both').mean(), c=clrs[i], label=str(y))
        # axins.scatter(temp['doy'], temp[what], color=clrs[i], s=2)


    ax.legend(loc='lower left', )
    ax.grid('both')
    myFmt = mdates.DateFormatter('%b-%d')
    ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.set_xlim(0, 365)
    ax.set_ylim(0.1, 1)
    ax.set_ylabel('Albedo', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=18, rotation=45)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_major_formatter(myFmt)

    # axins.set_xlim(170, 258)
    # axins.set_ylim(0.1, 1)
    # axins.set_xticks([181, 212, 243])
    # axins.xaxis.set_major_formatter(myFmt)
    # axins.yaxis.tick_right()
    # ax.indicate_inset_zoom(axins, edgecolor="black")
    # axins.grid('both')

    fig.savefig('figs/timeseries_allinone_'+what+'.png', bbox_inches='tight', dpi=300, transparent = 'True')


def albedoMesh(ds):
    df_mean = ds
    df_mean['day'] = df_mean.index.day
    df_mean['month'] = df_mean.index.month
    df_mean['year'] = df_mean.index.year
    df_mean['doy'] = df_mean.index.dayofyear

    df_mean = df_mean.loc[(df_mean['year'] > 2017) & (df_mean['year'] < 2024)]

    # unstacked_raw = df_mean.set_index(['year', 'doy']).albedo.unstack(-2)
    unstacked = df_mean.set_index(['year', 'doy']).albedo_sum.unstack(-2)

    cmap = cm.get_cmap('plasma', int(50/5))
    norm = colors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    levels = np.arange(0, 50, 5)
    x = unstacked.columns.values
    y = unstacked.index.values
    z = unstacked.values
    xi, yi = np.meshgrid(x, y)
    cs = ax.pcolormesh(xi, yi, z, norm=norm, cmap=cmap)
    cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
    cbar = fig.colorbar(cs, cax=cax)

    plt.subplots_adjust(right=0.85)

    xlim = [0, 366]
    ax.set_ylabel('Day of year')
    myFmt = mdates.DateFormatter('%b-%d')
    ax.set_yticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax.yaxis.set_major_formatter(myFmt)
    ax.set_xlabel('Year')
    cbar.set_label('Albedo')
    fig.savefig('figs/meshgridAlbedo.png')


def albedoAllYears_CorSubplots(df, df_mean):

    mask = (df.index.hour >= 10) & (df.index.hour <= 12)
    df = df[mask]


    df['doy'] = df.index.dayofyear

    fig, ax = plt.subplots(2, 2, figsize=(12, 5))
    ax = ax.flatten()

    clrs = cm.tab20(range(6))
    # axins = ax.inset_axes([0.72, -0.30, 0.4, 0.5])
    for i, y in enumerate([2018, 2019, 2020, 2021, 2022, 2023]):
        temp = df.loc[df.index.year == y]
        temp2 = df_mean.loc[df_mean.index.year == y]

        ax[0].scatter(temp2['albedo_Cor'], temp2['albedo'], color=clrs[i])
        ax[1].scatter(temp['doy'], temp['albedo_Cor']-temp['albedo'], color=clrs[i], s=2)
        ax[2].scatter(temp2['doy'], temp2['difrad'], color=clrs[i], s=2)
        ax[3].scatter(temp2['doy'], temp2['albedo_Cor']-temp2['albedo'], color=clrs[i], s=2)

    ax[0].legend(loc='lower left')
    ax[0].grid('both')
    myFmt = mdates.DateFormatter('%b-%d')
    # ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    # ax.set_xlim(0, 365)
    ax[0].set_ylim(0, 1.2)
    ax[0].set_xlim(0, 1.2)

    fig.savefig('figs/albedo_cor_subplots.png', bbox_inches='tight', dpi=300)


def Scatterplot_Cor(data2, df_daily):
    data2['doy'] = data2.index.dayofyear
    # set bad values to nan
    data2['albedo'].loc[data2.albedo < 0] = np.nan
    data2['albedo'].loc[data2.albedo > 1] = np.nan
    # do the same for "corrected albedo"
    data2['albedo_Cor'].loc[data2.albedo_Cor < 0] = np.nan
    data2['albedo_Cor'].loc[data2.albedo_Cor > 1] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # ax = ax.flatten()

    clrs = cm.tab20(range(6))

    df_daily['albedo_Cor_daily'] = data2['SWout_Avg'].resample('D').sum()/data2['POA'].resample('D').sum()
    difcor = df_daily['albedo_Cor_daily']-df_daily['albedo_sum']
    monthlymedian = difcor.groupby(difcor.index.month).median()
    # print(monthlymedian)

    ax.scatter(df_daily['doy'], difcor.values, color='k', s=1)
    ax.set_title('Difference between tilt corrected albedo and measured albedo')
    ax.grid('both')
    ax.set_xlabel('Day of year')

    fig.savefig('figs/albedo_cor.png', bbox_inches='tight', dpi=300)


def dailyCor(both):
    mask = (both.SWin_Avg <= 2)
    both[mask] = np.nan

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    ax = ax.flatten()
    ax[0].plot(both.index, both.SWin_Avg, label='AWS in', color='k')
    ax[0].plot(both.index, both.SWout_Avg, label='AWS out', color='k', linestyle='--')

    ax[0].plot(both.index, both['POA'], label='POA', color='red')
    ax[0].plot(both.index, both['GHI'], label='GHI', color='orange')
    ax[0].set_ylabel('Irradiance ($W/m^2$)')
    # ax1.legend()
    ax[0].legend()

    ax[1].plot(both.index, both['difrad'])
    ax[1].set_ylabel('SWin-POA ($W/m^2$)')

    ax[2].set_ylim(0.2,0.75)

    mask = (both.index.hour >= 10) & (both.index.hour <= 12)

    masked = both[mask]
    print(masked)

    ax[2].plot(both.index, both['albedo'], label='albedo, raw')
    ax[2].plot(both.index, both['albedo_Cor'], label='albedo, cor')

    ax[2].set_ylabel('Albedo')

    dailyalbedo = masked.resample('d').mean()

    ax[2].scatter(dailyalbedo.index.shift(freq='12H'), dailyalbedo['albedo'], label='daily albedo, raw')
    ax[2].scatter(dailyalbedo.index.shift(freq='12H'), dailyalbedo['albedo_Cor'], label='daily albedo, cor')
    ax[2].legend()

    for d in both.index.dayofyear.unique():
        x1 = both.loc[(both.index.dayofyear==d) & (both.index.hour==10)].index[0]
        x2 = both.loc[(both.index.dayofyear==d) & (both.index.hour==13)].index[0]
        print(x1, x2)
        ax[2].axvspan(x1, x2, alpha=0.5, color='grey')

    for a in ax:
        a.grid('both')


# shows a week of data in summer and winter to exemplify the effect of the surface tilt.
def dailyCor_SummerWinter(bothWinter, bothSummer):
    # mask night time values:
    mask = (bothWinter.SWin_Avg <= 2)
    mask1 = (bothSummer.SWin_Avg <= 2)
    bothWinter[mask] = np.nan
    bothSummer[mask1] = np.nan

    # ------------

    dailyalbedoWinter = bothWinter['SWout_Avg'].resample('d').sum()/bothWinter['SWin_Avg'].resample('d').sum()
    dailyalbedoSummer = bothSummer['SWout_Avg'].resample('d').sum()/bothSummer['SWin_Avg'].resample('d').sum()

    dailyalbedoWinter_Cor = bothWinter['SWout_Avg'].resample('d').sum()/bothWinter['POA'].resample('d').sum()
    dailyalbedoSummer_Cor = bothSummer['SWout_Avg'].resample('d').sum()/bothSummer['POA'].resample('d').sum()


    fig, ax = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    ax = ax.flatten()
    ax[0].plot(bothWinter.index, bothWinter.SWin_Avg, label='AWS in', color='k')
    ax[0].plot(bothWinter.index, bothWinter.SWout_Avg, label='AWS out', color='k', linestyle='--')
    ax[0].plot(bothWinter.index, both['GHI'], label='GHI, horizontal', color='orange', linewidth=0.5)
    ax[0].plot(bothWinter.index, both['POA'], label='Tilted', color='red', linewidth=0.5)
    
    ax[0].grid('both')
    ax[0].set_ylabel('Shortwave radiation (W/m$^2$)')
    ax[0].legend()

    ax0 = ax[0].twinx()
    ax0.scatter(dailyalbedoWinter.index.shift(freq='11.15H'), dailyalbedoWinter, label='daily albedo as measured', color='skyblue')
    ax0.scatter(dailyalbedoWinter_Cor.index.shift(freq='11.15H'), dailyalbedoWinter_Cor, label='daily albedo with tilt correction', color='orange')
 
    ax0.legend(ncols=2, loc='upper left', bbox_to_anchor=(0, 1.2))
    ax0.set_ylim(0.5, 1)
    ax0.set_ylabel('Albedo')
    ax[0].set_xlim(dailyalbedoWinter.index[0], dailyalbedoWinter.index[-1]+pd.Timedelta('1d'))

    ax[1].plot(bothSummer.index, bothSummer.SWin_Avg, label='AWS up', color='k')
    ax[1].plot(bothSummer.index, bothSummer.SWout_Avg, label='AWS down', color='k', linestyle='--')

    ax[1].plot(bothSummer.index, bothSummer['POA'], label='POA', color='red', linewidth=0.5)
    ax[1].plot(bothSummer.index, bothSummer['GHI'], label='GHI', color='orange', linewidth=0.5)
    ax[1].grid('both')

    ax1 = ax[1].twinx()
    ax1.scatter(dailyalbedoSummer.index.shift(freq='11.15H'), dailyalbedoSummer, label='daily albedo, raw', color='skyblue')
    ax1.scatter(dailyalbedoSummer_Cor.index.shift(freq='11.15H'), dailyalbedoSummer_Cor, label='daily albedo, cor.', color='orange')

    ax1.set_ylim(0.5, 1)
    ax[1].set_xlim(dailyalbedoSummer.index[0], dailyalbedoSummer.index[-1]+pd.Timedelta('1d'))
    ax[1].set_ylabel('Shortwave radiation ($W/m^2$)')
    ax1.set_ylabel('Albedo')

    ax0.text(dailyalbedoWinter.index[0], 0.7,'a', fontsize=12, bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))
    ax1.text(dailyalbedoSummer.index[0], 0.7,'b', fontsize=12, bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    fig.savefig('figs/albedo_cor_summerwinter.png', bbox_inches='tight', dpi=300)

# scatterplot comparing AWS and S2, color coded by month.
# prints mean bias and RMSE

def timeseriesplot_simple(df, st, nogood, what):

    # df = ds#.resample('H').mean()

    dfpoint = pd.read_csv('/Users/leahartl/Desktop/WSS/output_GEE/extractpoint_AWS_allyear_5.csv', parse_dates=True)
    dfpoint.index = pd.to_datetime(dfpoint.dttime)
    dfpoint['QC'] = 0

    for no in nogood: 
        dfpoint.loc[(dfpoint.index.day==no.day) & (dfpoint.index.month==no.month) & (dfpoint.index.year==no.year), 'liangAlbedo'] = np.nan
        dfpoint.loc[(dfpoint.index.day==no.day) & (dfpoint.index.month==no.month) & (dfpoint.index.year==no.year), 'QC'] = 1

    B1 = dfpoint[['liangAlbedo', 'name', 'dttime', 'QC']].copy()
    B1.index = B1.index.date
    merged = pd.merge(df, B1, left_index=True, right_index=True)
    
    # print quantiles:
    print(merged[['liangAlbedo', 'albedo_sum']].quantile([0.01, .05, 0.1]))

    # print low albedo subset:
    print(merged.loc[merged.liangAlbedo<0.4])

    # print manually removed values:
    print(merged.loc[merged.QC==1])
    print('percentage removed: ', 100* merged.loc[merged.QC==1].shape[0] / merged.shape[0])


    # set colors to indicate winter (blue-ish) and summer (red-ish)
    clrs = ['skyblue', 'cornflowerblue', 'royalblue', 'blue', 'teal', 'darkturquoise', 'silver', 'slategrey', 'violet', 'magenta', 'darkviolet', 'crimson']

    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    ax = ax.flatten()

    # print bias and rmse
    biasAll = merged.liangAlbedo - merged[what]
    print('meanbias all data:', biasAll.abs().mean())

    dif2 = biasAll**2
    rms = np.sqrt(dif2.mean())
    print('rms all data:', rms)


    # # print bias and rmse for very low albedo values
    v_low = merged.loc[merged[what]<0.2]
    biasAll_vlow = v_low.liangAlbedo - v_low[what]
    print('meanbias vlow:', biasAll_vlow.abs().mean())

    dif2_vlow = biasAll_vlow**2
    rms_vlow = np.sqrt(dif2_vlow.mean())
    print('rms vlow:', rms_vlow)

    for i, m in enumerate([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        sub = merged.loc[merged.index.month == m]
        lb = datetime.datetime(2000, m, 1)
        ax[0].scatter(sub.liangAlbedo, sub[what], c=clrs[i], label=lb.strftime('%b'))

        # errorbars: 14% of AWS albedo
        ax[0].errorbar(sub.liangAlbedo, sub[what], yerr=sub[what]*0.14, linestyle="None", color='k', linewidth=0.4, zorder=0)
        
        bias = sub.liangAlbedo - sub[what]
        ax[1].scatter(bias, sub[what], c=clrs[i], label=m)

    ax[0].grid('both')
    ax[0].plot([0, 1], [0, 1], transform=ax[0].transAxes, color='k', zorder=1, linewidth=0.5)
    ax[1].grid('both')

    ax[0].set_xlabel('S2 derived albedo', fontsize=12)
    ax[1].set_xlabel('bias ( S2 albedo - AWS albedo)', fontsize=12)
    ax[0].set_ylabel('AWS albedo', fontsize=12)

    ax[0].set_ylim([0, 1])
    ax[0].set_xlim([0, 1])
    ax[0].legend()

    ax[0].annotate('a', xy=(0.9, 0.1), xycoords='axes fraction', color='k', fontweight='bold', bbox=dict(facecolor='none', edgecolor='k'))
    ax[1].annotate('b', xy=(0.9, 0.1), xycoords='axes fraction', color='k', fontweight='bold', bbox=dict(facecolor='none', edgecolor='k'))
    

    fig.savefig('figs/scatter_AWS_S2_'+what+'.png', bbox_inches='tight', dpi=300)


# same as above, only one panel
def correlation_scatter(ds, st, nogood, what, buffer):

    df = ds#.resample('H').mean()

    dfpoint = pd.read_csv('/Users/leahartl/Desktop/WSS/WSS_Albedo/output_GEE/extractpoint_AWS_allyear_'+buffer+'.csv', parse_dates=True)
    dfpoint.index = pd.to_datetime(dfpoint.dttime)
    dfpoint['QC'] = 0

    for no in nogood: 
        dfpoint.loc[(dfpoint.index.day==no.day) & (dfpoint.index.month==no.month) & (dfpoint.index.year==no.year), 'liangAlbedo'] = np.nan
        dfpoint.loc[(dfpoint.index.day==no.day) & (dfpoint.index.month==no.month) & (dfpoint.index.year==no.year), 'QC'] = 1

    B1 = dfpoint[['liangAlbedo', 'name', 'dttime', 'QC']].copy()
    B1.index = B1.index.date
    merged = pd.merge(df, B1, left_index=True, right_index=True)
    
    # print quantiles:
    print(merged[['liangAlbedo', 'albedo_sum']].quantile([0.01, .05, 0.1]))

    # print low albedo subset:
    print(merged.loc[merged.liangAlbedo<0.4])

    # print manually removed values:
    print(merged.loc[merged.QC==1])
    print('percentage removed: ', 100* merged.loc[merged.QC==1].shape[0] / merged.shape[0])
    #stop


    # set colors to indicate winter (blue-ish) and summer (red-ish)
    clrs = ['skyblue', 'cornflowerblue', 'royalblue', 'blue', 'teal', 'darkturquoise', 'silver', 'slategrey', 'violet', 'magenta', 'darkviolet', 'crimson']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharey=True)

    # print bias and rmse
    biasAll = merged.liangAlbedo - merged[what]
    print('meanbias all data:', biasAll.abs().mean(), 'N:', biasAll.shape)

    dif2 = biasAll**2
    rms = np.sqrt(dif2.mean())
    print('rms all data:', rms)


    # # print bias and rmse for very low albedo values
    v_low = merged.loc[merged['liangAlbedo']<0.2]
    biasAll_vlow = v_low.liangAlbedo - v_low[what]
    print('meanbias vlow:', biasAll_vlow.abs().mean(), 'N:', biasAll_vlow.shape)

    dif2_vlow = biasAll_vlow**2
    rms_vlow = np.sqrt(dif2_vlow.mean())
    print('rms vlow:', rms_vlow)

    cor_monthly = pd.DataFrame(index=[10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=['rms', 'absbias', 'mean_dif'])

    for i, m in enumerate([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        sub = merged.loc[merged.index.month == m]
        lb = datetime(2000, m, 1)
        ax.scatter(sub.liangAlbedo, sub[what], c=clrs[i], label=lb.strftime('%b'))

        # errorbars: 
        #14% of AWS albedo
        #13% of S2 albedo 
        ax.errorbar(sub.liangAlbedo, sub[what], xerr=sub[what]*0.16, yerr=sub[what]*0.14, linestyle="None", color='k', linewidth=0.4, zorder=0)
        
        difS2AWS = sub.liangAlbedo - sub[what]
        cor_monthly.loc[m, 'absbias'] = difS2AWS.abs().mean().round(decimals=3)
        cor_monthly.loc[m, 'rms'] = np.sqrt((difS2AWS**2).mean()).round(decimals=3)
        cor_monthly.loc[m, 'mean_dif'] = difS2AWS.mean().round(decimals=3)
        cor_monthly.loc[m, 'median_dif'] = difS2AWS.median().round(decimals=3)
    
    print(cor_monthly)
    cor_monthly.to_csv('out/monthly_comparison.csv')

    ax.annotate("RMSE: "+ str(rms.round(decimals=3)),
             xy=(0.7, 0.18), xycoords='data', fontsize=12,
             #xytext=(x2, y2), textcoords='data',
             ha="center", va="center")
    ax.annotate("Mean bias: "+ str(biasAll.abs().mean().round(decimals=3)),
             xy=(0.7, 0.14), xycoords='data', fontsize=12,
             #xytext=(x2, y2), textcoords='data',
             ha="center", va="center")

    ax.grid('both')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', zorder=1, linewidth=0.5)

    ax.set_xlabel('S2-derived albedo', fontsize=16)
    ax.set_ylabel('AWS albedo', fontsize=16)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.legend(fontsize=14)

    fig.savefig('figs/scatter_AWS_S2_'+what+'buffer'+buffer+'_2.png', bbox_inches='tight', dpi=400, transparent=False)




# This produces the following output:
# Figure: timeseries plot of albedo anomalies, subplots for each year. S2 obs as black dots for comparison.
# csv file with monthly stats
# csv file with number of dark days
def albedo_stats(df, nogood, skiplist, what):

    df_day = df
    df_ts = df_day.groupby('doy').mean()
    df_ts['date'] = pd.to_datetime(df_ts.index, format='%j')

    df_month = df_day[[what]].groupby(df_day.index.month).mean()
    df_month_st = df_day[[what]].groupby(df_day.index.month).std()
    df_month_st['avg'] = df_month[what]
    df_month_st['unc_20perc'] = df_month_st['avg']*0.2

    # print(df_month_st.round(2))
    df_month_st.round(2).T.to_csv('out/monthlystats_'+what+'.csv')

    # load S2 data
    df_point = pd.read_csv('/Users/leahartl/Desktop/WSS/WSS_Albedo/output_GEE/extractpoint_AWS_allyear_5.csv', parse_dates=True)
    df_point.index = pd.to_datetime(df_point.dttime)
    # print(df_point)
    for no in nogood: 
        df_point.loc[(df_point.index.day==no.day) & (df_point.index.month==no.month) & (df_point.index.year==no.year)] = np.nan

    myFmt = mdates.DateFormatter('%b')
    fig, ax = plt.subplots(7, 1, figsize=(14, 10), sharex=True, sharey=True)
    l04 = []
    l03 = []
    l02 = []
    mindate = []
    minalb = []
    for i, y in enumerate(np.arange(2018, 2025)):
        tm = df_day.loc[df_day.index.year == y]
        df_p = df_point.loc[df_point.index.year == y]
        df_p['doy'] = df_p.index.dayofyear
        df_p['date'] = pd.to_datetime(df_p['doy'], format='%j')
        # print(df_p)
        tm.index = tm['doy']
        mrg = pd.merge(tm[what], df_ts[what], left_index=True, right_index=True)
        mrg['anom'] = mrg[what+'_x'] - mrg[what+'_y']

        mrg['strdate'] =str(y)+mrg.index.astype(str)
        # print(mrg)
        mrg['ydate'] = pd.to_datetime(mrg.strdate, format='%Y%j')
        mrg['date'] = pd.to_datetime(mrg.index, format='%j')
        
        mrgbright = mrg.copy()
        mrgbright.loc[mrg.anom < 0] = np.nan

        mrgdark = mrg.copy()
        mrgdark.loc[mrg.anom >= 0] = np.nan
        

        lt04 = mrg.loc[mrg[what+'_x']<0.4]
        # print('less than 0.4', y, len(lt04))

        lt03 = mrg.loc[mrg[what+'_x']<0.3]

        lt02 = mrg.loc[mrg[what+'_x']<0.2]
        # print('less than 0.2', y, len(lt02), lt02)

        md = mrg.loc[mrg[what+'_x'] == mrg[what+'_x'].min()]['ydate'].values[0]

        l02.append(len(lt02))
        l03.append(len(lt03))
        l04.append(len(lt04))
        mindate.append(md)
        minalb.append(mrg[what+'_x'].min())

        ax[i].fill_between(mrgdark['date'], mrgdark[what+'_x'], mrgdark[what+'_y'], color='darkgrey')
        ax[i].fill_between(mrgbright['date'], mrgbright[what+'_x'], mrgbright[what+'_y'], color='powderblue')

        ax[i].plot(df_ts['date'], df_ts[what], c='k', linestyle='-', linewidth=0.8)

        ax[i].scatter(df_p['date'], df_p.liangAlbedo, c='k', s=8)


        ax[i].grid('both')
        ax[i].set_ylim(0.1, 1)
        ax[i].set_xlim(pd.to_datetime(1, format='%j'), pd.to_datetime(366, format='%j'))
        ax[i].set_yticks([0.2, 0.4, 0.6, 0.8])
        ax[i].set_title(y, fontsize=16)

        ax[i].xaxis.set_major_formatter(myFmt)
        ax[i].xaxis.set_tick_params(labelsize=16)
        ax[i].yaxis.set_tick_params(labelsize=18)

        ax[i].axhline(y=0.4, color='k', linewidth=0.5)
        # ax[i].axhline(y=0.34, color='blue', linewidth=0.5)


        for gap in skiplist:
            if int(gap[0][:4])==y:
                print ('gap')
                d1=pd.to_datetime(gap[0]).dayofyear
                d2=pd.to_datetime(gap[1]).dayofyear


                ax[i].axvspan(pd.to_datetime(d1, format='%j'), pd.to_datetime(d2, format='%j'), color='yellow', alpha=0.4)

        ax[3].set_ylabel('Albedo', fontsize=18)


        ps = []
        g = mpatches.Patch(color='yellow', alpha=0.4, label='data gap')
        ps.append(g)
        n = mpatches.Patch(color='darkgrey', alpha=0.8, label='neg. anom.')
        ps.append(n)
        p = mpatches.Patch(color='powderblue', alpha=0.8, label='pos. anom.')
        ps.append(p)
        normal = Line2D([0], [0], linestyle='-', color='k', zorder=10, label='time series mean')
        ps.append(normal)
        s2 = Line2D([0], [0], marker='o', linestyle='None', color='k',  markersize=4, zorder=10, label='S2-derived albedo')
        ps.append(s2)

        # ps.append(g)
        handles = (ps)
        ax[0].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.35, 0.3), ncol=5)

        plt.tight_layout()

    dfdark = pd.DataFrame(index=np.arange(2018, 2025))
    dfdark['lt04'] = l04
    dfdark['lt03'] = l03
    dfdark['lt02'] = l02
    dfdark['minalb'] = minalb
    dfdark['mindate'] = mindate
    dfdark['minalb'] = dfdark['minalb'].round(2)
    print(dfdark)
    # make csv file with the number of dark days
    dfdark.to_csv('out/countdays_'+what+'2025.csv')
    # save figure
    fig.savefig('figs/timeseries_anomalies_'+what+'2025.png', bbox_inches='tight', dpi=300)

# similar to the above function, figure adjusted based on revier comments.
def albedo_stats_rev(df, nogood, skiplist, what):

    df_day = df
    df_ts = df_day.groupby('doy').mean()
    df_ts['date'] = pd.to_datetime(df_ts.index, format='%j')

    df_month = df_day[[what]].groupby(df_day.index.month).mean()
    df_month_st = df_day[[what]].groupby(df_day.index.month).std()
    df_month_st['avg'] = df_month[what]
    df_month_st['unc_20perc'] = df_month_st['avg']*0.2

    df_month_st.round(2).T.to_csv('out/monthlystats_'+what+'.csv')

    # load S2 data
    # df_point = pd.read_csv('/Users/leahartl/Desktop/WSS/WSS_Albedo/output_GEE/extractpoint_AWS_allyear_5.csv', parse_dates=True)
    # df_point.index = pd.to_datetime(df_point.dttime)
    # print(df_point)
    # for no in nogood: 
        # df_point.loc[(df_point.index.day==no.day) & (df_point.index.month==no.month) & (df_point.index.year==no.year)] = np.nan

    myFmt = mdates.DateFormatter('%b')
    fig, ax = plt.subplots(7, 1, figsize=(14, 12), sharex=True, sharey=True)
    l04 = []
    l03 = []
    l02 = []
    mindate = []
    minalb = []
    for i, y in enumerate(np.arange(2018, 2025)):
        tm = df_day.loc[df_day.index.year == y]
        # df_p = df_point.loc[df_point.index.year == y]
        # df_p['doy'] = df_p.index.dayofyear
        # df_p['date'] = pd.to_datetime(df_p['doy'], format='%j')
        tm.index = tm['doy']
        mrg = pd.merge(tm[what], df_ts[what], left_index=True, right_index=True)
        mrg['anom'] = mrg[what+'_x'] - mrg[what+'_y']

        mrg['strdate'] =str(y)+mrg.index.astype(str)
        # print(mrg)
        mrg['ydate'] = pd.to_datetime(mrg.strdate, format='%Y%j')
        mrg['date'] = pd.to_datetime(mrg.index, format='%j')
        
        mrgbright = mrg.copy()
        mrgbright.loc[mrg.anom < 0] = np.nan

        mrgdark = mrg.copy()
        mrgdark.loc[mrg.anom >= 0] = np.nan
        

        lt04 = mrg.loc[mrg[what+'_x']<0.4]
        # print('less than 0.4', y, len(lt04))

        lt03 = mrg.loc[mrg[what+'_x']<0.3]

        lt02 = mrg.loc[mrg[what+'_x']<0.2]
        # print('less than 0.2', y, len(lt02), lt02)

        md = mrg.loc[mrg[what+'_x'] == mrg[what+'_x'].min()]['ydate'].values[0]

        l02.append(len(lt02))
        l03.append(len(lt03))
        l04.append(len(lt04))
        mindate.append(md)
        minalb.append(mrg[what+'_x'].min())

        ax[i].fill_between(mrgdark['date'], mrgdark[what+'_x'], mrgdark[what+'_y'], color='darkgrey')
        ax[i].fill_between(mrgbright['date'], mrgbright[what+'_x'], mrgbright[what+'_y'], color='powderblue')

        ax[i].plot(df_ts['date'], df_ts[what], c='grey', linestyle='-', linewidth=0.8)
        ax[i].plot(pd.to_datetime(tm['doy'], format='%j'), tm[what], c='k', linestyle='-', linewidth=1, zorder=100)

        # ax[i].scatter(df_p['date'], df_p.liangAlbedo, c='k', s=8)


        ax[i].grid('both')
        ax[i].set_ylim(0.1, 1)
        ax[i].set_xlim(pd.to_datetime(1, format='%j'), pd.to_datetime(366, format='%j'))
        ax[i].set_yticks([0.2, 0.4, 0.6, 0.8])
        ax[i].set_yticklabels([ '', '0.4', '', '0.8'])
        ax[i].set_title(y, fontsize=18)

        ax[i].xaxis.set_major_formatter(myFmt)
        ax[i].xaxis.set_tick_params(labelsize=18)
        ax[i].yaxis.set_tick_params(labelsize=20)

        ax[i].axhline(y=0.4, color='k', linewidth=0.5)
        # ax[i].axhline(y=0.34, color='blue', linewidth=0.5)


        for gap in skiplist:
            if int(gap[0][:4])==y:
                print ('gap')
                d1=pd.to_datetime(gap[0]).dayofyear
                d2=pd.to_datetime(gap[1]).dayofyear


                ax[i].axvspan(pd.to_datetime(d1, format='%j'), pd.to_datetime(d2, format='%j'), color='yellow', alpha=0.4)

        # ax[-1].legend()
        ax[3].set_ylabel('Albedo', fontsize=18)


        ps = []

        g = mpatches.Patch(color='yellow', alpha=0.4, label='data gap')
        ps.append(g)
        dailyalbedo = Line2D([0], [0], linestyle='-', color='k', zorder=10, label='daily albedo')
        ps.append(dailyalbedo)
        
        normal = Line2D([0], [0], linestyle='-', linewidth=0.8, color='grey', zorder=10, label='time series mean')
        ps.append(normal)
        
        n = mpatches.Patch(color='darkgrey', alpha=0.8, label='neg. anom.')
        ps.append(n)
        p = mpatches.Patch(color='powderblue', alpha=0.8, label='pos. anom.')
        ps.append(p)

        # s2 = Line2D([0], [0], marker='o', linestyle='None', color='k',  markersize=4, zorder=10, label='S2-derived albedo')
        # ps.append(s2)

        # ps.append(g)
        handles = (ps)
        ax[0].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.42, 0.3), ncol=5, fontsize=14)

        plt.tight_layout()

    dfdark = pd.DataFrame(index=np.arange(2018, 2025))
    dfdark['lt04'] = l04
    dfdark['lt03'] = l03
    dfdark['lt02'] = l02
    dfdark['minalb'] = minalb
    dfdark['mindate'] = mindate
    dfdark['minalb'] = dfdark['minalb'].round(2)
    print(dfdark)
    # make csv file with the number of dark days
    dfdark.to_csv('out/countdays_'+what+'2025.csv')
    # save figure
    fig.savefig('figs/timeseries_anomalies_'+what+'2025_rev.png', bbox_inches='tight', dpi=300)





# print quantiles:
print('Quantiles:', df_mean.quantile([0.01, .05, 0.1]))


####
# This is prep for a fig with two subplots showing the effect of the tilt correction for example weeks in winter and summer.
# 'albedo_cor_summerwinter'
# set desired dates for the summer and winter examples here:
# day = '2020-11-23'
# day1 = '2020-12-01'
day = '2020-12-17'
day1 = '2020-12-21'

# daySummer = '2021-06-12'
# day1Summer = '2021-06-22'
daySummer = '2021-06-14'
day1Summer = '2021-06-18'

# day = '2019-08-10'
# day1 = '2019-09-02'

both = data2.loc[day+' 00:00' : day1+' 23:50']
bothSummer = data2.loc[daySummer+' 00:00' : day1Summer+' 23:50']


#### PLOTS:
########
# sup fig: 2 panel fig with example weeks in winter and summer showing impact pf tilt correction
# dailyCor_SummerWinter(both, bothSummer)


# ######
# # sup fig: this plots the difference between tilt corrected daily albedo and uncorrected values.
# # 'albedo_cor'
# Scatterplot_Cor(data2, df_mean)



# ####
# # the last parameter passed to the function is the name of the "albedo" column to use. 
# # set to "albedo_sum" to use albedo computed as ratio of daily sum SWout to daily sum SWin
# # make time series figure ("timeseries_anomalies"), make csv files "countdays" and "monthly stats"
# albedo_stats(df_mean, baddates_sat, skiplist, 'albedo_sum')
#albedo_stats_rev(df_mean, baddates_sat, skiplist, 'albedo_sum')

######
# scatter plots AWS with s2:
# #only one panel: 
# SET THE BUFFER VALUE HERE! USE 5 for main analysis, 10 to check how a larger buffer impacts results.
# This also prints RMSE, mean bias, and the perentage of removed data.
# paper fig, scatter plot:
correlation_scatter(df_mean, 'wss', baddates_sat, 'albedo_sum', '5')

# sup fig, scatter plot for larger buffer:
# correlation_scatter(df_mean, 'wss', baddates_sat, 'albedo_sum', '10')



# not needed:
####
# this is an alternative visual of the AWS albedo data (all years in one plot, color coded)
# inset axis zooms in on summer months. Figure: "timeseries_allinone"
# albedoAllYears(df_mean, 'albedo_sum')

####
# meshgrid plot: another alternative visual ("meshgridAlbedo")
# albedoMesh(df_mean)


plt.show()
stop





