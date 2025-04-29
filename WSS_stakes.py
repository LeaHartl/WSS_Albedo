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


import glob

from matplotlib_scalebar.scalebar import ScaleBar

# supress copy warning - careful 
pd.options.mode.chained_assignment = None  # default='warn'

# set file path to stake data file:
# stakesfile = '/Users/leahartl/Desktop/WSS/process_stakes/WSS_stakes_point_mass_balance.csv'
stakesfile = '/Users/leahartl/Desktop/WSS/Re__Updates_WSS_paper/WSS_stakes_point_mass_balance.csv'

# load AWS data:
file = '/Users/leahartl/Desktop/WSS/AWS_wss/out/AWS_WSS_proc2025.csv'
data = pd.read_csv(file, parse_dates=True, skiprows=[0,2,3])

data.index = pd.to_datetime(data['TIMESTAMP'])
data = data[['SWin_Avg', 'SWout_Avg']]

## make daily albedo: 
data.loc[data['SWout_Avg']<2, 'SWout_Avg']=np.nan
data.loc[data['SWout_Avg']<2, 'SWin_Avg']=np.nan
df_mean = data.resample('d').mean()
df_mean['albedo_sum'] = data.resample('d').sum()['SWout_Avg'] / data.resample('d').sum()['SWin_Avg']
df_mean.loc[df_mean.albedo_sum<0, 'albedo_sum']=np.nan
df_mean.loc[df_mean.albedo_sum>1, 'albedo_sum']=np.nan

## clean daily albedo (manual filtering of bad dates and faulty values)
# these periods have riming/faulty values in the AWS data - remove
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

for s in skiplist:
    df_mean.loc[(df_mean.index >= pd.to_datetime(s[0])) & (df_mean.index <= pd.to_datetime(s[1])), 'albedo_sum'] = np.nan

# set to nan if albedo is less than 0.4 between November and May, less than 0.6 from Dec to Mar.
df_mean.loc[((df_mean.index.month>=11) | (df_mean.index.month<=5)) & (df_mean.albedo_sum<0.4), 'albedo_sum'] = np.nan
df_mean.loc[((df_mean.index.month>=12) | (df_mean.index.month<=3)) & (df_mean.albedo_sum<0.6), 'albedo_sum'] = np.nan


## prepare list of dates with bad satellite data over the summit region. These will be removed from the sat data:
# list of bad dates: 
nogood = pd.to_datetime(['2018-08-02',
                             '2018-09-29', '2018-11-25', '2018-12-08',
                             '2019-01-27', '2019-04-02', '2019-04-04', '2019-08-22', '2019-08-30','2019-12-15',
                             '2020-02-06', '2020-11-19', '2020-12-04', 
                             '2021-01-31', '2021-12-14', '2021-11-19',
                             '2022-01-03', '2022-02-07', '2022-08-26', '2022-10-20', '2022-12-27', '2022-12-29', 
                             '2023-01-13', '2023-02-20', '2023-12-29',
                             '2024-01-08', '2024-01-28', '2024-02-15', '2024-12-11'])
    # why: clouds, cloud shadow, bad image, cloud shadow, 18
    #bad image- too dark, bad img-too dark, cloud, cloud/cloud shadow, likely shadow(also dark ice though), too dark-bad image, 19
    #too dark-bad image, cloud, cloud shadow, 20
    #cloud, bad image, bad image-toodark, 21
    #bad image, cloud, cloud, bad image/cloud, cloud, cloud or shadow?, 22
    # cloud or shadow?, too dark-weird shadow?, too dark-bad image? 23
    #bad image, not sure - shadow?, not sure - shadow?, 12 11 not sure - maybe cloud?
    #'2018-11-10', --> only one stake (f) in cloud shadow


# function to load stake data file and assign colors to be used in plotting functions
def ReadStakes(fname):
    stakes = pd.read_csv(fname)
    stakes['date1'] = pd.to_datetime(stakes['date1'], format="%d.%m.%Y")
    stakes['date0'] = pd.to_datetime(stakes['date0'], format="%d.%m.%Y")
    sts = stakes['name'].unique()
    stakes['daily'] = stakes['ice_ablation_mmwe'] / stakes['period'] 

    clrs = cm.tab10(np.linspace(0, 1, len(sts)))#np.linspace(0.7, 0.9, len(fls)))
    clrs_yr = cm.tab10(np.linspace(0, 1, len([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])))
    stakes['color'] = ''
    stakes['color_yr'] = ''

    for j, st in enumerate(sts):
        stakes.loc[stakes['name'] ==st, 'color'] = mcolors.rgb2hex(clrs[j], keep_alpha=True)
   
    for js, yr in enumerate(stakes.date1.dt.year.unique()):
        stakes.loc[stakes.date1.dt.year ==yr, 'color_yr'] = mcolors.rgb2hex(clrs_yr[js], keep_alpha=True)

    return(stakes, clrs_yr)



# function to read csv files with GEE ouput of albedo at stake positions
def ReadAlbedo(stake_data, buffer, nogood):

    # set file path to load GEE output:
    fldr = 'output_GEE/'

    # set some additional parameters needed for comptations and plotting:
    pts = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'AWS']
    clrs = cm.tab10(np.linspace(0, 1, len(pts)))

    stakes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
    data = []

    low = []
    vlow = []
    gt = []
    all_dat = []

    # loop through the files that contain the data extracted per pixel (GEE output). Each stake has a csv file:
    for i, pt in enumerate(pts):
        df = pd.read_csv(fldr+'extractpoint_'+pt+'_allyear_'+buffer+'.csv', index_col='dttime')
        df.index = pd.to_datetime(df.index)

        # remove bad dates (clouds, shadows, ... in the satellite imagery)
        for no in nogood: 
            df.loc[(df.index.day==no.day) & (df.index.month==no.month) & (df.index.year==no.year)] = np.nan

        if pt == 'F':
            df.loc[(df.index.day==10) & (df.index.month==11) & (df.index.year==2018)] = np.nan

        if pt != 'AWS':
            dat22 = stake_data.loc[stake_data['name']==stakes[i]]

        # add data to lists: seprate lists for different albedo thresholds:
        lt02 = df.loc[df.liangAlbedo<=0.2]
        lt04 = df.loc[df.liangAlbedo<=0.4]
        gt06 = df.loc[df.liangAlbedo>0.6]
        all_dat.append(df)
        low.append(lt04)
        vlow.append(lt02)
        gt.append(gt06)


    # turn lists into data frames for all stakes:
    vlowdf = pd.concat(vlow)
    lowdf = pd.concat(low)
    gt_06 = pd.concat(gt)
    all_d = pd.concat(all_dat)

    # print as reality check
    print(all_d)
  
    # print quantiles
    print(all_d[['liangAlbedo']].quantile(q=0.1))
    print(all_d[['liangAlbedo']].quantile(q=0.05))
    print(all_d[['liangAlbedo']].quantile(q=0.01))

    # count v low albedo days per month:
    vlowdf['month'] = vlowdf.index.month
    grouped = vlowdf[['liangAlbedo', 'month']].groupby('month').count()
    grouped['perc'] = grouped['liangAlbedo'] / vlowdf.shape[0]
    # print(vlowdf)
    # print(grouped)
    # print(vlowdf.loc[vlowdf.month==7])
    # print(vlowdf.loc[vlowdf.month==9])

    return(all_d)



# plot: two panels showing time series of albedo at stake positions and AWS albedo. 
def timeseriesplot_S2stakes_2panels(fname, df_mean, nogood):

    fldr = '/Users/leahartl/Desktop/WSS/satAlbedo/output/'
    pts = ['a', 'b', 'c', 'd', 'e', 'f', 'BL0319']
    

    pts2 = ['a', 'b', 'c', 'd', 'e', 'f', 'BL0319', 'g']
    stakes = ['A', 'B', 'C', 'D', 'E', 'F', 'BL0319']
    clrs3 = cm.tab10(np.linspace(0, 1, len(pts2)))

    clrs = ['blue', 'orange', 'green', 'purple', 'cyan', 'pink', 'grey']

    day1 = pd.to_datetime('2021-08-01')
    day2 = pd.to_timedelta(31, unit='d')

    day3 = pd.to_datetime('2022-07-05')
    day4 = pd.to_timedelta(82, unit='d')

    left, bottom, width, height = (day1, 0, day2, 1)
    rect1 = plt.Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.1)

    left2, bottom, width2, height = (day3, 0, day4, 1)
    rect2 = plt.Rectangle((left2, bottom), width2, height,
                     facecolor="black", alpha=0.1)
    rect22 = plt.Rectangle((left2, bottom), width2, height,
                     facecolor="black", alpha=0.1)


    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = ax.flatten()

    ax[0].add_patch(rect2)
    ax[1].add_patch(rect22)

    myFmt2 = mdates.DateFormatter('%Y-%m-%d')

    low = []
    vlow =[]
    gt = []
    all_dat = []

    for i, pt in enumerate(pts):
        df = pd.read_csv(fldr+'extractpoint_'+pt+'_allyear_5.csv', index_col='dttime')
        df.index = pd.to_datetime(df.index)
        df = df[df.Blue <=1]
        df = df[df.Green <=1]
        df = df[df.Red <=1]

        for no in nogood: 
            df.loc[(df.index.day==no.day) & (df.index.month==no.month) & (df.index.year==no.year)] = np.nan

        if pt == 'f':
            df.loc[(df.index.day==10) & (df.index.month==11) & (df.index.year==2018)] = np.nan

        lt02 = df.loc[df.liangAlbedo<=0.2]
        lt04 = df.loc[df.liangAlbedo<=0.4]
        gt06 = df.loc[df.liangAlbedo>0.6]
        all_dat.append(df)
        vlow.append(lt02)
        low.append(lt04)
        gt.append(gt06)
        # print(lt04)

        if pt == 'BL0319':
            ax[1].scatter(df.index, df.liangAlbedo, color=mcolors.rgb2hex(clrs[i], keep_alpha=True), label='H', s=9)
        else:
            ax[1].scatter(df.index, df.liangAlbedo, color=mcolors.rgb2hex(clrs[i], keep_alpha=True), label=pt.upper(), s=9)
     
    dfAWS = pd.read_csv('/Users/leahartl/Desktop/WSS/satAlbedo/output/extractpoint_AWS_allyear_5.csv', parse_dates=True)
    dfAWS.index = pd.to_datetime(dfAWS.dttime)
    dfAWS = dfAWS[dfAWS.Blue <= 1]
    dfAWS = dfAWS[dfAWS.Green <= 1]
    dfAWS = dfAWS[dfAWS.Red <= 1]

    for no in nogood:
        dfAWS.loc[(dfAWS.index.day == no.day) & (dfAWS.index.month == no.month) & (dfAWS.index.year == no.year)] = np.nan

    ax[0].scatter(dfAWS.index, dfAWS.liangAlbedo, color='r', label='AWS, S2', s=12, zorder=100, marker='*')

    ax[0].plot(df_mean.index, df_mean.albedo_sum, color='k', label='AWS, in situ', zorder=100, linewidth=0.2)

    vlowdf = pd.concat(vlow)
    lowdf = pd.concat(low)
    gt_06 = pd.concat(gt)
    all_d = pd.concat(all_dat)

    # print low albedo subsets to check dates:
    print(dfAWS.loc[dfAWS.liangAlbedo<=0.4,'liangAlbedo'])
    print(lowdf)
    # print(df_mean.loc[df_mean.albedo_sum<=0.4,'albedo_sum'].head(20))

    # stop
    print('Quantiles:', all_d[['liangAlbedo']].quantile(q=0.1))
    print(all_d[['liangAlbedo']].quantile(q=0.05))
    print(all_d[['liangAlbedo']].quantile(q=0.01))
    # stop

    lowdf['month'] = lowdf.index.month
    grouped = lowdf[['liangAlbedo', 'month']].groupby('month').count()
    grouped['perc'] = grouped['liangAlbedo'] / lowdf.shape[0]
    # print(vlowdf)
    # print(grouped)

    # print(vlowdf.loc[vlowdf.month==7])
    # print(vlowdf.loc[vlowdf.month==9])

    ax[0].set_title('AWS position: in situ vs. S2 albedo', fontsize=18)
    ax[1].set_title('Stake positions: S2 albedo', fontsize=18)
    ax[0].legend(loc='center right', bbox_to_anchor=(1.22, 0.5),fontsize=14)
    ax[1].legend(loc='center right', bbox_to_anchor=(1.12, 0.5),fontsize=14)
    
    for a in ax:
        a.grid('both')
        a.set_ylim([0, 1])
        a.set_ylabel('Albedo', fontsize=18)

        a.set_xlim(pd.to_datetime('2018-01-01'), pd.to_datetime('2024-12-31'))
        ticks = pd.to_datetime(['2018-07-01','2019-01-01', '2019-07-01', '2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01', '2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01'])
        a.set_xticks(ticks)
        a.tick_params(axis='x', rotation=45, labelsize=16)
        a.tick_params(axis='y', labelsize=18)

        a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        a.xaxis.set_major_formatter(myFmt2)

    ax[0].text(pd.to_datetime('2018-04-01'), 0.1,'a', fontsize=12,
        bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    ax[1].text(pd.to_datetime('2018-04-01'), 0.1,'b', fontsize=12,
        bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))


    fig.savefig('figs/albedo_timeseriesAWSandstakes.png', dpi=200, bbox_inches='tight')




def countLowAlbDays(all_d, stake_data2, annual_stakes):

    # load AWS data
    AWSdat = pd.read_csv('/Users/leahartl/Desktop/WSS/WSS_Albedo/out/albedo_cor_daily2025.csv', parse_dates=True, index_col='TIMESTAMP')

    # add count of scenes with albedo below 0.4 for each intermediate stake reading:
    stake_data2['lt04_count']=np.nan
    stake_data2['img_count']=np.nan
    stake_data2['img_count_extrapl']=0

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    fs = 14
    ax = ax.flatten()
    clrs_yr = cm.viridis(np.linspace(0, 1, len([2018, 2019, 2020, 2021, 2022, 2023, 2024])))
    clrs_hex = cm.tab10(np.linspace(0, 1, len(stake_data2.name.unique())))

    stake_data2 = stake_data2.sort_values('date0')
    # remove if no value recorded.
    stake_data2 = stake_data2.dropna(subset=['ice_ablation_cm'])

    dts = list(dict.fromkeys(zip(pd.to_datetime(stake_data2.date0), pd.to_datetime(stake_data2.date1))))
    clrs2 = cm.viridis(np.linspace(0, 1, len(dts)))

    thresh = 0.4
    counts_toplot = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E', 'F', 'H'], index=[2018, 2019, 2020, 2021, 2022, 2023, 2024])
    for j, st in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'H', 'AWS']):
        print(st)
        # if st != 'AWS':
        stks = stake_data2.loc[stake_data2['name'] == st]
        albdat = all_d.loc[all_d['name'] == st][['name', 'liangAlbedo']]

        stksub = albdat.loc[albdat.liangAlbedo<thresh]
        stksub.index = pd.to_datetime(stksub.index).date
        mrg = pd.merge(AWSdat, stksub, left_index=True, right_index=True, how='outer')

        mrg['diff_aws'] = mrg.albedo_sum.diff()
        mrg['itsnowed'] = 0
        mrg.loc[(mrg.diff_aws > 0.2), 'itsnowed'] = 1
        mrg['count_low'] = 0
        mrg.loc[~mrg.liangAlbedo.isnull(), 'count_low'] = 1
        mrg.loc[mrg.itsnowed==1, 'count_low'] = 2

        newvals = mrg.count_low.values.copy()
        for i, v in enumerate(newvals[:-1]):
            if (v == 1) & (newvals[i+1]==0):
                newvals[i+1] = 1
        mrg['count_low2'] = newvals
        # discard nan values due to AWS data gaps
        mrg.loc[mrg.diff_aws.isnull(), 'count_low2'] = np.nan
        # print(mrg.loc[(mrg.index.year==2018)&(mrg.index.month==8)])

        forplot1 = mrg.loc[mrg['count_low2']==1]
        # print(forplot1)
        forplot = forplot1.resample('YE').sum()
        forplot.index=forplot.index.year
        #print(forplot)
        counts_toplot[st] = forplot['count_low2']#.values

        # separate by intervals of stake readings
        if st != 'AWS':
            for d in stks.date1.values:
                d0 = stks.loc[stks.date1==d]['date0'].values[0]
                albsub = albdat.loc[d0 : d]
                lt04_sub = albsub.loc[albsub.liangAlbedo < thresh].shape[0]

                stake_data2.loc[(stake_data2.date1==d) & (stake_data2.name==st), 'lt04_count'] = lt04_sub
                stake_data2.loc[(stake_data2.date1==d) & (stake_data2.name==st), 'img_count'] = albsub.shape[0]

                mrg_sub = mrg.loc[d0:d]
                counts = mrg_sub['count_low2'].value_counts()
                if counts.loc[counts.index==1].values.size>0:
                    stake_data2.loc[(stake_data2.date1==d) & (stake_data2.name==st), 'img_count_extrapl'] = counts.loc[counts.index==1].values[0]


    # print(counts_toplot.to_latex(float_format="%.0f"))
    # print(annual_stakes.to_latex(float_format="%.0f"))
    # print(counts_toplot)
    # print(annual_stakes.sum())
    print(annual_stakes)


    # format and merge dataframes to produce the table for the manuscript:
    annual_st_fortable = annual_stakes.drop(columns='G')
    # add 0.5 to index so that we can sort by index later to get alternating rows of day-counts and ablation on the table:
    annual_st_fortable.index = annual_st_fortable.index + 0.5
    fortable = pd.concat([counts_toplot.drop(columns='AWS'), annual_st_fortable], ignore_index=False, sort=False)
    fortable = fortable.sort_index()
    # write to file:
    fortable.to_csv('out/yearlystakes_countsandablation.csv')
    fortable.to_latex('out/yearlystakes_countsandablation.tex', float_format="%.0f")

    AWS_fortable = counts_toplot['AWS']
    AWS_fortable.to_csv('out/counts_lowalb_sat_AWS.csv')


    # compute correlation: 
    counts_cr = counts_toplot[['A', 'B', 'C', 'D', 'E', 'F', 'H']]
    counts_cr1 = counts_cr.values.ravel()

    annual_stakes_cr1 = annual_stakes.drop(columns=['G']).values.ravel()

    both = pd.DataFrame(columns=['counts', 'stakevals'])
    both['counts']=counts_cr1
    both['stakevals']=annual_stakes_cr1
    print(both)

    R2 = both.corr()
    print('correlation, R2:', R2)


    clrs = ['blue', 'orange', 'green', 'purple', 'cyan', 'pink', 'grey']

    for j, st in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'H']):
        ax[0].scatter(annual_stakes[st], counts_toplot[st], color=mcolors.rgb2hex(clrs[j], keep_alpha=True), label=st, s=80)
    ax[0].legend(fontsize=fs, loc='upper left', bbox_to_anchor=[-0.01, 1.2], ncol=4)
    # sort by date
    for i, d in enumerate(dts):
        if d[0].year>2017:
            tm = stake_data2.loc[(pd.to_datetime(stake_data2.date0) == d[0])&(pd.to_datetime(stake_data2.date1) == d[1])].dropna(subset=['ice_ablation_cm'])#& (stake_data2.period < 100)
            
            if len(tm.date0.values)>0:
                ax[1].scatter(tm.ice_ablation_mmwe/tm.period, 100*tm.img_count_extrapl/tm.period, label=d[0].strftime('%Y-%m-%d')+' - '+d[1].strftime('%Y-%m-%d'), color=clrs2[i], s=80)#, s=tm.period)
                awssub = AWSdat.loc[tm.date0.values[0]:tm.date1.values[0]]
                awscount = awssub.loc[awssub.albedo_sum<0.4]
                tm['awscount'] = len(awscount)
         
    ax[0].grid('both')
    ax[1].grid('both')
    ax[1].legend(loc='center right', bbox_to_anchor=(1.7, 0.5), fontsize=fs)

    for a in ax:
        a.tick_params(axis='x', labelsize=16)
        a.tick_params(axis='y', labelsize=16)

    ax[0].set_xlabel('Annual ice ablation at stakes [mm w.e.]', fontsize=18)
    ax[1].set_xlabel('Ice ablation rate at stakes [mm w.e./day]', fontsize=18)

    ax[0].set_ylabel('Low albedo at stake positions [days per year]', fontsize=18)
    ax[1].set_ylabel('Low albedo at stake positions [% of measurement period]', fontsize=18)

    ax[0].text(-1700, 0,'a', fontsize=14,
        bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    ax[1].text(-27, 0,'b', fontsize=14,
        bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    fig.savefig('figs/albedo_ablation.png', dpi=200, bbox_inches='tight')


    f2, ax2 = plt.subplots(1, 1, figsize=(8, 9))
    # clrs = ['blue', 'orange', 'green', 'purple', 'cyan', 'pink', 'grey']

    # for j, st in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'H']):
    ax2.scatter(annual_stakes/1000, counts_toplot, color='yellow', s=80, marker='*', edgecolor='k')
    # ax2.legend(fontsize=fs, loc='upper left', bbox_to_anchor=[-0.01, 1.2], ncol=4)
    ax2.set_xlabel('Annual ice ablation at stakes [m w.e.]', fontsize=18)
    ax2.set_ylabel('Low albedo (<0.4) at stake positions [days per year]', fontsize=18)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.grid('both')

    f2.savefig('figs/albedo_ablation_poster.png', dpi=400, bbox_inches='tight', transparent=True)
    # sort by date


## run functions: 

# read csv file with stake data:
stake_data, clrs_yr = ReadStakes(stakesfile)
sts = stake_data['name'].unique()


# compute annual ablation values at the stakes
annual_stakes = pd.DataFrame(columns=sts, index=[2018, 2019, 2020, 2021, 2022, 2023, 2024])
for st in sts:#['A', 'B', 'C', 'D', 'E', 'F', 'H']:
    tm = stake_data.loc[stake_data.name == st]
    tm.index = tm.date1
    an_st = tm['ice_ablation_mmwe'].resample('YE').sum()
    an_st.index = an_st.index.year
    annual_stakes[st] = an_st
    print(st, an_st)

print(annual_stakes)
print('cumulative 2018-2023:', annual_stakes.loc[annual_stakes.index < 2024].sum())
print('cumulative 2018-2024:', annual_stakes.sum())

# this reads the files with S2 albedo extracted for the points. Second input is the buffer (5 or 10m)
all_d = ReadAlbedo(stake_data, '5', nogood)

### PAPER FIG HERE ######
# this makes time series plot of the albedo values at the stakes, two panels
timeseriesplot_S2stakes_2panels(stakesfile, df_mean, nogood)

# counts low albedo days, makes plot, computes and prints correlation between annual stake ablation and low albedo days.
countLowAlbDays(all_d, stake_data, annual_stakes)
plt.show()





