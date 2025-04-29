# -*- coding: utf-8 -*-
"""

AWS RAW DATA PARSING AND FLAGGING PROCEDURE
Based on "Ablation drivers over Weißseespitze: a surface energy balance study", master thesis by Anna Baldo - UIBK, 2024
Adapted to mirror "AWS_processing.py" by Lea Hartl - https://github.com/LeaHartl/MWKVK_processing/blob/main/AWS_processing.py
### ADAPTED AGAIN BY LEA TO INCLUDE SNOW FOR COSIPY PREPRO
"""
#%% Section 1: importing modules and setting variables value

# Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin, lmax


# Variables to set
# Filepath to the AWS raw data file
filepath = '/Users/leahartl/Desktop/WSS/AWS_wss/20172031_20250107_WSS_gesamt.dat' #"insert your filepath to the .dat raw data file"

# yr = '2018_summer'
# start = '2018-07-01'
# end = '2018-09-30'

# yr = '2021_summer'
# start = '2021-08-01' # '2022-07-05' > 2.53
# end = '2021-10-28' # '2022-09-25' > 2.76

# yr = '2021_summer_Aug05_new'
# start = '2021-08-05' # '2022-07-05' > 2.53
# end = '2021-10-28' # '2022-09-25' > 2.76

# yr = '2022_summer'
# start = '2022-07-05'
# end = '2022-09-25'

# yr = 'mean_2018_2024'
# start = '2018-01-01'
# end = '2024-12-31'

# # yr = 'heatwave2022'
# # start = '2022-07-15'
# # end = '2022-07-22'

yr = 'alldata'
start = '2018-01-01'
end = '2025-01-01'

# Sonic ranger height over glacier ice in that specific hydrological year
# sr_hy = 2.53
# Sonic ranger instrument accuracy - to be used as threshold in snow data cleaning procedure
sr_acc = 0.02
# Threshold for riming correction (WSS: 3 deg, adjust as needed)
dm = 3

# supress copy warning - careful
pd.options.mode.chained_assignment = None  # default='warn'

"""
FLAGS LEGEND

All variables (except for snow):
    - System error/Nan value = 1
    - Out of range = 2
    - Riming (on wind speed and direction) = 3
    - Unphysical (SWout > SWin, LWout > 315.6574) = 4

OR

    - Unphysical: SWout > SWin = 4
    - Unphysical: LWout > 315.6574 = 5

Snow:
    - System error/Nan value = 1
    - Out of range = 2  
    - Subsequent measurements too far apart = 3
    - Major gaps = 4
    - Single observations and daily mean too far apart = 5

"""

#%% Section 2: functions

# read in file
def readfile(filepath, start, end):
    
    ds = pd.read_csv(filepath, sep = ',', parse_dates=True, low_memory=False)  # removed skiprows = [0,2,3]

    # set date as index and fix the format - time needs to be UTC
    ds.set_index('TIMESTAMP', inplace=True)
    ds.index = pd.to_datetime(ds.index, format='%Y-%m-%d %H:%M:%S')

    # convert accidental strings in numeric values
    ds = ds.astype(float)
    # drop 'Rain_Tot' because it is always 0 - valid for WSS!
    # ds.drop(columns=['Rain_Tot'], inplace=True)

    # resample to regular 10 minutes interval
    ds_resample = ds.resample('10min').mean()
    # check how this affects the index
    ix_dif=ds.index.difference(ds_resample.index, sort=False)
    if len(ix_dif)>0:
        print('RESAMPLING ISSUE - CHECK')
        
    # Make sure that all NaN values are `np.nan` not `'NAN'` (strings)
    ds_resample = ds_resample.replace('NAN', np.nan)
    
    # make subset based on start and end date
    ds_resample = ds_resample[start:end]
    ds = ds[start:end]

    return (ds_resample, ds)

# write raw data to file:
def writetofile(dat, yr):
    
    # define headers
    header1 = '"TOA5","WSS","CR3000","12216","CR3000.Std.31.08","CPU:WSS.CR3","3076","WSS.1"'
    header2 = '"TIMESTAMP","RECORD","Batt_Min","PTemp_Avg","Tair_Avg","Hum_Avg","SWin_Avg","SWout_Avg","LWin_Avg","LWout_Avg","NR01TC_Avg","NR01K_Avg","NetSW_Avg","NetLW_Avg","Albedo_Avg","UpTot_Avg","DownTot_Avg","NetTot_Avg","LWinCor_Avg","LWoutCor_Avg","Wspeed","Wdir","Wspeed_Max","Dist_Avg","Press_Avg","EisT1_Avg","EisT2_Avg","EisT3_Avg","EisT4_Avg","Snow_Avg"'
    header3 = '"TS","RN","Volt","Celsius","Celsius","%","W/m2","W/m2","W/m2","W/m2","Celsius","Kelvin","W/m2","W/m2","W/m2","W/m2","W/m2","W/m2","W/m2","W/m2","m/s","degree","m/s","m","mbar","mm","Celsius","Celsius","Celsius","Celsius","m"'

    fn = 'AWS/AWS_WSS_raw_'+yr+'.csv'

    with open(fn, 'a') as file:
        file.write(header1+'\n')
        file.write(header2+'\n')
        file.write(header3+'\n')
        dat.to_csv(file, header=False, index=True, sep=',')

# writes processed data to file:
def writetofileProc(dat, yr):

    header1 = '"TOA5","WSS","CR3000","12216","CR3000.Std.31.08","CPU:WSS.CR3","3076","WSS.1"'
    header2 = '"TIMESTAMP","Batt_Min","Tair_Avg","Hum_Avg","Press_Avg","Wspeed","Wdir","Wspeed_Max","SWin_Avg","SWout_Avg","LWin_Cor","LWout_Cor","Dist_Cor","Snow","EisT1_Avg","EisT2_Avg","EisT3_Avg","EisT4_Avg","Batt_Min_flag","Tair_Avg_flag","Hum_Avg_flag","Press_Avg_flag","Wdir_flag","Wspeed_flag","Wspeed_Max_flag","SWin_Avg_flag","SWout_Avg_flag","LWin_Cor_flag","LWout_Cor_flag","EisT1_Avg_flag","EisT2_Avg_flag","EisT3_Avg_flag","EisT4_Avg_flag","Snow_flag"'
    header3 = '"TS","Volt","Celsius","%","mbar","m/s","degree","m/s","W/m2","W/m2","W/m2","W/m2","m","m",,,,,,,,,,,,,,,,'
    
    fn = 'AWS/AWS_WSS_proc_'+yr+'.csv'

    with open(fn, 'a') as file:
        file.write(header1+'\n')
        file.write(header2+'\n')
        file.write(header3+'\n')
        dat.to_csv(file, header=False, index=True, sep=',')

# set flag for no data:
def flagSysError(ds, flag, param):
    ds.loc[(ds[param].isnull()), flag] = 1
    return(ds)

# set flags for data outside of sensor range:
def flagRange(ds, flag, param, low, high):
    if param == 'Tair_Avg':
        ds.loc[(ds[param].values.astype(np.float64) <= low) | (ds[param].values.astype(np.float64) >= high), flag] = 2
    else:
        ds.loc[(ds[param].values.astype(np.float64) < low) | (ds[param].values.astype(np.float64) > high), flag] = 2
    return(ds)

# count occurence of flags
def CountFlags(ds, param):
    nrs = []
    for flag in [1, 2, 3, 4, 5]:
        nr = ds.loc[ds[param+'_flag']==flag, param+'_flag'].shape[0]
        nrs.append(nr)
    return(nrs)

# data processing and flag creation for all variables except snow height
def processData(ds_resample, tr):
    ds_PROC = ds_resample.copy()
    ds_PROC = ds_PROC.astype(float)
    
    # add extra column with sensor height of the SR50
    ds_PROC['sr_hy'] = np.nan

    ds_PROC.loc['2017-10-31':'2018-10-27', 'sr_hy'] = 3.48
    ds_PROC.loc['2018-10-28':'2019-08-23', 'sr_hy'] = 3.63
    ds_PROC.loc['2019-10-01':'2021-10-29', 'sr_hy'] = 2.53
    ds_PROC.loc['2021-10-29':'2022-08-04', 'sr_hy'] = 2.76
    ds_PROC.loc['2022-08-04':'2022-10-04', 'sr_hy'] = 1.95
    ds_PROC.loc['2022-10-04':'2023-10-13', 'sr_hy'] = 2.83
    ds_PROC.loc['2023-10-13':'2024-08-23', 'sr_hy'] = 2.87

    ds_PROC.loc['2024-08-23':'2024-09-20', 'sr_hy'] = 2.25
    #HIER ANPASSEN
    ds_PROC.loc['2024-09-20':'2025-01-07', 'sr_hy'] = 2.92

    # flag battery value when battery is less than 11V, below that the logger doesn't work reliably
    # (Martin checked the instrument documentation and that seems to be the only relevant limit) - valid also for WSS?
    ds_PROC['Batt_Min_flag'] = 0
    ds_PROC.Batt_Min_flag.loc[(ds_PROC.Batt_Min.values.astype(np.float64) <= 11)] = 1

    # TEMPERATURE
    #Set flags for air temp:
    ds_PROC['Tair_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'Tair_Avg_flag', 'Tair_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'Tair_Avg_flag', 'Tair_Avg', -40, 60)

    # number of incidences of each flag - write to data frame
    dsFlags = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=['nr_T', 'percent_T'])
    # create columns for counting temperature flags + percentage 
    dsFlags['nr_T'] = CountFlags(ds_PROC, 'Tair_Avg')
    dsFlags['percent_T'] = 100 * dsFlags['nr_T'] / len(ds_PROC.Tair_Avg)

    # REL HUM
    #Set flags for relative humidity:
    ds_PROC['Hum_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'Hum_Avg_flag', 'Hum_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'Hum_Avg_flag', 'Hum_Avg', 0, 100)

    # humidity flags + percentage 
    dsFlags['nr_Hum'] = CountFlags(ds_PROC, 'Hum_Avg')
    dsFlags['percent_Hum_Avg'] = 100 * dsFlags['nr_Hum'] / len(ds_PROC.Hum_Avg)

    # PRESSURE
    #Set flags for relative humidity:
    ds_PROC['Press_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'Press_Avg_flag', 'Press_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'Press_Avg_flag', 'Press_Avg', 500, 1100)

    # pressure flags + percentage 
    dsFlags['nr_Press'] = CountFlags(ds_PROC, 'Press_Avg')
    dsFlags['percent_Press_Avg'] = 100 * dsFlags['nr_Press'] / len(ds_PROC.Hum_Avg)

    # WIND
    # Set general flags for wind dir:
    ds_PROC['Wdir_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'Wdir_flag', 'Wdir')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'Wdir_flag', 'Wdir', 0, 360)

    # Set general flags for wind speed:
    ds_PROC['Wspeed_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'Wspeed_flag', 'Wspeed')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'Wspeed_flag', 'Wspeed', 0, 60)

    # Set general flags for wind speed max:
    ds_PROC['Wspeed_Max_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'Wspeed_Max_flag', 'Wspeed_Max')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'Wspeed_Max_flag', 'Wspeed', 0, 60)

    # RIMING!
    # Wind speed: flag if it is 0 for more than 1 hour consecutively. 
    # make column that is 0 if wind speed is greater than 0 and 1 otherwise:
    ds_PROC['Wspeed_0'] = 0
    ds_PROC['Wspeed_0'].where(ds_PROC['Wspeed'] != 0, 1, inplace=True)
    # make column counting consecutive ones in the above column:
    ds_PROC['consecutive'] = ds_PROC['Wspeed_0'].groupby((ds_PROC['Wspeed_0'] != ds_PROC['Wspeed_0'].shift()).cumsum()).transform('size') * ds_PROC['Wspeed_0']
    ds_PROC['shift'] = ds_PROC['Wspeed_0'] != ds_PROC['Wspeed_0'].shift()
    # set flag to 3 if windspeed is 0 for more than 6 consecutive values (1 hour)
    ds_PROC.loc[ds_PROC['consecutive'] > 6, 'Wspeed_flag'] = 3

    # Wind direction: flag all 3 consecutive (OR MORE) wind direction measurements that fall within a defined threshold
    ds_PROC['Wdiff'] = ds_PROC['Wdir'].diff(periods=-1)
    # binary column: 0 default, 1 if diff < dm
    ds_PROC['Wdir_bin'] = 0
    ds_PROC.loc[ds_PROC['Wdiff'].abs() < dm, 'Wdir_bin'] = 1
    # column counting consecutive occurence of diff below threshold:
    ds_PROC['consecutive_wdir'] = ds_PROC['Wdir_bin'].groupby((ds_PROC['Wdir_bin'] != ds_PROC['Wdir_bin'].shift()).cumsum()).transform('size') * ds_PROC['Wdir_bin']
    # set flag to 3 if wind dir changes less than 3 deg for more than 3 consecutive measurements
    ds_PROC.loc[ds_PROC['consecutive_wdir'] > 3, 'Wdir_flag'] = 3

    # wind speed flags + percentage 
    dsFlags['nr_Wspeed'] = CountFlags(ds_PROC, 'Wspeed')
    dsFlags['percent_Wspeed'] = 100 * dsFlags['nr_Wspeed'] / len(ds_PROC.Wspeed)
    # wind speed max flags + percentage 
    dsFlags['nr_Wspeed_Max'] = CountFlags(ds_PROC, 'Wspeed_Max')
    dsFlags['percent_Wspeed_Max'] = 100 * dsFlags['nr_Wspeed_Max'] / len(ds_PROC.Wspeed_Max)
    # wind direction flags + percentage 
    dsFlags['nr_Wdir'] = CountFlags(ds_PROC, 'Wdir')
    dsFlags['percent_Wdir'] = 100 * dsFlags['nr_Wdir'] / len(ds_PROC.Wdir)

    # drop the extra wind processing columns:
    ds_PROC.drop(columns=['Wspeed_0', 'consecutive', 'shift', 'Wdiff', 'Wdir_bin', 'consecutive_wdir'], inplace=True)

    # RADIATION SW
    # Set general flags for SW radiation:
    ds_PROC['SWin_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'SWin_Avg_flag', 'SWin_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'SWin_Avg_flag', 'SWin_Avg', 0, 2000)
    # flag instances where SWout > SWin: 4
    ds_PROC.loc[ds_PROC['SWout_Avg'] > ds_PROC['SWin_Avg'], 'SWin_Avg_flag'] = 4   # modified + CHECK WITH LEA/MARTIN: just unphysical?

    # SWin flags + percentage 
    dsFlags['nr_SWin'] = CountFlags(ds_PROC, 'SWin_Avg')
    dsFlags['percent_SWin'] = 100 * dsFlags['nr_SWin'] / len(ds_PROC.SWin_Avg)

    ds_PROC['SWout_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'SWout_Avg_flag', 'SWout_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'SWout_Avg_flag', 'SWout_Avg', 0, 2000)
    # flag instances where SWout > SWin: 4
    ds_PROC.loc[ds_PROC['SWout_Avg'] > ds_PROC['SWin_Avg'], 'SWout_Avg_flag'] = 4   # modified + CHECK WITH LEA/MARTIN: just unphysical?

    # SWout flags + percentage 
    dsFlags['nr_SWout'] = CountFlags(ds_PROC, 'SWout_Avg')
    dsFlags['percent_SWout'] = 100 * dsFlags['nr_SWout'] / len(ds_PROC.SWout_Avg)

    # RADIATION LW
    # Correction with sensor temperature: (CHECK W MARTIN)
    ds_PROC['LWin_Cor'] = ds_PROC['LWin_Avg'] + (5.67*10**-8 * ds_PROC['NR01K_Avg']**4)
    ds_PROC['LWout_Cor'] = ds_PROC['LWout_Avg'] + (5.67*10**-8 * ds_PROC['NR01K_Avg']**4)

    # Set general flags for LW radiation:
    ds_PROC['LWin_Cor_flag'] = 0
    # flag system error value: 1, flag no data values: 2
    ds_PROC = flagSysError(ds_PROC, 'LWin_Cor_flag', 'LWin_Cor')
    # flag values outside of sensor range: 3
    ds_PROC = flagRange(ds_PROC, 'LWin_Cor_flag', 'LWin_Cor', 0, 1000)

    ds_PROC['LWout_Cor_flag'] = 0
    # flag system error value: 1, flag no data values: 2
    ds_PROC = flagSysError(ds_PROC, 'LWout_Cor_flag', 'LWout_Cor')
    # flag values outside of sensor range: 3
    ds_PROC = flagRange(ds_PROC, 'LWout_Cor_flag', 'LWout_Cor', 0, 1000)
    # flag instances where LWout > 315.6574 W/m2, i.e. physical limit of melting surface: 5
    ds_PROC.loc[ds_PROC['LWout_Cor'] > 315.6574, 'LWout_Cor_flag'] = 5

    # LWin flags + percentage 
    dsFlags['nr_LWin'] = CountFlags(ds_PROC, 'LWin_Cor')
    dsFlags['percent_LWin'] = 100 * dsFlags['nr_LWin'] / len(ds_PROC.LWin_Cor)
    # LWout flags + percentage 
    dsFlags['nr_LWout'] = CountFlags(ds_PROC, 'LWout_Cor')
    dsFlags['percent_LWout'] = 100 * dsFlags['nr_LWout'] / len(ds_PROC.LWout_Cor)
    
    # ICE TEMPERATURE
    # EisT1:
    ds_PROC['EisT1_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'EisT1_Avg_flag', 'EisT1_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'EisT1_Avg_flag', 'EisT1_Avg', -40, 85)

    # EisT1 flags + percentage 
    dsFlags['nr_EisT1'] = CountFlags(ds_PROC, 'EisT1_Avg')
    dsFlags['percent_EisT1'] = 100 * dsFlags['nr_EisT1'] / len(ds_PROC.EisT1_Avg)
    
    # EisT2:
    ds_PROC['EisT2_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'EisT2_Avg_flag', 'EisT2_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'EisT2_Avg_flag', 'EisT2_Avg', -40, 85)

    # EisT1 flags + percentage 
    dsFlags['nr_EisT2'] = CountFlags(ds_PROC, 'EisT2_Avg')
    dsFlags['percent_EisT2'] = 100 * dsFlags['nr_EisT2'] / len(ds_PROC.EisT2_Avg)
    
    # EisT3:
    ds_PROC['EisT3_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'EisT3_Avg_flag', 'EisT3_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'EisT3_Avg_flag', 'EisT3_Avg', -40, 85)

    # EisT1 flags + percentage 
    dsFlags['nr_EisT3'] = CountFlags(ds_PROC, 'EisT3_Avg')
    dsFlags['percent_EisT3'] = 100 * dsFlags['nr_EisT3'] / len(ds_PROC.EisT3_Avg)
    
    # EisT4:
    ds_PROC['EisT4_Avg_flag'] = 0
    # flag no data values: 1
    ds_PROC = flagSysError(ds_PROC, 'EisT4_Avg_flag', 'EisT4_Avg')
    # flag values outside of sensor range: 2
    ds_PROC = flagRange(ds_PROC, 'EisT4_Avg_flag', 'EisT4_Avg', -40, 85)

    # EisT1 flags + percentage 
    dsFlags['nr_EisT4'] = CountFlags(ds_PROC, 'EisT4_Avg')
    dsFlags['percent_EisT4'] = 100 * dsFlags['nr_EisT4'] / len(ds_PROC.EisT4_Avg)
    
    # SNOW HEIGHT
    # Correct distance value with air temp: 
    ds_PROC['Dist_Cor'] = ds_PROC['Dist_Avg']*np.sqrt((ds_PROC['Tair_Avg']+273.15)/273.15)
    # Subtract sensor height:


    ds_PROC['Snow'] = ds_PROC['sr_hy'] - ds_PROC['Dist_Cor']



    # Compute daily mean and rsample back to 10 min frequency
    delta_tr = 0.1
    d_snow = ds_PROC['Snow'].resample('24h').mean()
    ds_PROC['snow_d'] = d_snow.resample('10min').ffill()
    # create a flag for snow quality: set to 1 when the difference between single observation and daily mean is higher than threshold
    ds_PROC['Snow_flag'] = 0
    ds_PROC.loc[abs(ds_PROC['snow_d']-ds_PROC['Snow']) > delta_tr, 'Snow_flag'] = 1
    ds_PROC.loc[ds_PROC['Snow'] < 0, 'Snow_flag'] = 2 # negative snow height
    # ds_PROC['Snow'] = ds_PROC.Snow.where(ds_PROC['Snow_flag'] == 0)
    # ds_PROC['Surf'] = ds_PROC.Snow.where(ds_PROC['Snow_flag'] == 0)


    return(ds_PROC, dsFlags)


#%% Section 3: Data processing
# WSS read in data
ds_WSS_resample, dsWSS_original = readfile(filepath, start, end)

# write raw data to csv
# dsWSS_original = dsWSS_original[["RECORD","Batt_Min","Tair_Avg","Hum_Avg","Press_Avg","SWin_Avg","SWout_Avg","LWin_Avg","LWout_Avg","NR01K_Avg","Wspeed","Wdir","Wspeed_Max","Dist_Avg","EisT1_Avg","EisT2_Avg","EisT3_Avg","EisT4_Avg"]]
#writetofile(dsWSS_original)

# Flag
dsWSS_CORR, dsFlags_WSS = processData(ds_WSS_resample, sr_acc)

# snow_og =dsWSS_CORR['Snow']
# print(snow_og)
# # stop

# dsWSS_CORR['Surf'] = dsWSS_CORR['Snow']

# dsWSS_CORR.Snow.loc[dsWSS_CORR.Snow_flag>0]=np.nan 
# # allow negative values for surface:
# dsWSS_CORR.Surf.loc[dsWSS_CORR.Snow_flag==1]=np.nan 



# manually remove outlier values with unrealistic amount of surface change or unrealistic surface values
# dsWSS_CORR.Surf.loc[abs(dsWSS_CORR.Surf)>0.8]=np.nan 
# dsWSS_CORR.Surf.loc[abs(dsWSS_CORR.Surf.diff())>0.01]=np.nan 
# smooth the surface: 
# dsWSS_CORR['Surf'].loc[abs(dsWSS_CORR['Surf'].rolling('24h').mean()-dsWSS_CORR['Surf']) > 0.05] = np.nan


flags = ['Batt_Min_flag', 'Tair_Avg_flag', 'Hum_Avg_flag', 'Press_Avg_flag', 'Wdir_flag', 'Wspeed_flag',
         'Wspeed_Max_flag', 'SWin_Avg_flag', 'SWout_Avg_flag', 'LWin_Cor_flag', 'LWout_Cor_flag']
         #'Snow_flag']

# make additional DF with flagged values set to nan:
# set data with flags to np.nan
dsWSS_CORR2 = dsWSS_CORR.copy()
for flag in flags:
    dsWSS_CORR2.loc[(dsWSS_CORR2[flag] != 0), flag[:-5]] = np.nan

def smoothSnow(ds):
    tr = 0.02
    ds['Snow_Avg_Corr'] = ds['sr_hy'] - ds['Dist_Cor']
    ds.Snow_Avg_Corr.loc[ds.Snow_Avg_Corr.diff().abs() > tr ] = np.nan

    # compute 48h rolling mean and reindex to 10 minutes
    d_snow = ds['Snow_Avg_Corr'].rolling('48h', min_periods=1, center=True).mean()
    ds['snow_d'] = d_snow.resample('10min').ffill()

    ds.loc[abs(ds['snow_d']-ds['Snow_Avg_Corr']) > 0.15, 'snow_d'] = np.nan

    lmin, lmax = hl_envelopes_idx(ds['Snow_Avg_Corr'].values)

    ds2 = pd.DataFrame(index=ds.index.values[lmin])
    ds2['Snow_lm'] = ds['Snow_Avg_Corr'].values[lmin]
    lmin2, lmax2 = hl_envelopes_idx(ds2['Snow_lm'].values)

    ds3 = pd.DataFrame(index=ds2.index.values[lmin2])
    ds3['Snow_lm'] = ds2['Snow_lm'].values[lmin2]

    ds4 = ds3.reindex(ds.index).interpolate(method='linear', limit_direction ='both')
    ds['snow_d3'] = ds4['Snow_lm']
    d4 = ds['snow_d3'].rolling('24h', center=True).mean()
    ds['snow_24roll'] = d4#.resample('H').mean().ffill()
    print(ds.head())
    # stop
    return(ds)


dsWSS_CORR = smoothSnow(dsWSS_CORR)

dsWSS_CORR2.loc[dsWSS_CORR2['SWout_Avg'] < 2, 'SWout_Avg'] = np.nan
dsWSS_CORR2.loc[dsWSS_CORR2['SWin_Avg'] < 2, 'SWin_Avg'] = np.nan

# daily albedo!
albedo_daily = dsWSS_CORR2['SWout_Avg'].resample('D').sum() / dsWSS_CORR2['SWin_Avg'].resample('D').sum()
dsWSS_CORR2['Albedo'] = albedo_daily.resample('H').mean().ffill()
dsWSS_CORR2.Albedo.loc[dsWSS_CORR2.Albedo > 1] = np.nan


dsWSS_CORR2['Precipitation'] = 0

dsWSS_CORR2['Surf_step1'] = dsWSS_CORR.snow_d3
dsWSS_CORR2['Surf'] = dsWSS_CORR.snow_24roll
dsWSS_CORR2['Snowfall'] = dsWSS_CORR2.Surf.diff()
dsWSS_CORR2.loc[dsWSS_CORR2['Snowfall']<0, 'Snowfall'] = 0


def adjustalbedo(start, end, df1, target):
    df = df1.copy()
    val = df.Albedo.loc[df.index == pd.to_datetime(target)].mean()
    df.loc[start:end, 'Albedo'] = val
    return (df)


def snowplot(dsWSS_CORR, dsWSS_CORR2):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax = ax.flatten()
    dsWSS_CORR = dsWSS_CORR.loc['2021-08-01 00:00':'2021-09-01 00:00']

    sn = dsWSS_CORR['sr_hy'] - dsWSS_CORR['Dist_Cor']
    ax[0].scatter(sn.index, sn.values, color='red', s=1, label='SR50, uncorrected')

    ax[0].scatter(dsWSS_CORR2.index, dsWSS_CORR2.Surf_step1, color='black', s=2, label='SR50, lower envelope')


    ax[1].plot(dsWSS_CORR2.index, dsWSS_CORR2.Surf_step1, color='black', label='surface, not smoothed')
    ax[1].plot(dsWSS_CORR2.index, dsWSS_CORR2.Surf, color='grey', label='surface, 24h rolling')
    ax1 = ax[1].twinx()

    dailySnow = dsWSS_CORR2.Snowfall.resample('d').sum() *100
    ax1.bar(dailySnow.index, dailySnow.values, color='skyblue', label='daily snowfall sum')

    day1 = pd.to_datetime('2021-08-01')
    day2 = pd.to_datetime('2021-08-30')
    ax[0].set_xlim([day1, day2])

    ax[0].set_ylim([-0.5, 2.5])
    ax[0].set_ylabel('Surface height (m)')

    ax[1].set_ylabel('Surface height (m)')
    ax1.set_ylabel('Snowfall (cm)')

    ax[0].legend(ncols=2, loc='upper left', bbox_to_anchor=(0.2, 1.2))
    ax[1].legend(loc='upper center')
    ax[1].set_ylim([-0.5, 0.35])
    ax1.set_ylim([0, 17])

    ax[1].legend()

    dsWSS_CORR2.Surf_step1.resample('H').mean().to_csv('out/cleaned_surface_hourly.csv')

    fig.savefig('figs/snowcorrection_2.png', bbox_inches='tight', dpi=200)

    # --------


def snowplotAll(dsWSS_CORR, dsWSS_CORR2):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax = ax.flatten()

    sn = dsWSS_CORR['sr_hy'] - dsWSS_CORR['Dist_Cor']
    ax[0].scatter(sn.index, sn.values, color='red', s=1, label='SR50, uncorrected')

    ax[0].scatter(dsWSS_CORR2.index, dsWSS_CORR2.Surf_step1, color='black', s=1, label='SR50, lower envelope')

    ax[1].plot(dsWSS_CORR2.index, dsWSS_CORR2.Surf, color='grey', label='surface, 24h rolling mean')
    ax1 = ax[1].twinx()
    ax1.plot(dsWSS_CORR2.Tair_Avg.resample('D').mean().rolling(7).mean(), color='orange', label='Air temperature (7 day rolling mean)')


    ax[0].set_ylim([-2.0, 2.6])
    ax[0].set_ylabel('Surface height (m)')

    ax[1].set_ylabel('Surface height (m)')
    ax1.set_ylabel('Air temperature (°C)')

    ax[0].legend(ncols=2, loc='upper left', bbox_to_anchor=(0.2, 1.2))
    
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.2, 1.18))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.7, 1.18))
    # ax[1].set_ylim([-0.6, 2.5])

    day1 = pd.to_datetime('2018-01-01')
    day2 = pd.to_datetime('2024-12-31')
    ax[0].set_xlim([day1, day2])
    fig.savefig('figs/snow_allyears.png', bbox_inches='tight', dpi=200)

    # --------


def snowplot_seasonal(dsWSS_CORR, dsWSS_CORR2):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax = ax.flatten()


    # for i, y in enumerate(dsWSS_CORR2.index.year.unique()):
    for i, y in enumerate([2020, 2022, 2024]):
        ds = dsWSS_CORR2.loc[dsWSS_CORR2.index.year==y]
        ds = ds.loc[~((ds.index.month==2) & (ds.index.day==29))]
        # ds['doy'] = ds.index.dayofyear
        ds.index = ds.index.map(lambda t: t.replace(year=1900))
        ds_daily = ds.resample('D').mean()
        ds_daily['Tair_Avg'] = ds['Tair_Avg'].resample('D').min()

        ax[0].plot(ds_daily.index, ds_daily.Surf_step1, label=y)

        # ax[1] = ax[0].twinx()
        ax[1].plot(ds_daily.index, ds_daily.Tair_Avg, label=y)

    day1 = pd.to_datetime('1900-06-01')
    day2 = pd.to_datetime('1900-09-30')
    ax[0].set_xlim([day1, day2])
    ax[1].set_ylim([-6, 6])
    ax[0].legend()
    ax[1].legend()
    fig.savefig('figs/snowcorrection_4.png', bbox_inches='tight', dpi=200)


snowplot(dsWSS_CORR, dsWSS_CORR2)
snowplotAll(dsWSS_CORR, dsWSS_CORR2)
#snowplot_seasonal(dsWSS_CORR, dsWSS_CORR2)

plt.show()

