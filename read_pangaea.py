# ! /usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import geopandas as gpd
import csv as cf
import glob


# supress copy warning - careful 
pd.options.mode.chained_assignment = None  # default='warn'

# set file path to stake data file - pangaea version:
stakesfile = '/Users/leahartl/Desktop/WSS/WSS_Albedo/data/WSS_ablation_2017-2022.tab'
folder = '/Users/leahartl/Desktop/WSS/WSS_Albedo/data/'

# the file has a header added by pangaea that needs to be removed before further processing:
# this allows for multiple .tab files in the directory. At present there is just one.
def loadfiles(folder):
    dfs = []
    fls = glob.glob(folder+'*.tab')
    # this is a loop to identify the header line in the pangaea format.
    for f in fls:
        with open(f, 'r') as fin:
            reader = cf.reader(fin)
            for idx, row in enumerate(reader):
                if row[0].startswith('*/'):
                    headerline = idx+1
                    print(row)
                    print(idx)
                    print('header found')
   
        data = pd.read_csv(f, header = headerline, parse_dates=True, delimiter='\t', index_col=0)
        dfs.append(data)

    data_merged = pd.concat(dfs).sort_index()
    data_merged.reset_index(inplace=True)
    return(data_merged)


data = loadfiles(folder)

# rename cols for easier handling:
data.columns = ['name','date0','date1','period','snow_height_cm','ice_ablation_cm','ice_ablation_mmwe','X','Y','Z','position_quality']
print(data.head())
print(data.columns)
# write to csv as reality check 
data.to_csv('data/stakes_pangaea.csv')


