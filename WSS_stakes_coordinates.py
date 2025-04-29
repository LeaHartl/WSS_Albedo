# ! /usr/bin/env python3
import numpy as np
import pandas as pd
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

# import fontawesome as fa
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
import rasterio as rio
from rasterio.plot import show
from matplotlib_scalebar.scalebar import ScaleBar

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
# from matplotlib_scalebar.scalebar import ScaleBar

from shapely.geometry import Polygon, LineString, Point

import matplotlib.patheffects as pe

# supress copy warning - careful 
pd.options.mode.chained_assignment = None  # default='warn'


# load files and process data to generate centroid locations of the stakes and AWS.
# file with checked stake data (ablation and position information)
stakedata = '/Users/leahartl/Desktop/WSS/Re__Updates_WSS_paper/WSS_stakes_point_mass_balance.csv'

stakes = pd.read_csv(stakedata, parse_dates=True)
stakes['date1'] = pd.to_datetime(stakes['date1'])
stakes_gdf = gpd.GeoDataFrame(stakes, crs="EPSG:31254", geometry=gpd.points_from_xy(stakes.X, stakes.Y))
stakes_gdf.dropna(subset=['X'], inplace=True)
stakes_gdf = stakes_gdf[['date1', 'name', 'geometry']]

AWS_points = '/Users/leahartl/Desktop/WSS/process_stakes/sub_AWS.geojson'
AWS_pts = gpd.read_file(AWS_points)
# AWS_pts = AWS_pts.loc[AWS_pts['position_quality']==False]
AWS_pts['Name'] = 'AWS'
AWS_pts.dropna(subset=['X'], inplace=True)
AWS_pts = AWS_pts[['Datum', 'Name', 'geometry']]
AWS_pts.columns = ['date1', 'name', 'geometry']

AWS_tofile = AWS_pts.copy()
AWS_tofile['date1'] = AWS_tofile['date1'].astype(str)
AWS_tofile.to_file('/Users/leahartl/Desktop/WSS/outlines/AWS_pts_all.shp')

#merge AWS with stake positions:
both = pd.concat([stakes_gdf, AWS_pts])

both.dropna(subset=['geometry'], inplace=True)
print(both)

# get values from summer (july-October) readings and compute centroids, save to file:
summer = both.loc[(both.date1.dt.month < 11) & (both.date1.dt.month > 6)]
                                                                       
meanpos = summer.dissolve(by='name').centroid                                                
print(meanpos)
meanpos.to_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')

rmselist = []
stakelist = []
for sn in summer['name'].unique():
    stakelist.append(sn)
    tmp = summer.loc[summer.name==sn]

    point = Point(meanpos.loc[meanpos.index==sn].geometry.x, meanpos.loc[meanpos.index==sn].geometry.y)
    dist = tmp.distance(point)
    print(dist)
    meanSquaredError = ((dist.values) ** 2).mean()
    rmse = np.sqrt(meanSquaredError)
    rmselist.append(rmse)

print(stakelist)
print(rmselist)



def coordinates_fig(summer, meanpos):

    # load background imagery for figure (hillshade and RGB S2 scene)
    dem_f = '/Users/leahartl/Desktop/WSS/Gepatsch2017_clip.tif'
    satraster = '/Users/leahartl/Desktop/WSS/examples/2021_08_14.tiff'
    # make figure: 
    fig, ax = plt.subplots(1,1, figsize=(12, 5))

    axins = ax.inset_axes([0.01, 1.0, 0.7, 0.8])

    with rio.open(satraster) as src2:
        sat = src2.read()
        show(sat, transform=src2.transform, ax=ax, alpha=1)
        show(sat, transform=src2.transform, ax=axins, alpha=1)

    minx = 29250
    maxx = 29400
    miny = 189720
    maxy = 189800
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])

    axins.set_xlim([29000, 29800])
    axins.set_ylim([189600, 190000])

    summer2 = summer.loc[summer.name != 'g']
    summer2.plot(ax=ax, markersize=4, color='k')

    # a b c d H
    #skip g since we don't really use it in the paper (short time series)
    clrs = ['blue', 'orange', 'green', 'purple', 'cyan', 'pink', 'grey', 'red'] #,'olive'
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'AWS'] #,'g'

    patches = []

    for j, stn in enumerate(points):
        # plot the stake centroids
        mp = meanpos.loc[meanpos.index == stn]
        mp.plot(ax=ax, markersize=28, color=clrs[j], edgecolor='white', linewidth=0.2)
        mp.plot(ax=axins, markersize=28, color=clrs[j], edgecolor='white', linewidth=0.2)

        # show buffer on the plot:
        mp.geometry.buffer(5).boundary.plot(ax=ax, edgecolor='k', linestyle='--', linewidth=0.5)

        x = mp.geometry.x
        y = mp.geometry.y
        label = stn.upper()
        ax.annotate(label, xy=(x, y), xytext=(-2, 15), textcoords="offset points", fontsize=22, color=clrs[j],
            path_effects=[pe.withStroke(linewidth=1, foreground="white")])

        patch = Line2D([0], [0], marker='o', linestyle='None', label=label, color=clrs[j], markersize=8, zorder=10)
        patches.append(patch)

    ax.add_artist(ScaleBar(dx=1, location="lower left"))
    axins.add_artist(ScaleBar(dx=1, location="lower left"))

    lon_lat_list = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]

    polygon_geom = Polygon(lon_lat_list)
    box1 = gpd.GeoDataFrame(index=[0], crs='epsg:31254', geometry=[polygon_geom])       
    box1.boundary.plot(ax=axins, alpha=1, color='red', linewidth=2)

    axins.set_xticks([29000, 29400, 29800])
    axins.set_yticks([189700, 189900])

    cam = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/cam_position.shp')
    cam.to_crs(epsg=31254, inplace=True)
    cam.plot(ax=axins, markersize=28, color='lime', edgecolor='white', linewidth=0.2)

    campatch = Line2D([0], [0], marker='o', linestyle='None', label='camera', color='lime', markersize=8, zorder=10)

    smallpatch = Line2D([0], [0], marker='o', linestyle='None', label='coordinates', color='k', markersize=4, zorder=10)

    bufferpatch = Line2D([0], [0], marker='None', linestyle='--', color='k', label='buffer', markersize=12)

    patches.append(campatch)
    patches.append(smallpatch)
    patches.append(bufferpatch)

    fig.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.01, 0.9), ncol=2, fontsize=18)

    fs = 20
    ax.text(29390, 189790,'b', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    axins.text(29780, 189920,'a', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    fig.savefig('figs/stake_positions_S2_2024.png', dpi=200, bbox_inches='tight')

coordinates_fig(summer, meanpos)

plt.show()

