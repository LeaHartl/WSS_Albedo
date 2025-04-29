# ! /usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import geopandas as gpd

from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from PIL import Image
# from shapely.geometry import Point
# import glob
import contextily as cx
import cartopy.crs as ccrs

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
from rasterio.plot import show
from matplotlib_scalebar.scalebar import ScaleBar

from shapely.geometry import box
from shapely.geometry import Polygon

from rio_color.operations import sigmoidal

# supress copy warning - careful 
pd.options.mode.chained_assignment = None  # default='warn'

def add_north_arrow(ax):

  ax.annotate("",  xy=(0.065, 0.22), xytext=(0.065, 0.08),xycoords="axes fraction",
          arrowprops=dict(arrowstyle="simple", facecolor="black"))


def overview5(GI5, AWS, stakes, contours, countries, RGI, cam):

    # set file names:
    dem_f = '/Users/leahartl/Desktop/WSS/Gepatsch2017_clip.tif'
    satraster = '/Users/leahartl/Desktop/WSS/examples/2017_08_25.tiff'
    satrasterLarge = '/Users/leahartl/Desktop/WSS/examples/2021_08_14.tiff'

    # ensure correct crs:
    stakes.to_crs(epsg=31254, inplace=True)
    cam.to_crs(epsg=31254, inplace=True)
    AWS.to_crs(epsg=31254, inplace=True)
    AWS_latlon = AWS.to_crs(epsg=4326)

    contours.to_crs(epsg=31254, inplace=True)
    contours.to_crs(epsg=31254, inplace=True)

    GI5.to_crs(epsg=31254, inplace=True)

    # prep figures:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), sharey=True, sharex=True)
    
    # axis for sat image:
    axins2 = ax.inset_axes([-1.4, -0.15, 1.2, 1.3])
    
    # axis for country overview:
    axins3 = ax.inset_axes([-1.3, 0.8, 0.6, 0.6])

    # -------------------------------
    # photo AWS:
    axins4 = ax.inset_axes([-1.4, -1.21, 1.2, 1.3])
    imgAWS = np.asarray(Image.open('/Users/leahartl/Desktop/WSS/figs/20230819_103402_crop.jpg'))
    axins4.imshow(imgAWS)

    # photo webcam:
    axins5 = ax.inset_axes([-0.1, -1.21, 1.2, 1.3])
    imgCam = np.asarray(Image.open('/Users/leahartl/Desktop/WSS/figs/weissseespitze-2023-08-19-1200_hu_AWS.jpg'))
    axins5.imshow(imgCam)

    # -------------------------------
    # plot sat image
    with rio.open(satrasterLarge) as src3:
        satL = src3.read()
        show(sigmoidal(satL, 6, 0.25), transform=src3.transform, ax=axins2)
        bounds  = src3.bounds

    # plot box:
    geom = box(*bounds)
    df_box = gpd.GeoDataFrame({"id":1,"geometry":[geom]})
    df_box.set_crs(src3.crs, inplace=True)

    axins2.set_xlim([27500, 37500])
    axins2.set_ylim([185000, 194000])
    # add glacier outline:
    GI5.boundary.plot(ax=axins2, alpha=1, color='orange', linewidth=2)

    # -------------------------------
    # plot hillshade:
    # Open the DEM with Rasterio
    with rio.open(dem_f) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan
        hillshade = es.hillshade(elevation)
        show(hillshade, transform=src.transform, ax=ax, cmap='Greys_r', alpha=1)

    # ----------------------------------    
    # plot overview panel (countries)
    axins3.set_xlim([8.5, 18.5])
    axins3.set_ylim([46.1, 49.5])

    countries.boundary.plot(ax=axins3, alpha=1, color='black', linewidth=0.5)
    # RGI.plot(ax=axins3, alpha=1, color='skyblue')

    df_box.to_crs(countries.crs, inplace=True)
    df_box.plot(ax=axins3, alpha=1, color='red', linewidth=4)

    axins3.annotate("AT",
             xy=(14.4, 47.5), xycoords='data', fontsize=18,
             #xytext=(x2, y2), textcoords='data',
             ha="center", va="center")
    axins3.annotate("IT",
             xy=(11.11, 46.4), xycoords='data', fontsize=18,
             #xytext=(x2, y2), textcoords='data',
             ha="center", va="center")
    axins3.annotate("GER",
             xy=(11.0, 48.8), xycoords='data', fontsize=18,
             #xytext=(x2, y2), textcoords='data',
             ha="center", va="center")


    # -------- general figure settings -----------------------
    # make handles for legend:
    lns = []
    AWS.plot(ax=ax, alpha=1, color='red', marker='*', markersize=40)
    patch1 = Line2D([0], [0], marker='*', linestyle='None', label='AWS', color='red', markersize=8, zorder=10)

    cam.plot(ax=ax, alpha=1, color='orange', marker='s', markersize=40)
    patch4 = Line2D([0], [0], marker='s', linestyle='None', label='camera', color='orange', markersize=8, zorder=20)

    stakes.plot(ax=ax, alpha=1, color='black', marker='o', markersize=28)
    patch2 = Line2D([0], [0], marker='o', linestyle='None', label='stakes', color='k', markersize=8, zorder=10)

    contours.boundary.plot(ax=ax, alpha=1, color='black', linewidth=0.5)
    patch3 = Line2D([0], [0], linestyle='-', label='50 m contours', color='k',)

    minx = 28855
    maxx = 30250
    miny = 189035
    maxy = 190476
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])

    lon_lat_list = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]

    polygon_geom = Polygon(lon_lat_list)
    box1 = gpd.GeoDataFrame(index=[0], crs='epsg:31254', geometry=[polygon_geom])       
    box1.boundary.plot(ax=axins2, alpha=1, color='red', linewidth=4)

    # annotations
    # panel labels:
    fs = 20
    axins3.text(8.6, 49.1,'a', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    axins2.text(37100, 193050,'b', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    ax.text(28940, 190400,'c', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    axins4.text(200, 200,'d', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    axins5.text(200, 200,'e', fontsize=fs,
    bbox=dict(boxstyle="square,pad=0.3",fc="lightgrey", ec="grey", lw=2))

    # contours:
    ax.text(29600, 189200,'3400 m.a.s.l.', fontsize=14,
    bbox=dict(boxstyle="square,pad=0.1",fc="white", ec="white", lw=2))

    ax.text(29250, 189400,'3450 m.a.s.l.', fontsize=14,
    bbox=dict(boxstyle="square,pad=0.1",fc="white", ec="white", lw=2))

    # axis settings:
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    axins2.set_xticklabels([])
    axins2.set_yticklabels([])
    axins2.set_xticks([])
    axins2.set_yticks([])

    axins3.set_xticklabels([])
    axins3.set_yticklabels([])
    axins3.set_xticks([])
    axins3.set_yticks([])

    axins4.set_xticklabels([])
    axins4.set_yticklabels([])
    axins4.set_xticks([])
    axins4.set_yticks([])

    axins5.set_xticklabels([])
    axins5.set_yticklabels([])
    axins5.set_xticks([])
    axins5.set_yticks([])

    # add scale bars:
    axins2.add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 14}))
    ax.add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 14}))

    # add manual symbols to auto legend
    patch5 = Line2D([0], [0], linestyle='-', label='glacier outline 2017', color='orange',)
    lns = [patch5, patch3, patch1, patch2, patch4]

    handles = (lns)

    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.5, 0.9), ncol=2, fontsize=18)
    fig.savefig('figs/map_summit_zoomed3.png', dpi=150, bbox_inches='tight')


GI5g = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/mergedGI5_3.shp')
print(GI5g)
Gepatsch = GI5g.loc[GI5g['Gletschern']=='Gepatsch Ferner']

stakes = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
stakes.to_crs(epsg=32632, inplace=True)

# separate AWS from stakes:
AWS = stakes.loc[stakes.name == 'AWS']
stakes = stakes.loc[stakes.name != 'AWS']

cam = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/cam_position.shp')
cam.to_crs(epsg=32632, inplace=True)

contours = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/ContourPolygons2017_Large.shp')
contours.to_crs(epsg=32632, inplace=True)

RGI_otz = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/RGI60_onlyOetz.shp')
RGI = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp')

countries = gpd.read_file('/Users/leahartl/Desktop/LTER/MWKVK_processing/misc/ne_10m_admin_0_countries') 


overview5(Gepatsch, AWS, stakes, contours, countries, RGI, cam)

plt.show()


