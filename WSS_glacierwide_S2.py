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
from matplotlib import gridspec
from matplotlib.pyplot import cm

import xarray as xr
import rasterio

from rasterio.plot import show
import rioxarray
from shapely.geometry import box
from shapely.geometry import Polygon

from matplotlib_scalebar.scalebar import ScaleBar
from rio_color.operations import sigmoidal

# supress copy warning - careful 
pd.options.mode.chained_assignment = None  # default='warn'

def pct_clip(xrdata,pct=[1,99]):#94]):
    #print(np.nanmax(array))
    #if np.nanmax(array)<0.9:
    array = xrdata.values
    array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
    # if np.nanmax(array)>=0.9:
        # array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,80)
    
    clip = (array - array_min) / (array_max - array_min)
    clip[clip>1]=1
    clip[clip<0]=0
    return clip


def make_rgb_rio_constant(s2tif, brightness=2):

    s2_clip = s2tif #.rio.clip(gdf2.geometry.values, gdf2.crs)
    # print(s2_clip)

    # stop
    green = s2_clip[1]#rioxarray.open_rasterio(g, masked=True)
    red = s2_clip[2]
    blue = s2_clip[0]
    # print(blue)
    del blue.attrs['long_name']
    del green.attrs['long_name']
    del red.attrs['long_name']

    # print(blue)
    blue.rio.to_raster("blue_clip.tif", compress='LZMA', tiled=True, dtype="int32")
    green.rio.to_raster("green_clip.tif", compress='LZMA', tiled=True, dtype="int32")
    red.rio.to_raster("red_clip.tif", compress='LZMA', tiled=True, dtype="int32")


    with rasterio.open("blue_clip.tif") as src:
        with rasterio.open(
            'RGB_Temp2_plot.tif', 'w+',
            driver='GTiff',
            dtype= rasterio.float32,
            count=3,
            crs = src.crs,
            width=src.width,
            height=src.height,
            transform=src.transform,
        ) as dst:
            # V = pct_clip(src.read(1))
            rd = rasterio.open("red_clip.tif") 
            gn = rasterio.open("green_clip.tif") 
            # bl = rasterio.open() 
            V = pct_clip(red.squeeze())
            dst.write(V, 1)
            V = pct_clip(green.squeeze())
            dst.write(V, 2)
            V = pct_clip(blue.squeeze())
            dst.write(V, 3)

    return 'test'


# histograms
def histograms(GI5, RGI, dts, polycontours):
    fig, ax = plt.subplots(4, 2, figsize=(8, 10), sharey=True, sharex=True)
    ax = ax.flatten()
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharey=True)
    for i, d in enumerate(dts):
        clrs = cm.tab10(range(7))
        fl = '/Users/leahartl/Desktop/WSS/ProcessS2/S2_10px_WSS/'+d+'.tif'
        S2 = rioxarray.open_rasterio(fl, masked=True)
        S2 = S2.to_dataset('band')
        S2 = S2.rename({i + 1: name for i, name in enumerate(S2.attrs['long_name'])})

        ND = S2['NIR'] / S2['SWIR1']
        ND2 = S2['Red'] / S2['SWIR1']
    
        S2_masked1 = S2.where(ND >= 1)
        S2_masked = S2.where(ND2 >= 2)
        albedo= 0.356 * S2_masked['Blue'] + 0.130 * S2_masked['Red'] +0.373 * S2_masked['NIR'] + 0.085*S2_masked['SWIR1'] + 0.072*S2_masked['SWIR2'] -0.0018
        albedo_clip = albedo.rio.clip(GI5.geometry.values, GI5.crs)
        counts, bins = np.histogram(albedo_clip.to_numpy().ravel(), bins=np.arange(0, 1, 0.05))
        area_frac = 100 * (counts) / np.count_nonzero(~np.isnan(albedo_clip))
        ax[i].stairs(area_frac, bins, color='k', zorder=2)#, color=clrs[i], label='20'+d)
        print(bins)
        print(counts)
        vdark = bins[bins<=0.15]
        print(vdark)
        # stop
        avg = albedo_clip.mean().round(2).to_numpy()
        # print(albedo_clip.median().round(2).to_numpy())
        ax[i].stairs(area_frac[:3], bins[:4], color='r', zorder=4, fill=True, alpha=0.5)
        ax[i].annotate('Area <= 0.15: '+ str(sum(area_frac[:3]).round(1)) + ' %', xy=(0.22, 0.68), xycoords='axes fraction', color='k')#, fontweight='bold', bbox=dict(facecolor='none', edgecolor='k'))
        ax[i].annotate('Mean albedo: '+ str(avg), xy=(0.22, 0.8), xycoords='axes fraction', color='k')
        ax[i].grid('both', zorder=0)
        ax[i].set_xlim(0, 1)
        ax[i].set_title('20'+d)


    ax[0].set_ylabel('Glacier area fraction (%)')
    ax[2].set_ylabel('Glacier area fraction (%)')
    ax[4].set_ylabel('Glacier area fraction (%)')
    ax[6].set_ylabel('Glacier area fraction (%)')
    # ax[3].set_xlabel('S2-derived albedo')
    ax[5].set_xlabel('S2-derived albedo')
    ax[6].set_xlabel('S2-derived albedo')
    ax[7].axis("off")
    fig.savefig('figs/histograms.png', dpi=200, bbox_inches='tight')


def histograms_poster(GI5, RGI, dts, polycontours):
    fig, ax = plt.subplots(2, 4, figsize=(12, 8), sharey=True, sharex=True)
    ax = ax.flatten()
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharey=True)
    for i, d in enumerate(dts):
        clrs = cm.tab10(range(7))
        fl = '/Users/leahartl/Desktop/WSS/ProcessS2/S2_10px_WSS/'+d+'.tif'
        S2 = rioxarray.open_rasterio(fl, masked=True)
        S2 = S2.to_dataset('band')
        S2 = S2.rename({i + 1: name for i, name in enumerate(S2.attrs['long_name'])})

        ND = S2['NIR'] / S2['SWIR1']
        ND2 = S2['Red'] / S2['SWIR1']
    
        S2_masked1 = S2.where(ND >= 1)
        S2_masked = S2.where(ND2 >= 2)
        albedo= 0.356 * S2_masked['Blue'] + 0.130 * S2_masked['Red'] +0.373 * S2_masked['NIR'] + 0.085*S2_masked['SWIR1'] + 0.072*S2_masked['SWIR2'] -0.0018
        albedo_clip = albedo.rio.clip(GI5.geometry.values, GI5.crs)
        counts, bins = np.histogram(albedo_clip.to_numpy().ravel(), bins=np.arange(0, 1, 0.05))
        area_frac = 100 * (counts) / np.count_nonzero(~np.isnan(albedo_clip))
        ax[i].stairs(area_frac, bins, color='k', zorder=2)#, color=clrs[i], label='20'+d)
        print(bins)
        print(counts)
        vdark = bins[bins<=0.15]
        print(vdark)
        # stop
        avg = albedo_clip.mean().round(2).to_numpy()
        # print(albedo_clip.median().round(2).to_numpy())
        ax[i].stairs(area_frac[:3], bins[:4], color='r', zorder=4, fill=True, alpha=0.5, label='albedo <= 0.15')
        # ax[i].annotate('Area <= 0.15: '+ str(sum(area_frac[:3]).round(1)) + ' %', xy=(0.22, 0.68), xycoords='axes fraction', color='k', fontsize=14)#, fontweight='bold', bbox=dict(facecolor='none', edgecolor='k'))
        # ax[i].annotate('Mean albedo: '+ str(avg), xy=(0.22, 0.8), xycoords='axes fraction', color='k', fontsize=14)
        ax[i].grid('both', zorder=0)
        ax[i].set_xlim(0, 1)
        ax[i].set_title('20'+d)

    for a in ax:
        a.tick_params(axis='x', labelsize=16)
        a.tick_params(axis='y', labelsize=16)

    ax[0].set_ylabel('Glacier area fraction (%)', fontsize=18)
    ax[0].legend()
    # ax[2].set_ylabel('Glacier area fraction (%)')
    ax[4].set_ylabel('Glacier area fraction (%)', fontsize=18)
    # ax[6].set_ylabel('Glacier area fraction (%)')
    # ax[3].set_xlabel('S2-derived albedo')
    # ax[4].set_xlabel('S2-derived albedo', fontsize=18)
    ax[5].set_xlabel('S2-derived albedo', fontsize=18)
    # ax[6].set_xlabel('S2-derived albedo', fontsize=18)
    ax[7].axis("off")
    fig.savefig('figs/histograms_poster.png', dpi=400, bbox_inches='tight', transparent=True)



# subplots, rgb maps with albedo contours
def subplotsRGBcontours(GI5, ST_gdf, RGI, dts, polycontours):

    df_mean = pd.DataFrame(index=polycontours['ELEV_MIN'].values, columns=dts)
    df_mean = df_mean.reindex(sorted(df_mean.columns), axis=1)
    df_mean = df_mean.sort_index()

    # clrs = cm.tab10(range(6))

    for c in df_mean.columns:
        print(c)
        dat = pd.read_csv('/Users/leahartl/Desktop/WSS/ProcessS2/output/contourstats_20'+c+'.csv', index_col=0)
        dat = dat.sort_index()
        #print(dat)
        # df_mean[c] = dat['median'].values
        df_mean[c] = dat['mean'].values

    fig, ax = plt.subplots(4, 2, figsize=(7, 8))
    ax = ax.flatten()
    # clrs = cm.plasma_r(np.linspace(0, 1, len(dts)))
    # clrs = cm.plasma_r(np.linspace(0.1, 1, len(dts)))
    clrs = cm.plasma_r(np.linspace(0.1, 1, len(dts)))
    fig.subplots_adjust(right=0.65)
    scatter_ax = fig.add_axes([0.7, 0.2, 0.2, 0.7])

    xmin, xmax = 631000, 635000
    ymin, ymax = 5.187e6, 5.19e6
    lon_lat_list = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]

    polygon_geom = Polygon(lon_lat_list)
    box = gpd.GeoDataFrame(index=[0], crs='epsg:31254', geometry=[polygon_geom])       
    

    for i, d in enumerate(dts):
        fl = '/Users/leahartl/Desktop/WSS/ProcessS2/S2_10px_WSS/'+d+'.tif'
        S2 = rioxarray.open_rasterio(fl, masked=True)
        print(S2)
        make_rgb_rio_constant(S2, brightness=4)
        with rasterio.open("RGB_Temp2_plot.tif") as src2:
            # out_image, out_transform = rasterio.mask.mask(src2, [shape], crop=True)
            show(src2.read(),transform=src2.transform,ax=ax[i])
            print(src2.crs)
            if i==4:
                axins=ax[4].inset_axes([1.1, -2.4, 1.5, 1.5])
                axins.set_ylim([ymin, ymax])
                axins.set_xlim([xmin, xmax])
                #ax[4].indicate_inset_zoom(axins, edgecolor="grey")

                show(src2.read(),transform=src2.transform,ax=axins)
                axins.set_axis_off()
                
            

        S2 = S2.to_dataset('band')
        S2 = S2.rename({i + 1: name for i, name in enumerate(S2.attrs['long_name'])})

        ND = S2['NIR'] / S2['SWIR1']
        ND2 = S2['Red'] / S2['SWIR1']

        S2_masked1 = S2.where(ND >= 1)
        S2_masked = S2.where(ND2 >= 2)
        albedo= 0.356 * S2_masked['Blue'] + 0.130 * S2_masked['Red'] +0.373 * S2_masked['NIR'] + 0.085*S2_masked['SWIR1'] + 0.072*S2_masked['SWIR2'] -0.0018
       
        #albedo_clip = albedo.rio.clip(clip.geometry.values, clip.crs)

        #albedo.plot.contour(ax=ax[i], levels=[0.15], alpha=1, colors='r', linewidths=0.2)
        if i==4:
            albedo.plot.contour(ax=axins, levels=[0.15], alpha=1, colors='orange', linewidths=0.8)
            #albedo.plot.contour(ax=axins, levels=[0.20], alpha=1, colors='MediumSpringGreen', linewidths=0.4)
            #alb = albedo.plot.contourf(ax=axins, levels=np.arange(0, 0.6, 0.05), alpha=1, cmap='viridis', linewidths=0.4,
            #    add_colorbar=False)#cbar_kwargs={'shrink':0.8,'pad':0.8})#'location':'bottom', 'anchor':(-0.02, -0.02)})
        
           # cb = alb.colorbar()

        ax[i].set_axis_off()



        ax[i].set_title('20'+d)
        lns = []
        GI5.boundary.plot(ax=ax[i], alpha=1, color='yellow', linewidth=0.4, zorder=10)
        lnGl = Line2D([0], [0], label='outline 2017', color='yellow', linewidth=0.1, linestyle='-')
        lns.append(lnGl)

        ST_gdf.plot(ax=ax[i], alpha=1, color='red', marker='*', markersize=20, zorder=20)
        patch1 = Line2D([0], [0], marker='*', linestyle='None', label='AWS', color='r', markersize=4, zorder=20)
        lns.append(patch1)
        # add manual symbols to auto legend
        handles = (lns)

        scatter_ax.scatter(df_mean[d].values, df_mean.index+10, label='20'+d[:2], color=clrs[i], s=4)


    # ax2.scatter(dat['median'], dat.index+10, label=dt)
    scatter_ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.38), ncol=1)
    scatter_ax.set_xlim(0, 0.8)
    scatter_ax.set_xticks([0.2, 0.4, 0.6])
    scatter_ax.yaxis.tick_right()
    scatter_ax.yaxis.set_label_position("right")
    scatter_ax.grid('both')
    # ax2.set_title(yr)
    scatter_ax.set_ylabel('Elevation (m)')
    scatter_ax.set_xlabel('S2-derived albedo')


        # fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.9, 0.2), ncol=2)
    ax[0].add_artist(ScaleBar(dx=1, location="lower left"))
    box.boundary.plot(ax=ax[4], alpha=1, color='magenta', linewidth=1)

    patchAWS = Line2D([0], [0], marker='*', linestyle='None', label='AWS', color='red', markersize=8)
    patchOutline = Line2D([0], [0], label='outline 2017', color='yellow', linewidth=0.5, linestyle='-')
    patchContour015 = Line2D([0], [0], label='albedo 0.15 (contour line)', color='orange', linewidth=0.8, linestyle='-')
    # patchContour020 = Line2D([0], [0], label='albedo 0.20 (contour line)', color='MediumSpringGreen', linewidth=0.5, linestyle='-')
    hls = [patchAWS, patchOutline, patchContour015]#, patchContour020]
    fig.legend(handles=hls, loc='lower right', bbox_to_anchor=(0.7, 0.18), ncol=1)
    axins.set_title('20220824, zoom')

    fs = 12
    axins.text(631200, 5.1873e6, 'h', fontsize=fs,
               bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))
    scatter_ax.text(0.6, 2200, 'i', fontsize=fs,
               bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    an = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    for i, a in enumerate(ax[:-1]):
        a.set_xlim(628810, 638200)
        a.set_ylim(5185320, 5193340)
        a.text(629500, 5.1915e6, an[i], fontsize=fs,
               bbox=dict(boxstyle="square,pad=0.3", fc="lightgrey", ec="grey", lw=2))

    ax[-1].set_axis_off()

    fig.savefig('figs/SummerMinima_glacierwide.png', dpi=200, bbox_inches='tight')






GI5g = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/GepatschGI5.shp')
GI5g.to_crs(epsg=32632, inplace=True)

stakes1 = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/stakes_AWS_centroids.shp')
stakes1.to_crs(epsg=32632, inplace=True)
AWS = stakes1.loc[stakes1.name == 'AWS']
stakes = stakes1.loc[stakes1.name != 'AWS']



RGI = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/RGI60_onlyOetz.shp')
RGI.to_crs(epsg=32632, inplace=True)

# overview(GI5g, AWS, stakes)

dts = ['220702', '220712', '220722', '220804', '220809', '220814', '220816', '220821', '220824', '220923']#
# dts2 = ['220824', '230910']
dts2 = ['180817', '190827', '200915', '210821', '220824', '230910', '240907']

ContourPoly = gpd.read_file('/Users/leahartl/Desktop/WSS/outlines/ContourPolygons2017.shp')
ContourPoly.to_crs(epsg=32632, inplace=True)

# histogram plots (Paper Fig.)
histograms(GI5g, RGI, dts2, ContourPoly)
# for poster - different layout and transparent background
# histograms_poster(GI5g, RGI, dts2, ContourPoly)

# subplots showing summer minimum conditions and albedo (Paper Fig.)
subplotsRGBcontours(GI5g, AWS, RGI, dts2, ContourPoly)


plt.show()


