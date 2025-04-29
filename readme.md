# Various scripts to process and visualize WSS data

## General data figs and snow cleaning:
**WSS_data_figure.py**
Makes a sup. fig. showing availability of different data types.

**AWS_WSS_snowcleaning.py**		
Version of the AWS data proc file that includes treatment of the surface height/snow to filter and smooth the noisy SR50 data.
Makes time series plot of surface height with temperature (sup. Fig.) and "zoomed" view of surf height and snow during the eval period in August 2021. 

## Stake data: 
**WSS_stakes_coordinates.py**
loads csv file with intermediate stake data and coordinates. 
outputs some statistics related to the dGPS coordinates. 
makes centroids of the dGPS points of the stakes and AWS, outputs as shapefile. Also outputs a .shp of the camera position.
makes a figure showing the stake positions overlayed on a sentinel-2 image (Fig. 2)

**extractpoints_GEE.py**		
extracts albedo data for the point locations of the centroids (stakes & aws) produced above from S2 imagery and saves to csv files (one per point). The csvs are stored in a dir called "output_GEE", which needs to be created before running the script.
set the buffer size in the script! (for paper: run with 5 and 10 m buffer)


## Albedo figures:	
**AWS_albedo_corr.py**        
computes solar time and checks sun angles. calls functions in "solar_helpers.py"
writes output csv file "albedo_cor.csv" with POA (plane of array) direct and diffuse components
contains additional experiments related to the slope angle - not needed for paper

**AWS_albedo_plots.py**        
various plots related to albedo 
SEE FILE FOR ADDITIONAL COMMENTS
makes plots of:
example weeks in winter and summer shoing impact of til corr (sup. Fig.)
scatter of diff between tilt corr and no correction (sup. Fig.)
time series of albedo anomalies, one panel per year (Fig. 3)
scatter plots of S2 vs in situ albedo (Fig. 4, csn set buffer to 5 or 10m)

**WSS_stakes.py**
makes time series plot with two panels: 1) S2 and AWS albedo at the AWS 2) S2 albedo at the stakes
counts low albedo days. writes table of yearly abblation and low albedo counts to csv and latex. makes plot.
computes and prints correlation between annual stake ablation and low albedo days.

**WSS_stakes_subset.py:**
Makes plots showing AWS albedo and additional AWS parameters for subperiods in summers 2018, 2021, 2022. 
Also makes figures with RGB composites of the summit area overlayed with albedo contours for the same summers.


## COSIPY output figures:		
**constants.py**
File setting constants to be used in cosipy runs. Note commentary in the file. This is used in the cosipy runs.
The modified COSIPY branch is here: https://github.com/baldoa/cosipy_MSc/

**Figures_Modeloutput.py**
Loads COSIPY output 
makes all model related figures
prints stats for average 15 day periods to csv


## Glacierwide and overview figs:		
**WSS_glacierwide_S2.py**
makes histogram plot 
males plot with sat images of minimum snow cover. 
Note: download S2 data with GEE script before running this.  
Run	ProcessS2/processFiles_S2.py to generate csv files with elevation zone data.

**WSS_overview.py**		
Makes multi-panel overview figure.









