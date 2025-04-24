# Various scripts to process and visualize WSS data

## Meteo data:

general scripts to load and process the AWS data (not specific to paper, adjust later)
**AWS_WSS_mergefiles.py:**  
+ load individual data files
+ merge (delete any duplicate data)
+ save to new file

**AWS_processingWSS.py** 
+ load data file
+ apply QC process and set quality flags
+ save new files "raw" and "proc" (with flags) to folder "out"
+ save additional file "no flags" wth flagged data set to nan and without the quality flag columns


**AWS_plots.py**
+ various plotting functions to visualize the AWS data


-------------------
## process stake data
**process_stakes/proc_stakes.py**		
reads MSW's excel file with the stake data (prior QC carried out on .xlsx) and outputs a csv file formated for further processing and upload to pangaea. 


**WSS_stakes_coordinates.py**
outputs some statistics related to the dGPS coordinates. 
makes centroids of the dGPS points of the stakes and AWS, outputs as shapefile. Also outputs a .shp of the camera position.
makes the figure showing the stake positions.


**extractpoints_GEE.py**		
extracts albedo data for the point locations of the centroids (stakes & aws) produced above from S2 imagery and saves to csv files (one per point).
set the buffer size in the script! (for paper: run with 5 and 10 m)


specific to paper:
## Albedo figures:	
**AWS_albedo_corr.py**        
computes solar time and checks sun angles. calls functions in "solar_helpers.py"

writes output csv file "albedo_cor.csv" with POA (plane of array) direct and diffuse components
rest not needed for paper


**AWS_albedo_plots.py**        
various plots related to albedo 
SEE FILE FOR COMMENTS, TRANSFER MAIN POINTS TO READ ME.
makes plots of:
example weeks in winter and summer shoing impact of til corr (sup.)
scatter of diff between tilt corr and no correction (sup.)
time series of albedo anomalies, one panel per year (paper fig)
scatter plots of S2 vs in situ albedo (buffer 5 and buffer 10, paper and sup fig)






**WSS_stakes.py**
makes time series plot with two panels: 1) S2 and AWS albedo at the AWS 2) S2 albedo at the stakes
counts low albedo days. writes table of yearly abblation and low albedo counts to csv and latex. makes plot.
computes and prints correlation between annual stake ablation and low albedo days.


**WSS_stakes_subset.py:**
Makes plots showing AWS albedo and additional AWS parameters for subperiods in summers 2018, 2021, 2022. 
Also makes figures with RGB composites of the summit area overlayed with albedo contours for the same summers.


**Figures_Modeloutput.py**
Loads COSIPY output 
makes all model related figures
prints stats for average 15 day periods to csv


**WSS_glacierwide_S2.py**
makes histogram plot 
males plot with sat images of minimum snow cover. 
Note: download S2 data with GEE script before running this.  
Run	ProcessS2/processFiles_S2.py to generate csv files with elevation zone data.





**constants.py**
File setting constants to be used in cosipy runs. Note commentary in the file. 



