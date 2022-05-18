# -*- coding: utf-8 -*-
"""
Created on 2022-04-28.

Program name: 
 GGF_Training_Model.py

Program objectives:
 This program reads in data from an archive of ground based UK magnetometer 
 data and an archive of real-time solar wind data, and it then fits a 
 regression model between systematic random samples of the data.

Input data requirements: 
 - GGF_Training_Model_Dataset_of_Storm_Epochs.dat. 
    - Description: This is a ranked list of days according to the minimum Dst
      within them, compiled by Yulia Bogdanova.
    - Source: Yulia Bogdanova, pers comm. via Mervyn Freeman (mpf@bas.ac.uk).
    - Is it included here: yes. These data do not need to be downloaded 
      because the file is provided in the same directory as this program.
    - Format: ascii file loaded as numpy array where rows are individual days, 
      and the columns are [year, month, day, storm rank], e.g. 2011 08 06 085.
 - GGF_Training_Model_Dataset_of_Dst_from_2000_to_2017.dat.
    - Description: hourly values of the Disturbance Storm Time (Dst) index 
      spanning the years 2000 to 2017.
    - Source: downloaded from https://omniweb.gsfc.nasa.gov/form/dx1.html on 
      2021-07-30.
    - Is it included here: yes. These data do not need to be downloaded 
      because the file is provided in the same directory as this program.
    - Format: ascii file loaded as numpy array where each row is the Dst record 
      for a given hour, and the columns are [year, day of year (starting at 1),
      hour of day (starting at 0), Dst value (units of nanoTesla)]. E.g. 
      [2000   1  0   -45].
 - British Geological Survey Observatory Daily Data Files.
    - Description: daily files of data from the three INTERMAGNET observatories
      Hartland, Eskdalemuir, and Lerwick. The files span the years 2000 to 
      2017. Each file is a day-long ASCII text file containing records at 
      1-minute cadence of the remnant external magnetic field once an estimate 
      of the core + crustal field have been removed (using a fixed value of 
      this estimate per day).
    - Source: obtained from British Geological Survey (pers comm.). The 
      observatory data can be downloaded directly from the INTERMAGNET ftp at 
      ftp://ftp.seismo.nrcan.gc.ca/intermagnet/minute/definitive/IAGA2002, and 
      these ascii files can be parsed using the same code used to parse the 
      'BGS_file_data' in this program. The core and crustal field estimates can
      be removed from these data using the CHAOS geomagnetic field model. The 
      latest iteration of this model is described in Finlay et al. 2020, 'The 
      CHAOS-7 geomagnetic field model and observed changes in the South 
      Atlantic Anomaly', https://doi.org/10.1186/s40623-020-01252-9. As stated 
      in that paper, the CHAOS-7 model and its updates are available at 
      http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/. A python 
      package for using the CHAOS model is available at 
      https://pypi.org/project/chaosmagpy/.
    - Is it included here: no. These data are stored on the British Antarctic 
      Survey (BAS) servers (location given below), or can be downloaded from 
      the location stated above if this program is being run outside of BAS.
    - Format: each file contains 1440 rows and the 5 columns are 
      [day-month-year, hour:minute, x-component, y-component, z-component]. The
      magnetic components are in units of nanoTesla. For an example row from 
      Eskdalemuir station: [02-01-2016  23:47        -2.8       4.2       7.7].
      The missing data value is 99999.9.
 - Archived daily files of real-time solar wind data from the Advanced 
   Composition Explorer (ACE) spacecraft.
    - Description: daily files of magnetometer and plasma measurements taken in
      real time from the ACE spacecraft and archived at the date of access. 
      These data have not been subjected to Level-2 processing.
    - Source: obtained from British Geological Survey (pers comm.). If the data
      need to be downloaded anew, then they are presently accessible at 
      https://webapps.bgs.ac.uk/services/ngdc/accessions/index.html#item172549.
      These data are originally from the SPace Weather Precition Center (SWPC).
      If the data need to be obtained directly from SWPC, then the top-level 
      data description is at 
      https://www.swpc.noaa.gov/products/ace-real-time-solar-wind. As described 
      under the 'Data' tab on that page, the ACE real-time data are archived 
      back to August 2001 at 
      https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ (download files of 
      type 'YYYYMMDD_ace_mag_1m.txt' and 'YYYYMMDD_ace_swepam_1m.txt', where 
      YYYY is year, MM is month and DD is day). The current real-time data can 
      be accessed at https://services.swpc.noaa.gov/text/ace-magnetometer.txt 
      and https://services.swpc.noaa.gov/text/ace-swepam.txt.
    - Is it included here: no. These data are stored on the British Antarctic 
      Survey (BAS) servers (location given below), or can be downloaded from 
      the location stated above if this program is being run outside of BAS.
    - Format: 
       - The ACE magnetometer files named 'YYYYMMDD_ace_mag_1m.txt' are daily 
         ascii files of 1-min cadence data. There are 20 header lines, and 
         the remaining 1440 rows each have the following columns: [year, month, 
         day, time (HHMM), Modified Julian day, seconds within the day, data 
         status flag, Bx, By, Bz, Bt, Lat., Long.], where B(xyz) are the GSM 
         frame magnetic components, Bt is the total field in the GSM frame, and
         Lat. and Long. are the GSM coordinates. The missing data value is 
         -999.9.
       - The ACE swepam files named 'YYYYMMDD_ace_swepam_1m.txt' are daily 
         ascii files of 1-min cadence data. There are 18 header lines, and 
         the remaining 1440 rows each have the following columns: [year, month, 
         day, time (HHMM), Modified Julian day, seconds within the day, data 
         status flag, proton density, bulk speed, ion temperature], where the 
         last three columns pertain to the solar wind. The missing data values 
         are: Density and Speed = -9999.9, Temp. = -1.00e+05.
 - GGF_Training_Model_Dataset_of_Indices_for_Ensemble_Storm_Epoch_Selection.pkl.
    - Description: the dates in GGF_Training_Model_Dataset_of_Storm_Epochs.dat 
      are subsampled 100 times with an 80%/20% ratio into training and test 
      sets. The data within this file are a set of indices pertaining to the 
      processed storm-date epochs, such that the same training data can be 
      selected every time this program is run.
    - Source: generated by program Randomised_Training_Regression_v2p0.py. This
      program is not included here, but the pertinent code showing how these 
      indices were generated is included (commented-out) nearby the load-in of
      this file in the code below.
    - Is it included here: yes. These data do not need to be downloaded 
      because the file is provided in the same directory as this program.
    - Format: each variable here is a numpt array of index values used to 
      select elements of data lists which have the same order as 
      all_storms_epochs_set_sorted (defined in this program). The file contains 
      two variables, with the following formatting:
       - index_for_storms_epochs_all_randomised_training_sets: 100 rows, 50 
         columns. Each row is a different random selection of the full set of 
         storm epochs. The columns colectively specify 50 training storms.
       - index_for_storms_epochs_all_randomised_test_sets: 100 rows, 12 
         columns. Each row is a different random selection of the full set of 
         storm epochs. The columns colectively specify 12 training storms.

Outputs: 
 - GGF_Training_Model_Dataset_of_Stored_Model_Coefficients.pkl.
    - Description: an output file of the regression model coefficients produced 
      by this program.
    - Source: made by this program.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: a Python list containing a 5-dimensional numpy array of model 
      coefficients. The dimensions are: [stations, local time bins, magnetic 
      components, model parameters, randomised training data instances]. The 
      details of each dimension is as follows.
       - There are three stations (i.e. observatories) in the order: 
         Eskdalemuir, Hartland, Lerwick.
       - There are 24 local time bins, which are spaced at 1 hour intervals 
         with a 3-hour width for each bin. The first bin has a centroid at 
         00:30 local time, and spans the three hours from 23:00 to 02:00 local 
         time.
       - The three magnetic components have the order: x, y, z.
       - There are 6 model parameters which are used to solve for an estimate 
         of the ground geomagnetic variation (y) for a given magnetic component
         at a given station at a given local time. The equation being solved is 
         y = a + b*(sin(DOY)) + c*(cos(DOY)) + d*epsilon + e*(sin(DOY) * epsilon) + f*(cos(DOY) * epsilon)
         where a is the intercept coefficient, and the other coefficients 
         describe the effect of the following parameters: b for sine of day of 
         year (DOY), c for cosine of DOY, d for epsilon, e for 
         (sine of DOY)*epsilon, and f for (cosine of DOY)*epsilon.
       - There are 100 randomised training data instances, each of which 
         relates to a different random selection of 80% of the full set of 
         geomagnetic storm intervals. The order of these randomised selections 
         is the same as in the variable 
         'index_for_storms_epochs_all_randomised_training_sets' in the file 
         'GGF_Training_Model_Dataset_of_Indices_for_Ensemble_Storm_Epoch_Selection.pkl'.

Instructions for running this program:
 This program is intended to be run on a machine with access to the British 
 Antarctic Survey (BAS)'s data servers. The variables 'project_directory' and 
 'data_directory' should be set manually by the user, dependent on where the 
 data are stored. If running outside of BAS, the program should be able to
 be run if the user downloads all the data files mentioned above, and 
 replicates (or alters) the file structure used throughout this program. At 
 the time of writing, the code and data are located in the following 
 directories.
 - On the BAS Linux servers:
    - The data are in /data/psdcomplexity/eimf/SAGE_BGS.
    - The code is in /data/psdcomplexity/eimf/SAGE_Model_Handover/GGF.
 - On the BAS machine bslthemesa:
    - The directory /data/psdcomplexity/eimf is linked to via /local/users/robore/shortcut_EIMF_data.
 - On the local hard drive of the BAS laptop used by robore@bas.ac.uk, i.e. 
   where these programs were developed:
    - The data are in C:/Users/robore/BAS_Data/SAGE_BGS.
    - The code is in C:/Users/robore/BAS_Files/Research/Code/SAGE/SAGE_Model_Handover/GGF.

@author: robore@bas.ac.uk. Robert Shore, ORCID: orcid.org/0000-0002-8386-1425.
For author's reference: this program was based on program 
 Randomised_Training_Regression_v3p0.py, shorthand 'RTRv3p0_'.
"""

#%% Import packages and set constants.

import numpy as np
import datetime
import pickle
import sys
import os
import re
import math
import pandas as pd

#Define the top-level directory location.
project_directory = os.path.join(os.sep,'local','users','robore','shortcut_EIMF_data','SAGE_Model_Handover','GGF')#string: /local/users/robore/shortcut_EIMF_data/SAGE_Model_Handover/GGF.
data_directory = os.path.join(os.sep,'local','users','robore','shortcut_EIMF_data','SAGE_BGS')#string: /local/users/robore/shortcut_EIMF_data/SAGE_BGS.
#Dear user: if running on a BAS machine, please edit the above strings to be
# project_directory = '/data/psdcomplexity/eimf/SAGE_Model_Handover/GGF', and
# data_directory = '/data/psdcomplexity/eimf/SAGE_BGS'.

#Define lags: how many minutes to lag the IMF series by?
index_lags = [0, 20]#scalar list, units of minutes' lag.
index_lag = np.nonzero(np.array(index_lags) == 20)[0][0]#scalar index, pertains to elements of index_lags.

#Set threshold of minimum Dst, for which the first Dst values below this value 
# define the start date for each storm.
Dst_threshold = -30#units of nT.

#%% Load storm epochs, and concatenate them into one set of epochs.

#Processing history is as follows.
#The starting point for these epochs is a ranked list of days according to the
# minimum Dst within them, compiled by Yulia Bogdanova. It's the top 100 such 
# days between 1 Jan 2000 and 30 Sep 2017. Processing steps are as follows, and
# were initially done by programs BGS_IMF_Regression_v1p1.py and 
# BGS_IMF_Regression_v5p3.py.
#
#I take the list and remove the dates which are outside of the BGS data span 
# (which is 2000.0--2016.0).
#
#I specify a threshold value for low negative Dst (defined above as -30 nT) 
# and for each storm I find the epoch at which Dst first drops below this 
# threshold prior to the storm's point of peak minimum Dst. This gives the 
# starting time of a given storm. The end of the storm is defined to be 24 
# hours after the epoch of peak Dst for the storm.
#
#Some days in the list belong to the same geomagnetic event, so I now remove 
# duplicate instances from the list which have idential start and end epochs. 
# The list is temporally sorted.

#Load storms.
all_storms_epochs_set = np.loadtxt(os.path.join(project_directory,'GGF_Training_Model_Dataset_of_Storm_Epochs.dat'))#np array, size [storms by 4 (year, month, date, storm rank)]

#%% Load all storm spans of BGS station data and lagged solar wind data.
#This sewt of code cells defines the span of each storm using Dst (as described 
# in the comments above), and it also ballistically propagates the solar wind 
# data to the bow shock nose, then adds additional pre-defined lags to account 
# for theionospheric reconfiguration timescale. Following this, the observatory 
# and L1 data are compiled into lists of numpy arrays: one array per storm.

print('Loading storm-time data ...')

#%%% Load Dst index for the years 2000-2017.

#Load Dst data.
dst_data = np.loadtxt(os.path.join(project_directory,'GGF_Training_Model_Dataset_of_Dst_from_2000_to_2017.dat'))#np array of floats, size [times (hours in 2000--2017) by 4 (year, day of year, hour, Dst (units of nT))]
# - Description: hourly values of the Disturbance Storm Time (Dst) index 
#   spanning the years 2000 to 2017.
# - Source: downloaded from https://omniweb.gsfc.nasa.gov/form/dx1.html on 
#   2021-07-30.
# - Is it included here: yes. These data do not need to be downloaded 
#   because the file is provided in the same directory as this program.
# - Format: ascii file loaded as numpy array where each row is the Dst record 
#   for a given hour, and the columns are [year, day of year (starting at 1),
#   hour of day (starting at 0), Dst value (units of nanoTesla)].

#Make a datetime object for the Dst dates, converting from day of year to date.
dst_epochs = np.zeros((np.shape(dst_data)[0],1), dtype='datetime64[s]')#np array of datetime64[s], size [times (hours in 2000--2017) by 1].
for i_epoch, dst_element in enumerate(dst_data,start=0):
    dst_epochs[i_epoch,0] = datetime.datetime(np.int64(dst_element[0]), 1, 1) + datetime.timedelta(int(dst_element[1]) - 1) + datetime.timedelta(hours=int(dst_element[2]))
#Conversion from https://stackoverflow.com/questions/2427555/python-question-year-and-day-of-year-to-date.

#%%% Define the time span of each storm, using the Dst values, and remove overlaps between storms.

#For each storm, find the date range for which Dst is below the threshold 
# level before the storm's peak minimum Dst, and one day after the peak.
all_storms_start_date_hour_precision = np.empty([len(all_storms_epochs_set),1],dtype='datetime64[s]')#np array of datetime64[s], size [storms by 1]
all_storms_end_date_hour_precision = np.empty([len(all_storms_epochs_set),1],dtype='datetime64[s]')#np array of datetime64[s], size [storms by 1]
for i_storm_counter, storm_epoch in enumerate(all_storms_epochs_set):
    #Get a variable for the date of this storm, at day-granularity.
    storm_day_date = datetime.datetime(np.int64(storm_epoch[0]),np.int64(storm_epoch[1]),np.int64(storm_epoch[2]))#scalar datetime object
    
    #Extract a 24-hour span of dates and Dst values for that day.
    index_storm_day_hours = np.transpose(np.nonzero((dst_epochs[:,0] >= storm_day_date) & (dst_epochs[:,0] <= (storm_day_date + datetime.timedelta(hours=23.5)))))#np array of ints, size [hours in one day by 1]. Values pertain to rows of dst_data and dst_epochs.
    dst_during_storm_day = dst_data[index_storm_day_hours,3]#np array of floats, size [24 (hours during the day) by 1]
    dates_during_storm_day = dst_epochs[index_storm_day_hours,:]#np array of datetime64[s], size [24 (hours during the day) by 1 by 1]
    
    #Find the date of the lowest Dst value for that day. 
    # Takes the first one, if there are multiple identical lowest values.
    subindex_lowest_dst_value = np.nonzero(dst_during_storm_day == (min(dst_during_storm_day)))[0][0]#scalar int sub-index, pertains to rows of dst_during_storm_day
    date_of_lowest_dst_in_storm_day = dates_during_storm_day[subindex_lowest_dst_value,0]#np scalar array of datetime64[s] object
    
    #Find the index of that lowest-Dst-date within the full 2013-2016 archive.
    index_lowest_dst_value = np.nonzero(dst_epochs[:,0] == date_of_lowest_dst_in_storm_day)[0][0]#scalar int index, pertains to rows of dst_data and dst_epochs.
    
    #Use the date of the minimum Dst during the given storm-day to split the 
    # entire Dst archive into 'before' and 'after' parts. Vertically invert the 
    # 'before' part to simplify indexing. Note that both parts will contain the 
    # storm centre value: this is not important as we only use them to get the 
    # storm span dates.
    dst_values_before_storm_centre = np.flipud(dst_data[0:index_lowest_dst_value+1,3])#np array of floats, size [(subset of dst times for 2000--2017) by 0].
    dst_dates_before_storm_centre = np.flipud(dst_epochs[0:index_lowest_dst_value+1,0])#np array pf datetime64[s], size [(subset of dst times for 2000--2017) by 0].
    
    #Find the sub-index of the inverted 'before' part at which Dst first rises 
    # above the threshold value which signifies the storm (i.e. find the storm start),
    # and extract that date.
    storm_start_date_hour_precision = dst_dates_before_storm_centre[np.nonzero(dst_values_before_storm_centre > Dst_threshold)[0][0]]#scalar datetime64 object, at hour-precision.
    
    #Find the sub-index of the 'after' part of the storm which is exactly 
    # 24 hours after the epoch of minimum Dst during the given storm-day, 
    # and call it the end date of the storm.
    storm_end_date_hour_precision = date_of_lowest_dst_in_storm_day[0] + np.timedelta64(24,'h')#scalar datetime64 object, at hour-precision.
    
    #Preserving the old approach, wherein the Dst threshold was used to
    # define the tail end of the storm.  However, this allowed the long 
    # cool-down of the ring current to control the polar ionospheric
    # training interval, and the iononsphere returns to normal much quicker
    # than the ring current does: this was causing a lot of quite data to
    # be in the training interval, which is not what we want.
    #storm_end_date_hour_precision = dst_dates_after_storm_centre[np.nonzero(dst_values_after_storm_centre > Dst_threshold)[0][0]]#scalar datetime64 object.
    
    #Store the start and end dates for this storm.
    all_storms_start_date_hour_precision[i_storm_counter] = storm_start_date_hour_precision#scalar datetime64[s] value sstored in np array of datetime64[s], size [storms by 1].
    all_storms_end_date_hour_precision[i_storm_counter] = storm_end_date_hour_precision#scalar datetime64[s] value sstored in np array of datetime64[s], size [storms by 1].
#End loop over storm epochs

#Define an index which will sort the storm dates in ascending order, 
# starting from the lowest value.
indices_of_sorted_storm_start_dates = np.argsort(all_storms_start_date_hour_precision,axis=0)#list of integer indices, size [storm by 1].

#Create versions of the storm dates which are sorted by the Dst-defined 
# start-hour for each storm.
all_storms_start_date_hour_precision_sorted = np.squeeze(np.array(all_storms_start_date_hour_precision[indices_of_sorted_storm_start_dates]),axis=2)#np array of datetime64[s], size [storms by 1]
all_storms_end_date_hour_precision_sorted = np.squeeze(np.array(all_storms_end_date_hour_precision[indices_of_sorted_storm_start_dates]),axis=2)#np array of datetime64[s], size [storms by 1]
all_storms_epochs_set_sorted = np.squeeze(np.array(all_storms_epochs_set[indices_of_sorted_storm_start_dates]),axis=1)#np array of numbers, size [storms by 4 (year, month, date, storm rank)]


#Loop through the sorted storm dates, and flag duplicate start-dates for removal.
index_of_sorted_storms_to_remove = []
for i_storm_counter in range(len(all_storms_start_date_hour_precision_sorted)):
    #Define an index of where this start date exits in the full set of 
    # start dates.
    index_same_start_date = np.nonzero(all_storms_start_date_hour_precision_sorted == all_storms_start_date_hour_precision_sorted[i_storm_counter])[0]#np array of ints, size [duplicate dates by 0]
    
    #If there's more than one start date, flag the non-leading date(s) for 
    # later removal.
    if(len(index_same_start_date) > 1):
        index_of_sorted_storms_to_remove = np.append(index_of_sorted_storms_to_remove,index_same_start_date[1:])#np array of ints, size [all duplicate dates by 0]
    #End conditional.
#End loop over storm dates.

#Retrieve just the unique indices of the duplicate storms.
index_of_sorted_storms_to_remove = np.transpose(np.unique(index_of_sorted_storms_to_remove)[np.newaxis].astype(np.int64))#np array of ints, size [unique values of duplicate dates by 0]

#Remove the duplicate epochs for the storm start and end dates, the sorting
# index, and the storm centre-day-dates.
indices_of_sorted_storm_start_dates = np.delete(indices_of_sorted_storm_start_dates,index_of_sorted_storms_to_remove,axis=0)#np array of ints, size [storms by 1].
all_storms_start_date_hour_precision_sorted = np.delete(all_storms_start_date_hour_precision_sorted,index_of_sorted_storms_to_remove,axis=0)#np array of datetime64[s], size [storms by 1]
all_storms_end_date_hour_precision_sorted = np.delete(all_storms_end_date_hour_precision_sorted,index_of_sorted_storms_to_remove,axis=0)#np array of datetime64[s], size [storms by 1]
all_storms_epochs_set_sorted = np.delete(all_storms_epochs_set_sorted,index_of_sorted_storms_to_remove,axis=0)#np array of numbers, size [storms by 4 (year, month, date, storm rank)]


#Loop through the sorted storm dates, and truncate the end-date if it 
# exceeds the start date of the following storm.
for i_storm_counter in range(len(all_storms_start_date_hour_precision_sorted) - 1):
    if(all_storms_end_date_hour_precision_sorted[i_storm_counter] > all_storms_start_date_hour_precision_sorted[i_storm_counter+1]):
        all_storms_end_date_hour_precision_sorted[i_storm_counter] = all_storms_start_date_hour_precision_sorted[i_storm_counter+1] - np.timedelta64(1,'h')
    #End conditional: does the end date of this storm come after that start date of the next storm? If so, they are likely the same storm.
#End loop over storms.

#Remove any storm dates where the start and end hour-precision dates are 
# the same, since these have a peak Dst which is below the miniumum 
# threshold.
index_of_sorted_storms_to_remove = []
for i_storm_counter in range(len(all_storms_start_date_hour_precision_sorted)):
    #If the start and end dates are the same, flag the non-leading date(s) for 
    # later removal.
    if(all_storms_start_date_hour_precision_sorted[i_storm_counter] == all_storms_end_date_hour_precision_sorted[i_storm_counter]):
        index_of_sorted_storms_to_remove = np.append(index_of_sorted_storms_to_remove,i_storm_counter)
    #End conditional.
#End loop over storm dates.

#Remove the flagged storms, if any are flagged.
if(len(index_of_sorted_storms_to_remove) > 0):
    #Alter the flagged storms to allow their indexing for removal.
    index_of_sorted_storms_to_remove = np.transpose(index_of_sorted_storms_to_remove[np.newaxis].astype(np.int64))
    
    #Remove the zero-length storms from the storm start and end dates, the sorting
    # index, and the storm centre-day-dates.
    indices_of_sorted_storm_start_dates = np.delete(indices_of_sorted_storm_start_dates,index_of_sorted_storms_to_remove,axis=0)
    all_storms_start_date_hour_precision_sorted = np.delete(all_storms_start_date_hour_precision_sorted,index_of_sorted_storms_to_remove,axis=0)
    all_storms_end_date_hour_precision_sorted = np.delete(all_storms_end_date_hour_precision_sorted,index_of_sorted_storms_to_remove,axis=0)
    all_storms_epochs_set_sorted = np.delete(all_storms_epochs_set_sorted,index_of_sorted_storms_to_remove,axis=0)
#End conditional: remove epoch data if there's nothing to concatenate for a given date.

#%%% Loop over all input storm epochs to collect all BGS external data and RTSW data.

#Preallocate storage: initiate empty lists to later fill with numpy arrays.
BGS_data_dates_all_stations_all_storms = []#empty list, size will be [all storms].
BGS_local_time_hours_all_stations_all_storms = []#empty list, size will be [all storms].
BGS_local_time_minutes_all_stations_all_storms = []#empty list, size will be [all storms].
BGS_local_time_seconds_all_stations_all_storms = []#empty list, size will be [all storms].
BGS_data_all_stations_all_storms = []#empty list, size will be [all storms].
BGS_data_size_each_storm = []#empty list, size will be [all storms].
RTSW_ACE_epochs_all_storms = []#empty list, size will be [all storms].
RTSW_ACE_epsilon_lagged_all_storms = []#empty list, size will be [all storms].
RTSW_ACE_Bx_lagged_all_storms = []#empty list, size will be [all storms].
RTSW_ACE_By_lagged_all_storms = []#empty list, size will be [all storms].
RTSW_ACE_Bz_lagged_all_storms = []#empty list, size will be [all storms].
RTSW_ACE_speed_lagged_all_storms = []#empty list, size will be [all storms].
for i_storm_counter, storm_epoch in enumerate(all_storms_epochs_set_sorted):
    print('Processing storm at ' + str(int(storm_epoch[0])) + '-' + str(int(storm_epoch[1])) + '-' + str(int(storm_epoch[2])))
    #%%%% Extract the time-span for this particular storm, using the values previously defined via Dst limits.
    
    #Extract hour-precision storm date range.
    storm_start_date_hour_precision = all_storms_start_date_hour_precision_sorted[i_storm_counter,0]#scalar datetime64 object.
    storm_end_date_hour_precision = all_storms_end_date_hour_precision_sorted[i_storm_counter,0]#scalar datetime64 object.
    
    #%%%% Loop over the BGS stations to load the magnetometer data for the (integer) days in the storm span.
    
    #Define a range of days that the storm spans, at day-granularity, and converted 
    # from numpy datetime to datetime, for the BGS file read-in. Need them in plain 
    # datetime to work with my date-range function ('datetime_range').
    storm_start_date_day_precision = datetime.date(storm_start_date_hour_precision.astype(object).year, storm_start_date_hour_precision.astype(object).month, storm_start_date_hour_precision.astype(object).day)#scalar date object, at day-precision.
    storm_end_date_day_precision = datetime.date(storm_end_date_hour_precision.astype(object).year, storm_end_date_hour_precision.astype(object).month, storm_end_date_hour_precision.astype(object).day)#scalar date object, at day-precision.
    
    #It may be that the gap between the hour-precision storm span 
    # and the start/end of the storm days span is not sufficient to allow 
    # the time required to buffer against the lags imposed. To counter 
    # this, we take the reasonable step of adding one day before, and one 
    # after, the storm days span.
    storm_start_date_day_precision = storm_start_date_day_precision - datetime.timedelta(days=1)#scalar date object.
    storm_end_date_day_precision = storm_end_date_day_precision + datetime.timedelta(days=1)#scalar date object.
    
    #Define a function to create an array at fixed time interval.
    def datetime_range(start, end, delta):
        current = start
        while current <= end:
            yield current
            current += delta
        #End while loop.
    #End function definition.
    
    #Create a list of dates at day-granularity and convert to numpy array.
    BGS_file_day_dates = np.array([dt for dt in datetime_range(storm_start_date_day_precision, storm_end_date_day_precision, datetime.timedelta(days=1))])#np array of date objects, size [days encompassing storm span by 0]
    
    #Loop over the three stations.
    BGS_stations = ['esk','had','ler']#list of size [3]
    BGS_station_longitudes = [(356.8 - 360), (355.5 - 360), (358.8 - 360)]#list of size [3] -- geodetic longitude. Converted to east longitude on a +-180 scale (for later local time calculations), so it's negative because it's slightly west.
    #Longitudes sourced from BGS daily data file headers, and INTERMAGNET site.
    #Preallocate storage.
    BGS_storm_days_all_stations_dates = np.empty([1440*len(BGS_file_day_dates),3],dtype='datetime64[us]')#np array of datetime64[us], size [(minutes in day)*(number of BGS daily files) by 3 stations]
    BGS_storm_days_all_stations_local_time_hours = np.empty([1440*len(BGS_file_day_dates),3])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 3 stations]
    BGS_storm_days_all_stations_local_time_minutes = np.empty([1440*len(BGS_file_day_dates),3])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 3 stations]
    BGS_storm_days_all_stations_local_time_seconds = np.empty([1440*len(BGS_file_day_dates),3])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 3 stations]
    BGS_storm_days_all_stations_x_data = np.empty([1440*len(BGS_file_day_dates),3])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 3 stations]
    BGS_storm_days_all_stations_y_data = np.empty([1440*len(BGS_file_day_dates),3])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 3 stations]
    BGS_storm_days_all_stations_z_data = np.empty([1440*len(BGS_file_day_dates),3])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 3 stations]
    for i_station, (BGS_station, BGS_station_longitude) in enumerate(zip(BGS_stations,BGS_station_longitudes),start=0):
        #%%%% Load BGS data for storm days for this station.
        
        #Preallocate storage for BGS daily file data.
        BGS_storm_days_single_station_dates = np.empty([1440*len(BGS_file_day_dates),1],dtype='datetime64[us]')#np array of datetime64[us], size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_single_station_x_data = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_single_station_y_data = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_single_station_z_data = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        #Loop over set of daily BGS files and load each.
        for i_day, BGS_file_single_day_date in enumerate(BGS_file_day_dates,start=0):
            #Define station file location string.
            BGS_file_location = os.path.join(data_directory,'BGS_external_data',str(BGS_file_single_day_date.year),\
                BGS_station,BGS_station + str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + str(BGS_file_single_day_date.day).zfill(2) + '.ext')#string.
            #End indenting for this variable.
            # - Description: daily files of data from the three INTERMAGNET observatories
            #   Hartland, Eskdalemuir, and Lerwick. The files span the years 2000 to 
            #   2017. Each file is a day-long ASCII text file containing records at 
            #   1-minute cadence of the remnant external magnetic field once an estimate 
            #   of the core + crustal field have been removed (using a fixed value of 
            #   this estimate per day).
            # - Source: obtained from British Geological Survey (pers comm.). The 
            #   observatory data can be downloaded directly from the INTERMAGNET ftp at 
            #   ftp://ftp.seismo.nrcan.gc.ca/intermagnet/minute/definitive/IAGA2002. The 
            #   core and crustal field estimates can be removed from these data using the 
            #   CHAOS geomagnetic field model. The latest iteration of this model is 
            #   described in Finlay et al. 2020, 'The CHAOS-7 geomagnetic field model and 
            #   observed changes in the South Atlantic Anomaly', 
            #   https://doi.org/10.1186/s40623-020-01252-9. As stated in that paper, the 
            #   CHAOS-7 model and its updates are available at 
            #   http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/. A python 
            #   package for using the CHAOS model is available at 
            #   https://pypi.org/project/chaosmagpy/.
            # - Is it included here: no. These data are stored on the British Antarctic 
            #   Survey (BAS) servers, or can be downloaded from the location stated above
            #   if this program is being run outside of BAS.
            # - Format: each file contains 1440 rows and the 5 columns are 
            #   [day-month-year, hour:minute, x-component, y-component, z-component]. The
            #   magnetic components are in units of nanoTesla. For an example row from 
            #   Eskdalemuir station: [02-01-2016  23:47        -2.8       4.2       7.7].
            #   The missing data flag is 99999.9.
            
            #Open BGS daily file.
            fid = open(BGS_file_location, 'r')
            
            #Read BGS file data.
            BGS_file_data = fid.readlines()#fid.readlines() gets all the content from the ascii file in one go, then we go through each line.
            
            #Close BGS daily data file.
            fid.close()
            
            #Loop over the daily file lines and extract variables of interest.
            BGS_single_day_single_station_dates = np.empty([1440,1],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in day by 1]
            BGS_single_day_single_station_x_data = np.empty([1440,1])#np array of floats, size [minutes in day by 1]
            BGS_single_day_single_station_y_data = np.empty([1440,1])#np array of floats, size [minutes in day by 1]
            BGS_single_day_single_station_z_data = np.empty([1440,1])#np array of floats, size [minutes in day by 1]
            i_line = 0
            for line in BGS_file_data:
                if(re.search('[|]',line)):
                    continue#ignore header lines containing character '|'.
                line = line.strip()#remove trailing \n
                columns = line.split()#split line string into space-delimited strings.
                #Convert strings to datetime64[us] or float formats and store each.
                BGS_single_day_single_station_dates[i_line,0] = np.datetime64(datetime.datetime.strptime(columns[0] + ',' + columns[1], '%d-%m-%Y,%H:%M'))#storing scalar datetime object.
                BGS_single_day_single_station_x_data[i_line,0] = np.float64(columns[2])#storing scalar float.
                BGS_single_day_single_station_y_data[i_line,0] = np.float64(columns[3])#storing scalar float.
                BGS_single_day_single_station_z_data[i_line,0] = np.float64(columns[4])#storing scalar float.
                #Increment storage index
                i_line += 1
            #End loop over each 1-min line of the daily file.
            
            #Store the daily file data in the arrays of BGS data.
            BGS_storm_days_single_station_dates[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = BGS_single_day_single_station_dates#storing set of 1440 values in the set of all days for the storm: size [1440 by 1].
            BGS_storm_days_single_station_x_data[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = BGS_single_day_single_station_x_data#storing set of 1440 values in the set of all days for the storm: size [1440 by 1].
            BGS_storm_days_single_station_y_data[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = BGS_single_day_single_station_y_data#storing set of 1440 values in the set of all days for the storm: size [1440 by 1].
            BGS_storm_days_single_station_z_data[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = BGS_single_day_single_station_z_data#storing set of 1440 values in the set of all days for the storm: size [1440 by 1].
        #End loop over each storm day.
        
        #Check for missing data values, and replace them with nan.
        BGS_storm_days_single_station_x_data[np.where(BGS_storm_days_single_station_x_data == 99999.9)] = np.nan
        BGS_storm_days_single_station_y_data[np.where(BGS_storm_days_single_station_y_data == 99999.9)] = np.nan
        BGS_storm_days_single_station_z_data[np.where(BGS_storm_days_single_station_z_data == 99999.9)] = np.nan
        
        #Compute the local time of the BGS station data, rounded to 1-second precision, 
        # because numpy doesn't accept float inputs to timedelta64.
        BGS_storm_days_single_station_dates_with_local_time_alteration = BGS_storm_days_single_station_dates + np.timedelta64(round((BGS_station_longitude / 15) * 60 * 60),'s')#np array of datetime64[us], size [(minutes in day)*(number of BGS daily files) by 1] -- converts longitude to hours, then minutes, then seconds.
        
        #Go through each item in the set of local-time-corrected 
        # BGS dates, strip out the year-month-day information, and then convert the 
        # hours, minute and seconds to separate arrays for later indexing.
        BGS_storm_days_single_station_local_time_hours = np.empty(np.shape(BGS_storm_days_single_station_dates))#np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_single_station_local_time_minutes = np.empty(np.shape(BGS_storm_days_single_station_dates))#np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_single_station_local_time_seconds = np.empty(np.shape(BGS_storm_days_single_station_dates))#np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        for i_date, date in enumerate(BGS_storm_days_single_station_dates_with_local_time_alteration,start=0):
            BGS_storm_days_single_station_local_time_hours[i_date,:] = date[0].tolist().time().hour#converts np scalar array of datetime64[us] to datetime object, restricts to hour,minute,second, extracts scalar hour float.
            BGS_storm_days_single_station_local_time_minutes[i_date,:] = date[0].tolist().time().minute
            BGS_storm_days_single_station_local_time_seconds[i_date,:] = date[0].tolist().time().second
            #Conversion advice from https://stackoverflow.com/questions/29834356/how-can-i-get-an-hour-minute-etc-out-of-numpy-datetime64-object
        #End loop over each minute of the BGS daily files.
        
        #%%%% Store data for this storm in the set of all storms.
        
        #Store BGS data for all storm days, for this station.
        BGS_storm_days_all_stations_dates[:,i_station,np.newaxis] = BGS_storm_days_single_station_dates#storing np array of datetime64[us], size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_all_stations_local_time_hours[:,i_station,np.newaxis] = BGS_storm_days_single_station_local_time_hours#storing np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_all_stations_local_time_minutes[:,i_station,np.newaxis] = BGS_storm_days_single_station_local_time_minutes#storing np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_all_stations_local_time_seconds[:,i_station,np.newaxis] = BGS_storm_days_single_station_local_time_seconds#storing np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_all_stations_x_data[:,i_station,np.newaxis] = BGS_storm_days_single_station_x_data#storing np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_all_stations_y_data[:,i_station,np.newaxis] = BGS_storm_days_single_station_y_data#storing np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        BGS_storm_days_all_stations_z_data[:,i_station,np.newaxis] = BGS_storm_days_single_station_z_data#storing np array of floats, size [(minutes in day)*(number of BGS daily files) by 1]
        #Note that the BGS dates won't vary between stations. The local time, calculated later, does vary between stations.
    #End loop over BGS stations.
    
    #%%% Load archived real-time solar wind (RTSW) data for this storm's days.
    
    #Preallocate storage for ACE daily file data.
    RTSW_ACE_storm_days_mag_dates = np.empty([1440*len(BGS_file_day_dates),1],dtype='datetime64[us]')#np array of datetime64[us], size [(minutes in day)*(number of ACE daily files) by 1]
    RTSW_ACE_storm_days_swepam_dates = np.empty([1440*len(BGS_file_day_dates),1],dtype='datetime64[us]')#np array of datetime64[us], size [(minutes in day)*(number of ACE daily files) by 1]
    RTSW_ACE_storm_days_Bx = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1]
    RTSW_ACE_storm_days_By = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1]
    RTSW_ACE_storm_days_Bz = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1]
    RTSW_ACE_storm_days_speed = np.empty([1440*len(BGS_file_day_dates),1])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1]
    #Loop over set of daily (BGS) file dates and load RTSW daily data files for each.
    for i_day, BGS_file_single_day_date in enumerate(BGS_file_day_dates,start=0):
        # ------------------------------------------- ACE mag data.
        #Define filename.
        ACE_mag_single_daily_filename = os.path.join(data_directory,'BGS_RTSW_Archive','ace',\
            str(BGS_file_single_day_date.year),'mag',str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + \
            str(BGS_file_single_day_date.day).zfill(2) + '_ace_mag_1m.txt')
        #End indenting for this variable.
        # Archived daily files of real-time solar wind data from the Advanced 
        # Composition Explorer (ACE) spacecraft.
        #   - Description: daily files of magnetometer measurements taken in
        #     real time from the ACE spacecraft and archived at the date of access. 
        #     These data have not been subjected to Level-2 processing.
        #   - Source: obtained from British Geological Survey (pers comm.). If the data
        #     need to be downloaded anew, then the top-level data description is at 
        #     https://www.swpc.noaa.gov/products/ace-real-time-solar-wind. As described 
        #     under the 'Data' tab on that page, the ACE real-time data are archived 
        #     back to August 2001 at 
        #     https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ (download files of 
        #     type 'YYYYMMDD_ace_mag_1m.txt', where 
        #     YYYY is year, MM is month and DD is day). The current real-time data can 
        #     be accessed at https://services.swpc.noaa.gov/text/ace-magnetometer.txt.
        #   - Format: The ACE magnetometer files named 'YYYYMMDD_ace_mag_1m.txt' are daily 
        #     ascii files of 1-min cadence data. There are 20 header lines, and 
        #     the remaining 1440 rows each have the following columns: [year, month, 
        #     day, time (HHMM), Modified Julian day, seconds within the day, data 
        #     status flag, Bx, By, Bz, Bt, Lat., Long.], where B(xyz) are the GSM 
        #     frame magnetic components, Bt is the total field in the GSM frame, and
        #     Lat. and Long. are the GSM coordinates. The missing data value is 
        #     -999.9.
        
        #Check for file existence.
        if(os.path.isfile(ACE_mag_single_daily_filename)):
            #If there's no file, then the dates will remain empty (hence 
            # not equalling the BGS dates), and the data will remain NaN, 
            # and will be assigned NaN again later when the dates are found 
            # to mismatch with the BGS dates
            
            #Open daily file of ACE mag data.
            fid = open(ACE_mag_single_daily_filename, 'r')
            
            #Read ACE file data.
            RTSW_ACE_single_day_mag_file_data = fid.readlines()#fid.readlines() gets all the content from the ascii file in one go, then we go through each line.
            
            #Close ACE daily data file.
            fid.close()
            
            #Loop over the daily file lines and extract variables of interest.
            RTSW_ACE_single_day_mag_times = np.empty([len(RTSW_ACE_single_day_mag_file_data)-20,1],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in day by 1]
            RTSW_ACE_single_day_mag_bx_data = np.empty([len(RTSW_ACE_single_day_mag_file_data)-20,1])#np array of floats, size [minutes in day by 1]
            RTSW_ACE_single_day_mag_by_data = np.empty([len(RTSW_ACE_single_day_mag_file_data)-20,1])#np array of floats, size [minutes in day by 1]
            RTSW_ACE_single_day_mag_bz_data = np.empty([len(RTSW_ACE_single_day_mag_file_data)-20,1])#np array of floats, size [minutes in day by 1]
            RTSW_ACE_single_day_mag_bx_data[:] = np.nan#NaN assigned to all entries.
            RTSW_ACE_single_day_mag_by_data[:] = np.nan
            RTSW_ACE_single_day_mag_bz_data[:] = np.nan
            i_line = 0
            for line in RTSW_ACE_single_day_mag_file_data:
                if(re.search('[:]',line) or re.search('[#]',line)):
                    continue#ignore 20 header lines containing these characters.
                line = line.strip()#remove trailing \n
                columns = line.split()#split line string into space-delimited strings.
                #Convert strings to datetime64[us] or float formats and store each.
                RTSW_ACE_single_day_mag_times[i_line,0] = np.datetime64(datetime.datetime.strptime(columns[0] + '-' + columns[1] + '-' + columns[2] + ',' + columns[3], '%Y-%m-%d,%H%M'))#storing scalar datetime object.
                RTSW_ACE_single_day_mag_bx_data[i_line,0] = np.float64(columns[7])#storing scalar float.
                RTSW_ACE_single_day_mag_by_data[i_line,0] = np.float64(columns[8])#storing scalar float.
                RTSW_ACE_single_day_mag_bz_data[i_line,0] = np.float64(columns[9])#storing scalar float.
                #Increment storage index
                i_line += 1
            #End loop over each 1-min line of the daily file.
            
            #There can be duplicate dates in RTSW_ACE_single_day_mag_times,
            # arising from errors in the stored RTSW archive. Here we flag 
            # them for removal, since they make the daily file have more than 1440 epochs!
            index_of_ACE_mag_times_to_remove = []#will be list of integer indices.
            for i_t in range(len(RTSW_ACE_single_day_mag_times)):
                #Define an index of where this time exists in the full set of times.
                index_same_time = np.nonzero(RTSW_ACE_single_day_mag_times == RTSW_ACE_single_day_mag_times[i_t])[0]#np array of ints, size [duplicate dates by 0]
                
                #If there's more than one of this time, flag the non-leading times(s) for later removal.
                if(len(index_same_time) > 1):
                    index_of_ACE_mag_times_to_remove = np.append(index_of_ACE_mag_times_to_remove,index_same_time[1:])
                #End conditional.
            #End loop over times.
            
            #If there are duplicate epochs to remove, then remove them, 
            # and tempoally re-sort the data afterwards.
            if(len(index_of_ACE_mag_times_to_remove) > 0):
                #Retrieve just the unique indices of the duplicate epochs. This is required 
                # since the progression through the entire set of times will, by definition, 
                # count each duplicate twice.
                index_of_ACE_mag_times_to_remove = np.transpose(np.unique(index_of_ACE_mag_times_to_remove)[np.newaxis].astype(np.int64))
                
                #Remove the duplicate epochs.
                RTSW_ACE_single_day_mag_times = np.delete(RTSW_ACE_single_day_mag_times,index_of_ACE_mag_times_to_remove,axis=0)
                RTSW_ACE_single_day_mag_bx_data = np.delete(RTSW_ACE_single_day_mag_bx_data,index_of_ACE_mag_times_to_remove,axis=0)
                RTSW_ACE_single_day_mag_by_data = np.delete(RTSW_ACE_single_day_mag_by_data,index_of_ACE_mag_times_to_remove,axis=0)
                RTSW_ACE_single_day_mag_bz_data = np.delete(RTSW_ACE_single_day_mag_bz_data,index_of_ACE_mag_times_to_remove,axis=0)
                
                #Re-sort the data, based on time.
                RTSW_ACE_single_day_mag_bx_data = RTSW_ACE_single_day_mag_bx_data[RTSW_ACE_single_day_mag_times[:,0].argsort()]
                RTSW_ACE_single_day_mag_by_data = RTSW_ACE_single_day_mag_by_data[RTSW_ACE_single_day_mag_times[:,0].argsort()]
                RTSW_ACE_single_day_mag_bz_data = RTSW_ACE_single_day_mag_bz_data[RTSW_ACE_single_day_mag_times[:,0].argsort()]
                RTSW_ACE_single_day_mag_times = RTSW_ACE_single_day_mag_times[RTSW_ACE_single_day_mag_times[:,0].argsort()]
            #End conditonal: are there duplicate epochs to remove?
            
            #Remove null values from the BGS ACE mag RTSW archive data: set them to nan.
            RTSW_ACE_single_day_mag_bx_data[np.where(RTSW_ACE_single_day_mag_bx_data == -999.9)] = np.NaN
            RTSW_ACE_single_day_mag_by_data[np.where(RTSW_ACE_single_day_mag_by_data == -999.9)] = np.NaN
            RTSW_ACE_single_day_mag_bz_data[np.where(RTSW_ACE_single_day_mag_bz_data == -999.9)] = np.NaN
            
            #Data check for if the file still has more than 1440 elements after removal of duplicate epochs
            if(len(RTSW_ACE_single_day_mag_times) > 1440):
                print('Warning: ACE mag daily file ' + str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + \
                    str(BGS_file_single_day_date.day).zfill(2) + ' contains data from other days.')
                #End indenting for print statement.
            #End conditional: data check.
            
            #Some files are not recorded for the full day. This is a problem, 
            # so we correct for that here. The signifier will be if the 
            # length of the loaded datafile is not equal to 1440 after the 
            # 20 header lines are discounted. 
            if(len(RTSW_ACE_single_day_mag_times) < 1440):
                #Need to re-load BGS daily file dates for this day. Use 
                # whichever station was last processed: the dates are identical
                # for all.
                
                #Define station file location string.
                BGS_file_location = os.path.join(data_directory,'BGS_external_data',str(BGS_file_single_day_date.year),\
                    BGS_station,BGS_station + str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + str(BGS_file_single_day_date.day).zfill(2) + '.ext')#string.
                #End indenting for this variable.
                
                #Open BGS daily file.
                fid = open(BGS_file_location, 'r')
                
                #Read BGS file data.
                BGS_file_data = fid.readlines()#fid.readlines() gets all the content from the ascii file in one go, then we go through each line.
                
                #Close BGS daily data file.
                fid.close()
                
                #Loop over the daily file lines and extract variables of interest.
                BGS_single_day_single_station_dates = np.empty([1440,1],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in day by 1]
                i_BGS_line = 0
                for line in BGS_file_data:
                    if(re.search('[|]',line)):
                        continue#ignore 26 header lines containing character '|'.
                    line = line.strip()#remove trailing \n
                    columns = line.split()#split line string into space-delimited strings.
                    #Convert strings to datetime64[us] or float formats and store each.
                    BGS_single_day_single_station_dates[i_BGS_line,0] = np.datetime64(datetime.datetime.strptime(columns[0] + ',' + columns[1], '%d-%m-%Y,%H:%M'))
                    #Increment storage index
                    i_BGS_line += 1
                #End loop over each 1-min line of the daily file.
                
                #Find the last matching epoch of ground-based and RTSW data.
                index_BGS_date_matching_last_RTSW_date = np.nonzero(BGS_single_day_single_station_dates == RTSW_ACE_single_day_mag_times[-1])[0][0]
                
                #Append the BGS times to the RTSW records from the last 
                # matching point onward, to complete the temporal record 
                # using the BGS date values.
                RTSW_ACE_single_day_mag_times = np.append(RTSW_ACE_single_day_mag_times,BGS_single_day_single_station_dates[index_BGS_date_matching_last_RTSW_date+1:])[:,np.newaxis]
                
                #And append NaN values to the RTSW data, to bring them up 
                # to the length of a full day.
                nan_filler_vector = np.empty([1440-index_BGS_date_matching_last_RTSW_date-1])
                nan_filler_vector[:] = np.nan
                RTSW_ACE_single_day_mag_bx_data = np.append(RTSW_ACE_single_day_mag_bx_data,nan_filler_vector)[:,np.newaxis]
                RTSW_ACE_single_day_mag_by_data = np.append(RTSW_ACE_single_day_mag_by_data,nan_filler_vector)[:,np.newaxis]
                RTSW_ACE_single_day_mag_bz_data = np.append(RTSW_ACE_single_day_mag_bz_data,nan_filler_vector)[:,np.newaxis]
            #End conditional: data check.
            
            #Store daily file of ACE data in the arrays of ACE data for all storm days.
            RTSW_ACE_storm_days_mag_dates[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = RTSW_ACE_single_day_mag_times
            RTSW_ACE_storm_days_Bx[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = RTSW_ACE_single_day_mag_bx_data
            RTSW_ACE_storm_days_By[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = RTSW_ACE_single_day_mag_by_data
            RTSW_ACE_storm_days_Bz[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = RTSW_ACE_single_day_mag_bz_data
        #End conditional: check for file existence.
        
        
        
        # ------------------------------------------- ACE swepam data.
        #Define filename.
        ACE_swepam_single_daily_filename = os.path.join(data_directory,'BGS_RTSW_Archive','ace',\
            str(BGS_file_single_day_date.year),'swepam',str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + \
            str(BGS_file_single_day_date.day).zfill(2) + '_ace_swepam_1m.txt')
        #End indenting for this variable.
        # Archived daily files of real-time solar wind data from the Advanced 
        # Composition Explorer (ACE) spacecraft.
        # - Description: daily files of plasma measurements taken in
        #   real time from the ACE spacecraft and archived at the date of access. 
        #   These data have not been subjected to Level-2 processing.
        # - Source: obtained from British Geological Survey (pers comm.). If the data
        #   need to be downloaded anew, then the top-level data description is at 
        #   https://www.swpc.noaa.gov/products/ace-real-time-solar-wind. As described 
        #   under the 'Data' tab on that page, the ACE real-time data are archived 
        #   back to August 2001 at 
        #   https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ (download files of 
        #   type 'YYYYMMDD_ace_swepam_1m.txt', where 
        #   YYYY is year, MM is month and DD is day). The current real-time data can 
        #   be accessed at https://services.swpc.noaa.gov/text/ace-swepam.txt.
        # - Format: The ACE swepam files named 'YYYYMMDD_ace_swepam_1m.txt' are daily 
        #   ascii files of 1-min cadence data. There are 18 header lines, and 
        #   the remaining 1440 rows each have the following columns: [year, month, 
        #   day, time (HHMM), Modified Julian day, seconds within the day, data 
        #   status flag, proton density, bulk speed, ion temperature], where the 
        #   last three columns pertain to the solar wind. The missing data values 
        #   are: Density and Speed = -9999.9, Temp. = -1.00e+05.
        
        #Check for file existence.
        if(os.path.isfile(ACE_swepam_single_daily_filename)):
            #If there's no file, then the dates will remain empty (hence 
            # not equalling the BGS dates), and the data will remain NaN, 
            # and wil be assigned NaN again later when the dates are found 
            # to mismatch with the BGS dates.
            
            #Open daily file of ACE swepam data.
            fid = open(ACE_swepam_single_daily_filename, 'r')
            #End indenting for this variable.
            
            #Read ACE file data.
            RTSW_ACE_single_day_swepam_file_data = fid.readlines()#fid.readlines() gets all the content from the ascii file in one go, then we go through each line.
            
            #Close ACE daily data file.
            fid.close()
            
            #Loop over the daily file lines and extract variables of interest: proton density and speed.
            RTSW_ACE_single_day_swepam_times = np.empty([len(RTSW_ACE_single_day_swepam_file_data)-18,1],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in day by 1]
            RTSW_ACE_single_day_swepam_speed_data = np.empty([len(RTSW_ACE_single_day_swepam_file_data)-18,1])#np array of floats, size [minutes in day by 1]
            RTSW_ACE_single_day_swepam_speed_data[:] = np.nan
            i_line = 0
            for line in RTSW_ACE_single_day_swepam_file_data:
                if(re.search('[:]',line) or re.search('[#]',line)):
                    continue#ignore 18 header lines containing these characters.
                line = line.strip()#remove trailing \n
                columns = line.split()#split line string into space-delimited strings.
                #Convert strings to datetime64[us] or float formats and store each.
                RTSW_ACE_single_day_swepam_times[i_line,0] = np.datetime64(datetime.datetime.strptime(columns[0] + '-' + columns[1] + '-' + columns[2] + ',' + columns[3], '%Y-%m-%d,%H%M'))
                RTSW_ACE_single_day_swepam_speed_data[i_line,0] = np.float64(columns[8])
                #Increment storage index
                i_line += 1
            #End loop over each 1-min line of the daily file.
            
            #There can be duplicate dates in RTSW_ACE_single_day_swepam_times,
            # arising from errors in the stored RTSW archive. here we flag 
            # them for removal.
            index_of_ACE_swepam_times_to_remove = []#will be list of integer indices.
            for i_t in range(len(RTSW_ACE_single_day_swepam_times)):
                #Define an index of where this time exists in the full set of times.
                index_same_time = np.nonzero(RTSW_ACE_single_day_swepam_times == RTSW_ACE_single_day_swepam_times[i_t])[0]#np array of ints, size [duplicate dates by 0]
                
                #If there's more than one of this time, flag the non-leading times(s) for later removal.
                if(len(index_same_time) > 1):
                    index_of_ACE_swepam_times_to_remove = np.append(index_of_ACE_swepam_times_to_remove,index_same_time[1:])
                #End conditional.
            #End loop over times.
            
            #If there are duplicate epochs to remove, then remove them, 
            # and temporally re-sort the data afterwards.
            if(len(index_of_ACE_swepam_times_to_remove) > 0):
                #Retrieve just the unique indices of the duplicate epochs. This is required 
                # since the progression through the entire set of times will, by definition, 
                # count each duplicate twice.
                index_of_ACE_swepam_times_to_remove = np.transpose(np.unique(index_of_ACE_swepam_times_to_remove)[np.newaxis].astype(np.int64))
                
                #Remove the duplicate epochs.
                RTSW_ACE_single_day_swepam_times = np.delete(RTSW_ACE_single_day_swepam_times,index_of_ACE_swepam_times_to_remove,axis=0)
                RTSW_ACE_single_day_swepam_speed_data = np.delete(RTSW_ACE_single_day_swepam_speed_data,index_of_ACE_swepam_times_to_remove,axis=0)
                
                #Re-sort the data, based on time.
                RTSW_ACE_single_day_swepam_speed_data = RTSW_ACE_single_day_swepam_speed_data[RTSW_ACE_single_day_swepam_times[:,0].argsort()]
                RTSW_ACE_single_day_swepam_times = RTSW_ACE_single_day_swepam_times[RTSW_ACE_single_day_swepam_times[:,0].argsort()]
            #End conditonal: are there duplicate epochs to remove?
            
            #Remove null values from the BGS ACE swepam RTSW archive data: set them to nan.
            RTSW_ACE_single_day_swepam_speed_data[np.where(RTSW_ACE_single_day_swepam_speed_data == -9999.9)] = np.NaN
            
            #Data check for if the file still has more than 1440 elements after removal of duplicate epochs
            if(len(RTSW_ACE_single_day_swepam_times) > 1440):
                print('Warning: ACE swepam daily file ' + str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + \
                    str(BGS_file_single_day_date.day).zfill(2) + ' contains data from other days. Expect errors.')
                #End indenting for print statement.
            #End conditional: data check.
            
            #Some files are not recorded for the full day. This is a problem, 
            # so we correct for that here. The signifier will be if the 
            # length of the loaded datafile is not equal to 1440 after the 
            # 18 header lines are discounted. 
            if(len(RTSW_ACE_single_day_swepam_times) < 1440):
                #Need to re-load BGS daily file dates for this day. Use 
                # whichever station was last processed: the dates are identical
                # for all.
                
                #Define station file location string.
                BGS_file_location = os.path.join(data_directory,'BGS_external_data',str(BGS_file_single_day_date.year),\
                    BGS_station,BGS_station + str(BGS_file_single_day_date.year) + str(BGS_file_single_day_date.month).zfill(2) + str(BGS_file_single_day_date.day).zfill(2) + '.ext')#string.
                #End indenting for this variable.
                
                #Open BGS daily file.
                fid = open(BGS_file_location, 'r')
                
                #Read BGS file data.
                BGS_file_data = fid.readlines()#fid.readlines() gets all the content from the ascii file in one go, then we go through each line.
                
                #Close BGS daily data file.
                fid.close()
                
                #Loop over the daily file lines and extract variables of interest.
                BGS_single_day_single_station_dates = np.empty([1440,1],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in day by 1]
                i_BGS_line = 0
                for line in BGS_file_data:
                    if(re.search('[|]',line)):
                        continue#ignore 26 header lines containing character '|'.
                    line = line.strip()#remove trailing \n
                    columns = line.split()#split line string into space-delimited strings.
                    #Convert strings to datetime64[us] or float formats and store each.
                    BGS_single_day_single_station_dates[i_BGS_line,0] = np.datetime64(datetime.datetime.strptime(columns[0] + ',' + columns[1], '%d-%m-%Y,%H:%M'))
                    #Increment storage index
                    i_BGS_line += 1
                #End loop over each 1-min line of the daily file.
                
                #Find the last matching epoch of ground-based and RTSW data.
                index_BGS_date_matching_last_RTSW_date = np.nonzero(BGS_single_day_single_station_dates == RTSW_ACE_single_day_swepam_times[-1])[0][0]
                
                #Append the BGS times to the RTSW records from the last 
                # matching point onward, to complete the temporal record 
                # using the BGS date values.
                RTSW_ACE_single_day_swepam_times = np.append(RTSW_ACE_single_day_swepam_times,BGS_single_day_single_station_dates[index_BGS_date_matching_last_RTSW_date+1:])[:,np.newaxis]
                
                #And append NaN values to the RTSW data, to bring them up 
                # to the length of a full day.
                nan_filler_vector = np.empty([1440-index_BGS_date_matching_last_RTSW_date-1])
                nan_filler_vector[:] = np.nan
                RTSW_ACE_single_day_swepam_speed_data = np.append(RTSW_ACE_single_day_swepam_speed_data,nan_filler_vector)[:,np.newaxis]
            #End conditional: data check.
            
            #Store daily file of ACE data in the arrays of ACE data for all storm days.
            RTSW_ACE_storm_days_swepam_dates[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = RTSW_ACE_single_day_swepam_times
            RTSW_ACE_storm_days_speed[(0 + (i_day * 1440)):(1440 + (i_day * 1440)),0,np.newaxis] = RTSW_ACE_single_day_swepam_speed_data
        #End conditional: check for file existence.
    #End loop over each storm day.
    
    #%%% Post-processing -- remove temporal errors in the BGS RSTW archive.
    
    #The BGS RTSW archive has instances of dates 
    # being in the wrong order, and sometimes for a different day. The 
    # RTSW mag/plasma data is not wrong per se, but it does pertain to the incorrect 
    # dates. Here we fix this by finding all instances where the ACE 
    # data don't match the BGS data in terms of time, and setting the
    # associated mag/plasma values to nan. Then we replace the ACE 
    # temporal records with the BGS station temporal records.
    #Check ACE mag time records for correspondence to BGS time records.
    index_mag_dates_in_wrong_place = []
    for i_t in range(len(RTSW_ACE_single_day_mag_times)):
        if(BGS_storm_days_all_stations_dates[i_t,0] != RTSW_ACE_storm_days_mag_dates[i_t][0]):
            index_mag_dates_in_wrong_place = np.append(index_mag_dates_in_wrong_place,i_t)
        #Conditional: check if the dates don't match for this temporal element of the data.
    #End loop over ACE mag times.
    
    #Set all mis-assigned ACE mag data to NaN.
    if(len(index_mag_dates_in_wrong_place) > 0):
        RTSW_ACE_storm_days_Bx[np.int64(index_mag_dates_in_wrong_place)] = np.nan
        RTSW_ACE_storm_days_By[np.int64(index_mag_dates_in_wrong_place)] = np.nan
        RTSW_ACE_storm_days_Bz[np.int64(index_mag_dates_in_wrong_place)] = np.nan
    #End conditional: check if list is empty before indexing with it.
    
    #Check ACE swepam time records for correspondence to BGS time records.
    index_swepam_dates_in_wrong_place = []
    for i_t in range(len(RTSW_ACE_single_day_swepam_times)):
        if(BGS_storm_days_all_stations_dates[i_t,0] != RTSW_ACE_storm_days_swepam_dates[i_t][0]):
            index_swepam_dates_in_wrong_place = np.append(index_swepam_dates_in_wrong_place,i_t)
        #Conditional: check if the dates don't match for this temporal element of the data.
    #End loop over ACE swepam times.
    
    #Set all mis-assigned ACE swepam data to NaN.
    if(len(index_swepam_dates_in_wrong_place) > 0):
        RTSW_ACE_storm_days_speed[np.int64(index_swepam_dates_in_wrong_place)] = np.nan
    #End conditional: check if list is empty before indexing with it.
    
    #Have the BGS temporal record for the storm days stand as the single 
    # temporal record for the ACE mag and swepam datasets.
    RTSW_ACE_storm_days_dates = BGS_storm_days_all_stations_dates[:,0,np.newaxis]
    
    #%%% Propagate the RSTW archive data from L1 to the Earth's bow shock nose.
    #Ballistically propagate ACE data from measurement location (Lagrange 1,
    # at ~220 Earth radii) to bow-shock of magnetosphere (12 Earth radii)
    # using solar wind speed. Projection is simple linear time = distance/speed.
    
    #First we need to interpolate over the (nan) gaps in the RTSW speed data, 
    # using the command '.astype(float)' to convert the datetime64 values to 
    # ordinal microseconds.
    RTSW_ACE_storm_days_speed_interpolated = np.interp(\
        RTSW_ACE_storm_days_dates[:,0].astype(float),\
        RTSW_ACE_storm_days_dates[np.nonzero(~np.isnan(RTSW_ACE_storm_days_speed))[0],0].astype(float),\
        RTSW_ACE_storm_days_speed[np.nonzero(~np.isnan(RTSW_ACE_storm_days_speed))[0],0])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0]
    #End indenting for variable definition.
    
    #Preallocate storage for propagated times.
    RTSW_ACE_storm_days_dates_ballistic_propagated = np.empty([len(RTSW_ACE_storm_days_dates),1],dtype='datetime64[s]')#np array of datetime64[s], size [minutes in storm days by 1]
    
    #Define constants.
    Earth_radius = 6371009.0
    metres_to_km = 1e-3
    
    #Loop over data records, and propagate each epoch to the bow shock nose.
    for i_t in range(len(RTSW_ACE_storm_days_dates)):
        #ACE RTSW speed data is in units of km/s, so we want the standoff 
        # propagation distance to be in km too, such that the propagation 
        # time will be in seconds.
        ballistic_propagation_in_seconds = \
            ((220 - 12) * (Earth_radius * metres_to_km)) / RTSW_ACE_storm_days_speed_interpolated[i_t]#scalar float, units of seconds.
        #End indenting for variable definition.
        
        #Store the propagated time.
        RTSW_ACE_storm_days_dates_ballistic_propagated[i_t] = RTSW_ACE_storm_days_dates[i_t] + np.timedelta64(np.int64(np.round(ballistic_propagation_in_seconds)),'s')#scalar datetime64[s] objct.
        
        #Check for 'NaT' values, caused by NaN values in the solar wind speed.
        if(pd.isnull(RTSW_ACE_storm_days_dates_ballistic_propagated[i_t])):
            print('Warning: NaT values encountered in temporal propagation to bow shock nose.')
        #End conditional: data check.
    #End loop over each RTSW storm days epoch.
    
    #Sort the RTSW data by the propagated time tags.
    RTSW_ACE_storm_days_Bx_prop_sorted = RTSW_ACE_storm_days_Bx[RTSW_ACE_storm_days_dates_ballistic_propagated[:,0].argsort()]#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1].
    RTSW_ACE_storm_days_By_prop_sorted = RTSW_ACE_storm_days_By[RTSW_ACE_storm_days_dates_ballistic_propagated[:,0].argsort()]#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1].
    RTSW_ACE_storm_days_Bz_prop_sorted = RTSW_ACE_storm_days_Bz[RTSW_ACE_storm_days_dates_ballistic_propagated[:,0].argsort()]#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1].
    RTSW_ACE_storm_days_speed_prop_sorted = RTSW_ACE_storm_days_speed[RTSW_ACE_storm_days_dates_ballistic_propagated[:,0].argsort()]#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1].
    RTSW_ACE_storm_days_dates_ballistic_propagated_sorted = RTSW_ACE_storm_days_dates_ballistic_propagated[RTSW_ACE_storm_days_dates_ballistic_propagated[:,0].argsort()]#np array of datetime64[s], size [(minutes in day)*(number of ACE daily files) by 1].
    
    #Interpolate the sorted data to a regular 1-min grid. For this, we use
    # the temporal record for the un-interpolated RTSW data for all storm 
    # days, which by this point in the code, is identical to the BGS 
    # temporal record for the storm days. note the conversion from 
    # nanoseconds to seconds precison for the 'RTSW_ACE_storm_days_dates' 
    # values.
    RTSW_ACE_storm_days_Bx_prop_sorted_gridded = np.interp(\
        RTSW_ACE_storm_days_dates[:,0].astype('datetime64[s]').astype(float),\
        RTSW_ACE_storm_days_dates_ballistic_propagated_sorted[:,0].astype(float),
        RTSW_ACE_storm_days_Bx_prop_sorted[:,0])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    #End indenting for interpolation function.
    RTSW_ACE_storm_days_By_prop_sorted_gridded = np.interp(\
        RTSW_ACE_storm_days_dates[:,0].astype('datetime64[s]').astype(float),\
        RTSW_ACE_storm_days_dates_ballistic_propagated_sorted[:,0].astype(float),
        RTSW_ACE_storm_days_By_prop_sorted[:,0])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    #End indenting for interpolation function.
    RTSW_ACE_storm_days_Bz_prop_sorted_gridded = np.interp(\
        RTSW_ACE_storm_days_dates[:,0].astype('datetime64[s]').astype(float),\
        RTSW_ACE_storm_days_dates_ballistic_propagated_sorted[:,0].astype(float),
        RTSW_ACE_storm_days_Bz_prop_sorted[:,0])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    #End indenting for interpolation function.
    RTSW_ACE_storm_days_speed_prop_sorted_gridded = np.interp(\
        RTSW_ACE_storm_days_dates[:,0].astype('datetime64[s]').astype(float),\
        RTSW_ACE_storm_days_dates_ballistic_propagated_sorted[:,0].astype(float),
        RTSW_ACE_storm_days_speed_prop_sorted[:,0])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    #End indenting for interpolation function.
    #Following the above processing, the RTSW data records (of type 
    # 'RTSW_ACE_storm_days_Bx_prop_sorted_gridded', etc.) will be like the 
    # OMNI data -- that is, temporally referenced to the bow shock nose, 
    # rather than the L1 measurement point. The temporal records that they 
    # are referenced to are in the RTSW_ACE_storm_days_dates variable.
    
    #%%% Compute epsilon from the RTSW data.
    
    #Set geometric constants.
    #Degrees to radians
    rad = np.pi / 180 #scalar
    #Radians to degrees
    deg = 180 / np.pi #scalar
    
    #Tips from here on angles and quadrants
    # 'http://www.mathworks.co.uk/matlabcentral/newsreader/view_thread/167016'
    # 'atan' only returns angles in the range -pi/2 and +pi/2.  'atan2'
    # implicitly divides the two inputs and will produce angles between -pi and
    # +pi.  If you want to convert these to angles between 0 and 2*pi, run:
    # mod(angle,2*pi). Strictly speaking, the part inside the brackets
    # should each be multipled by rad, but the ratio is used so it would
    # have no effect.
    IMF_clock_angle = np.empty((np.shape(RTSW_ACE_storm_days_By_prop_sorted_gridded)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    for i_t in range(len(RTSW_ACE_storm_days_By_prop_sorted_gridded)):
        IMF_clock_angle[i_t] = math.atan2(RTSW_ACE_storm_days_By_prop_sorted_gridded[i_t], RTSW_ACE_storm_days_Bz_prop_sorted_gridded[i_t]) * deg
    #End loop over RTSW archive times.
    #Note: here the clock angle spans -180 to 180, rather than 0 to 360. Not sure if it's an issue.
    
    #Calculate magnitude of full IMF vector.
    IMF_B_magnitude = np.empty((np.shape(RTSW_ACE_storm_days_By_prop_sorted_gridded)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    for i_t in range(len(RTSW_ACE_storm_days_By_prop_sorted_gridded)):
        IMF_B_magnitude[i_t] = math.sqrt((RTSW_ACE_storm_days_Bx_prop_sorted_gridded[i_t] ** 2) + (RTSW_ACE_storm_days_By_prop_sorted_gridded[i_t] ** 2) + (RTSW_ACE_storm_days_Bz_prop_sorted_gridded[i_t] ** 2))
    #End loop over RTSW archive times.
    
    #Convert from nT to Tesla.
    IMF_B_magnitude_in_Tesla = np.array(IMF_B_magnitude * (1e-9))#Converted to T, from nT. #np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    
    #Convert solar wind speed to SI units.
    speed_in_m_per_s = RTSW_ACE_storm_days_speed_prop_sorted_gridded * (1e3)#Converted to m/s from km/s. #np array of floats, size [(minutes in day)*(number of ACE daily files) by 0].
    
    #Assign a length-scale factor of 7Re.
    l_nought = 7 * (6371.2 * 1000) #scalar float.
    
    #Assign a factor for the amplitude of the permeability of free space, once 
    # 4 pi has been divided by mew_nought.
    four_pi_over_mew_nought = 1e7 #Note: 1/1e-7 = 1e7. #scalar float.
    
    #Form the epsilon coupling function.
    RTSW_ACE_storm_days_epsilon_prop_sorted_gridded = np.empty([np.shape(IMF_B_magnitude_in_Tesla)[0]])#np array of floats, size [(minutes in day)*(number of ACE daily files) by 1].
    for i_t in range(len(RTSW_ACE_storm_days_epsilon_prop_sorted_gridded)):
        RTSW_ACE_storm_days_epsilon_prop_sorted_gridded[i_t] = four_pi_over_mew_nought * speed_in_m_per_s[i_t] * (IMF_B_magnitude_in_Tesla[i_t] ** 2) * (math.sin((IMF_clock_angle[i_t] / 2) * rad) ** 4) * (l_nought ** 2)
    #Units: Watts.
    
    #%%% Apply lag times to the RTSW data
    #Important -- the RTSW data used here are temporally referenced to the 
    # bow shock nose, like the OMNI data. For ease of reference, I drop 
    # the '_prop_sorted_gridded' signifier for the lagged data, but note 
    # that it still applied. In short, these lag times are being applied 
    # as if they pertain to data recorded at the bow shock nose, not at L1.
    #Note also: as stated in cell 'Propagate the RSTW archive data from 
    # L1 to the Earth's bow shock nose.', the variable 
    # 'RTSW_ACE_storm_days_dates' has the correct temporal values to be 
    # used for the bow-shock-nose-referenced data
    
    #Preallocate empty array, fill with NaNs.
    RTSW_ACE_storm_days_Bx_lagged = np.empty((np.shape(RTSW_ACE_storm_days_Bx_prop_sorted_gridded)[0],np.shape(index_lags)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by number of lag types]
    RTSW_ACE_storm_days_Bx_lagged[:] = np.NaN
    RTSW_ACE_storm_days_By_lagged = np.empty((np.shape(RTSW_ACE_storm_days_By_prop_sorted_gridded)[0],np.shape(index_lags)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by number of lag types]
    RTSW_ACE_storm_days_By_lagged[:] = np.NaN
    RTSW_ACE_storm_days_Bz_lagged = np.empty((np.shape(RTSW_ACE_storm_days_Bz_prop_sorted_gridded)[0],np.shape(index_lags)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by number of lag types]
    RTSW_ACE_storm_days_Bz_lagged[:] = np.NaN
    RTSW_ACE_storm_days_speed_lagged = np.empty((np.shape(RTSW_ACE_storm_days_speed_prop_sorted_gridded)[0],np.shape(index_lags)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by number of lag types]
    RTSW_ACE_storm_days_speed_lagged[:] = np.NaN
    RTSW_ACE_storm_days_epsilon_lagged = np.empty((np.shape(RTSW_ACE_storm_days_epsilon_prop_sorted_gridded)[0],np.shape(index_lags)[0]))#np array of floats, size [(minutes in day)*(number of ACE daily files) by number of lag types]
    RTSW_ACE_storm_days_epsilon_lagged[:] = np.NaN
    #Loop over each lag instance.
    for i_lag in range(len(index_lags)):
        #Just a data check for the index and extraction being correct.
        #print(index_lags[i_lag])
        
        #Add (or subtract) the count of lag-minutes to the first epoch of the concantenated months of RTSW data.
        lagged_times_first_epoch = np.array(RTSW_ACE_storm_days_dates[0][0] + np.timedelta64(index_lags[i_lag],'m'))#np array of datetime64[s], size: scalar.
        
        #Find the element of the RTSW dates series that matches the current lag amount.
        #Note that if the lag amount is negative, this returns 0, the first element.
        lag_index = np.where(RTSW_ACE_storm_days_dates >= lagged_times_first_epoch)[0][0] #np scalar srray of int, pertains to rows of RTSW_ACE_epochs_storm.
        
        #Add the lag count of minutes to the first epoch of the RTSW time series.
        #If it's a negative lag, then we pad the end of the RTSW series with NaNs.
        #If it's a positive lag, then we pad the beginning of the RTSW series with NaNs.
        if(index_lags[i_lag] < 0):
            RTSW_ACE_storm_days_Bx_lagged[:-abs(index_lags[i_lag]),i_lag] = RTSW_ACE_storm_days_Bx_prop_sorted_gridded[abs(index_lags[i_lag]):]
            RTSW_ACE_storm_days_By_lagged[:-abs(index_lags[i_lag]),i_lag] = RTSW_ACE_storm_days_By_prop_sorted_gridded[abs(index_lags[i_lag]):]
            RTSW_ACE_storm_days_Bz_lagged[:-abs(index_lags[i_lag]),i_lag] = RTSW_ACE_storm_days_Bz_prop_sorted_gridded[abs(index_lags[i_lag]):]
            RTSW_ACE_storm_days_speed_lagged[:-abs(index_lags[i_lag]),i_lag] = RTSW_ACE_storm_days_speed_prop_sorted_gridded[abs(index_lags[i_lag]):]
            RTSW_ACE_storm_days_epsilon_lagged[:-abs(index_lags[i_lag]),i_lag] = RTSW_ACE_storm_days_epsilon_prop_sorted_gridded[abs(index_lags[i_lag]):]#newaxis needed for only epsilon because it has a different size to the other RTSW extractions.
        elif(index_lags[i_lag] == 0):
            RTSW_ACE_storm_days_Bx_lagged[:,i_lag] = RTSW_ACE_storm_days_Bx_prop_sorted_gridded[:]
            RTSW_ACE_storm_days_By_lagged[:,i_lag] = RTSW_ACE_storm_days_By_prop_sorted_gridded[:]
            RTSW_ACE_storm_days_Bz_lagged[:,i_lag] = RTSW_ACE_storm_days_Bz_prop_sorted_gridded[:]
            RTSW_ACE_storm_days_speed_lagged[:,i_lag] = RTSW_ACE_storm_days_speed_prop_sorted_gridded[:]
            RTSW_ACE_storm_days_epsilon_lagged[:,i_lag] = RTSW_ACE_storm_days_epsilon_prop_sorted_gridded[:]
        elif(index_lags[i_lag] > 0):
            RTSW_ACE_storm_days_Bx_lagged[lag_index:,i_lag] = RTSW_ACE_storm_days_Bx_prop_sorted_gridded[:-lag_index]#implicitly, NaNs are in the part missing from this fill-in.
            RTSW_ACE_storm_days_By_lagged[lag_index:,i_lag] = RTSW_ACE_storm_days_By_prop_sorted_gridded[:-lag_index]#implicitly, NaNs are in the part missing from this fill-in.
            RTSW_ACE_storm_days_Bz_lagged[lag_index:,i_lag] = RTSW_ACE_storm_days_Bz_prop_sorted_gridded[:-lag_index]#implicitly, NaNs are in the part missing from this fill-in.
            RTSW_ACE_storm_days_speed_lagged[lag_index:,i_lag] = RTSW_ACE_storm_days_speed_prop_sorted_gridded[:-lag_index]#implicitly, NaNs are in the part missing from this fill-in.
            RTSW_ACE_storm_days_epsilon_lagged[lag_index:,i_lag] = RTSW_ACE_storm_days_epsilon_prop_sorted_gridded[:-lag_index]#implicitly, NaNs are in the part missing from this fill-in.
        #End conditional: different lagging algorithms dependent on lag sign.
    #End loop over lags.
    
    #%%% Restrict the BGS and RTSW data from all stations to just the storm span, as determined by the Dst values.
    
    #Define an index pertaining to the rows of the BGS_storm_days_single_station_dates and BGS_storm_days_single_station_x_data/y/z
    # variables, which extracts the span of the storm as defined by Dst.
    #Use dates from esk to pertain to all other statons.
    index_storm_span_BGS = np.transpose(np.nonzero((BGS_storm_days_all_stations_dates[:,0] >= storm_start_date_hour_precision) & (BGS_storm_days_all_stations_dates[:,0] <= storm_end_date_hour_precision)))#np array of ints, size [Dst-determined storm span in minutes by 1]
    
    #Cut the BGS data down from the whole-days spanning the storm, to the 
    # storm's span at hour resolution, given by the Dst range.
    BGS_storm_span_all_stations_dates = BGS_storm_days_all_stations_dates[index_storm_span_BGS[:,0],:]#np array of datetime64[us], size [Dst-determined storm span in minutes by 1]
    BGS_storm_span_all_stations_local_time_hours = BGS_storm_days_all_stations_local_time_hours[index_storm_span_BGS[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_storm_span_all_stations_local_time_minutes = BGS_storm_days_all_stations_local_time_minutes[index_storm_span_BGS[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_storm_span_all_stations_local_time_seconds = BGS_storm_days_all_stations_local_time_seconds[index_storm_span_BGS[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_storm_span_all_stations_x_data = BGS_storm_days_all_stations_x_data[index_storm_span_BGS[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_storm_span_all_stations_y_data = BGS_storm_days_all_stations_y_data[index_storm_span_BGS[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_storm_span_all_stations_z_data = BGS_storm_days_all_stations_z_data[index_storm_span_BGS[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    
    #Define an index pertaining to the rows of RTSW_ACE_storm_days_epochs,
    # which extracts the span of the storm as defined by Dst.
    index_storm_span_RTSW = np.transpose(np.nonzero((RTSW_ACE_storm_days_dates[:,0] >= storm_start_date_hour_precision) & (RTSW_ACE_storm_days_dates[:,0] <= storm_end_date_hour_precision)))#np array of ints, size [Dst-determined storm span in minutes by 1]
    
    #Cut the RTSW data down from the whole-days spanning the storm, to the 
    # storm's span at hour resolution, given by the Dst range.
    RTSW_ACE_storm_span_epochs = RTSW_ACE_storm_days_dates[index_storm_span_RTSW[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by 1]
    RTSW_ACE_storm_span_Bx_lagged = RTSW_ACE_storm_days_Bx_lagged[index_storm_span_RTSW[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_storm_span_By_lagged = RTSW_ACE_storm_days_By_lagged[index_storm_span_RTSW[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_storm_span_Bz_lagged = RTSW_ACE_storm_days_Bz_lagged[index_storm_span_RTSW[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_storm_span_speed_lagged = RTSW_ACE_storm_days_speed_lagged[index_storm_span_RTSW[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_storm_span_epsilon_lagged = RTSW_ACE_storm_days_epsilon_lagged[index_storm_span_RTSW[:,0],:]#np array of floats, size [Dst-determined storm span in minutes by number of lags]
    
    #%%% Store the BGS data from this storm in the set of all storms.
    
    BGS_data_size_each_storm.append(np.array(np.shape(BGS_storm_span_all_stations_x_data)[0]))#size of stored element is scalar.
    BGS_data_dates_all_stations_all_storms.append(np.array(BGS_storm_span_all_stations_dates))#stored element is: np array of datetime64[us], size [Dst-determined storm span in minutes by 1]
    BGS_local_time_hours_all_stations_all_storms.append(np.array(BGS_storm_span_all_stations_local_time_hours))#stored element is: np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_local_time_minutes_all_stations_all_storms.append(np.array(BGS_storm_span_all_stations_local_time_minutes))#stored element is: np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_local_time_seconds_all_stations_all_storms.append(np.array(BGS_storm_span_all_stations_local_time_seconds))#stored element is: np array of floats, size [Dst-determined storm span in minutes by 1]
    BGS_data_all_stations_all_storms.append(np.array(np.concatenate((\
        np.expand_dims(BGS_storm_span_all_stations_x_data,axis=2),\
        np.expand_dims(BGS_storm_span_all_stations_y_data,axis=2),\
        np.expand_dims(BGS_storm_span_all_stations_z_data,axis=2)),axis=2)))#stored element is: np array of floats, size [Dst-determined storm span in minutes by 3 stations by 3 components]
    #End indenting for this command.
    
    #%%% Store the RTSW data from this storm in the set of all storms.
    
    RTSW_ACE_epochs_all_storms.append(np.array(RTSW_ACE_storm_span_epochs))#stored element is: np array of floats, size [Dst-determined storm span in minutes by 1]
    RTSW_ACE_Bx_lagged_all_storms.append(np.array(RTSW_ACE_storm_span_Bx_lagged))#stored element is: np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_By_lagged_all_storms.append(np.array(RTSW_ACE_storm_span_By_lagged))#stored element is: np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_Bz_lagged_all_storms.append(np.array(RTSW_ACE_storm_span_Bz_lagged))#stored element is: np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_speed_lagged_all_storms.append(np.array(RTSW_ACE_storm_span_speed_lagged))#stored element is: np array of floats, size [Dst-determined storm span in minutes by number of lags]
    RTSW_ACE_epsilon_lagged_all_storms.append(np.array(RTSW_ACE_storm_span_epsilon_lagged))#stored element is: np array of floats, size [Dst-determined storm span in minutes by number of lags]
#End loop over storms

print('... storm data loaded in list structures.')

#%% Define local time (LT) bins at 180 mins width and 60 mins cadence.

#Define LT bin width and LT bin cadence.
LT_bin_width = 180
LT_bin_cadence = 60#units of minutes.

#Define the centroids of the LT bins, in the form of datetimes throughout a 
# given day.
LT_bin_centroids_datetime = []#will be list of datetime objects of size [24].
#I choose the bin edge to be at 00:00 for a bin width which matches the bin 
# cadence (this will make it equal to earlier runs of 18-min contiguous bins).
# Hence the centroid of the first bin is at 00:09 for an 18-min cadence, and 
# 00:30 for a 60-min cadence.
date_x = datetime.datetime(2000,1,1) + datetime.timedelta(minutes=(LT_bin_cadence/2))#arbitrary scalar datetime object for starting date, at hour 0 on some given day, plus half the bin cadence.
LT_bin_centroids_datetime.append(date_x)
while (date_x + datetime.timedelta(minutes=LT_bin_cadence)) < datetime.datetime(2000,1,2):
    date_x += datetime.timedelta(minutes=LT_bin_cadence)
    LT_bin_centroids_datetime.append(date_x)#appends time in datetime format.
#End iteration over LT bin definitions within a sample day.


#Now define the edges of the bins by stepping out (both forward and backward 
# in time) from each bin's centroid. No seconds variable, because the spans 
# are at the 0th second. Convert to ordinal day fraction.
LT_bin_starts_day_fraction = np.empty([np.shape(LT_bin_centroids_datetime)[0],1])#np array of floats, size [LT bins by 1].
LT_bin_ends_day_fraction = np.empty([np.shape(LT_bin_centroids_datetime)[0],1])#np array of floats, size [LT bins by 1].
LT_bin_centroids_day_fraction = np.empty([np.shape(LT_bin_centroids_datetime)[0],1])#np array of floats, size [LT bins by 1].
LT_bin_starts_datetime = np.empty([np.shape(LT_bin_centroids_datetime)[0],1], dtype='datetime64[s]')#np array of datetime64[s], size [LT bins by 1].
LT_bin_ends_datetime = np.empty([np.shape(LT_bin_centroids_datetime)[0],1], dtype='datetime64[s]')#np array of datetime64[s], size [LT bins by 1].
seconds_per_day = 24*60*60# hours * mins * secs
for i_LT in range(len(LT_bin_centroids_datetime)):
    #Define temporary datetime objects for the starts and ends of this bin, 
    # for later conversion to decimal day format (within this cell).
    LT_bin_starts_datetime[i_LT] = LT_bin_centroids_datetime[i_LT] - datetime.timedelta(minutes=(LT_bin_width/2))
    LT_bin_ends_datetime[i_LT] =   LT_bin_centroids_datetime[i_LT] + datetime.timedelta(minutes=(LT_bin_width/2))
    
    #Create decimal day versions of the LT bin limits and centroids.
    LT_bin_starts_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_starts_datetime[i_LT][0].astype(datetime.datetime).time().hour, minutes=LT_bin_starts_datetime[i_LT][0].astype(datetime.datetime).time().minute, seconds=0).total_seconds()/seconds_per_day
    LT_bin_ends_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_ends_datetime[i_LT][0].astype(datetime.datetime).time().hour, minutes=LT_bin_ends_datetime[i_LT][0].astype(datetime.datetime).time().minute, seconds=0).total_seconds()/seconds_per_day
    LT_bin_centroids_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_centroids_datetime[i_LT].time().hour, minutes=LT_bin_centroids_datetime[i_LT].time().minute, seconds=0).total_seconds()/seconds_per_day
#End loop over LT bins.

#Manually alter the last element of the LT bin end hour to be 1, rather than 0.
# This occurs because the ending date of the LT_bin_edges variable is the start 
# (i.e. hour zero) of the next day, rather than hour 24 on the same day.
if(LT_bin_ends_day_fraction[len(LT_bin_ends_day_fraction)-1,0] == 0):
    LT_bin_ends_day_fraction[len(LT_bin_ends_day_fraction)-1,0] = 1.0
#End conditional: alter the last element of the LT bin ends if you need to.

#%% Index and remove the 'Pulkkinen storms' from the concatenated training/test dataset, and all similar datasets.
#The 'Pulkkinen storms' are those described in Pulkkinen et al. 2011
# (doi:10.1029/2010SW000600). These are removed from any instances of the 
# training ensemble (described later) in order that the models trained here can
# be tested on the storm events described in Pulkkinen et al. 2011.

#Loop over the Pulkkinen storms and remove them one by one.
for i_Pulkkinen_storm in range(3):
    #Manually specify the Pulkkinen storm epochs in the lists of concatenated storm data.
    if(i_Pulkkinen_storm == 0):
        index_Pulkkinen_storm = np.nonzero(np.all(np.concatenate((\
            (all_storms_epochs_set_sorted[:,0] == 2003)[:,np.newaxis],\
            (all_storms_epochs_set_sorted[:,1] == 10)[:,np.newaxis],\
            (all_storms_epochs_set_sorted[:,2] == 31)[:,np.newaxis],\
            ),axis=1),axis=1))[0][0]
        #End indenting.
    elif(i_Pulkkinen_storm == 1):
        index_Pulkkinen_storm = np.nonzero(np.all(np.concatenate((\
            (all_storms_epochs_set_sorted[:,0] == 2005)[:,np.newaxis],\
            (all_storms_epochs_set_sorted[:,1] == 8)[:,np.newaxis],\
            (all_storms_epochs_set_sorted[:,2] == 31)[:,np.newaxis],\
            ),axis=1),axis=1))[0][0]
        #End indenting.
    elif(i_Pulkkinen_storm == 2):
        index_Pulkkinen_storm = np.nonzero(np.all(np.concatenate((\
            (all_storms_epochs_set_sorted[:,0] == 2006)[:,np.newaxis],\
            (all_storms_epochs_set_sorted[:,1] == 12)[:,np.newaxis],\
            (all_storms_epochs_set_sorted[:,2] == 15)[:,np.newaxis],\
            ),axis=1),axis=1))[0][0]
        #End indenting.
    #End conditional: index storm within remaining storms epochs.
    
    #Remove the Pulkkinen storms from the lists of storm data that have been processed so far.
    del BGS_data_dates_all_stations_all_storms[index_Pulkkinen_storm]
    del BGS_local_time_hours_all_stations_all_storms[index_Pulkkinen_storm]
    del BGS_local_time_minutes_all_stations_all_storms[index_Pulkkinen_storm]
    del BGS_local_time_seconds_all_stations_all_storms[index_Pulkkinen_storm]
    del BGS_data_all_stations_all_storms[index_Pulkkinen_storm]
    del BGS_data_size_each_storm[index_Pulkkinen_storm]
    del RTSW_ACE_epochs_all_storms[index_Pulkkinen_storm]
    del RTSW_ACE_epsilon_lagged_all_storms[index_Pulkkinen_storm]
    all_storms_epochs_set_sorted = np.delete(all_storms_epochs_set_sorted,index_Pulkkinen_storm,0)
#End loop over Pulkkinen storms.

print('Pulkkinen storms removed from pre-randomised sets of training and test storms.')

#%% Start loop over randomised instances of training and test sets.
#The idea here is that we randomly divide the set of 
# all_storms_epochs_set_sorted with an 80%/20% ratio. We then train a model 
# with the 80% proportion, and test it on the 20% proportion. This randomised 
#training is repeated 100 times in order to arrive at 100 models, with 100 
# performance estimates.The randomised selection was performed once, by program 
# Randomised_Training_Regression_v2p0.py, and the same selection indices have 
# been loaded in below so that the same training sets can be generated here.
#
#Important note for later comprehension: at this point, the data in variables 
# of type 'BGS_data_all_stations_all_storms' (and similar) has the same
# count and order as the epochs in variable 'all_storms_epochs_set_sorted', 
# and may no longer relate to the order of variable 'all_storms_epochs_set'. 
# This is because changing the Dst threshold can alter which storms are  
# temporally unique.

#Load in the pre-randomised training and test set selection indices. The 
# original code used to generate these data is given below.
with open(os.path.join(project_directory,'GGF_Training_Model_Dataset_of_Indices_for_Ensemble_Storm_Epoch_Selection.pkl'),'rb') as f:
    index_for_storms_epochs_all_randomised_training_sets,\
    index_for_storms_epochs_all_randomised_test_sets = pickle.load(f)
#End indenting for this load command.
#The code used to generate the data in 
# GGF_Training_Model_Dataset_of_Indices_for_Ensemble_Storm_Epoch_Selection.pkl
# is as follows. It will need some manipulation to work properly here (i.e. 
# some of it is designed to operate outside of this loop over 
# i_random_storm_combination), but it provides a record of how the storm 
# epochs were randomly selected in each iteration over 
# i_random_storm_combination.
# # ----------------------
# #Run this code BEFORE the loop over i_random_storm_combination: 
# #Preallocate storage.
# index_for_storms_epochs_all_randomised_training_sets = np.empty([100,50])
# index_for_storms_epochs_all_randomised_test_sets = np.empty([100,12])
# # ----------------------
# #Run this code DURING the loop over i_random_storm_combination: 
# #Make a set of index fiducials which define the order of the sorted storm epochs.
# index_sorted_storms_ordering = np.array(list(range(len(all_storms_epochs_set_sorted))))#size [storms by 1]
# #
# #Define the training and test sets by dividing the number of storms up in 
# # an 80/20 ratio.
# np.random.shuffle(index_sorted_storms_ordering)#do NOT use the 'random' package for this: it makes duplicates when it should not!
# index_for_storms_epochs_randomised_training_set = index_sorted_storms_ordering[:round(len(index_sorted_storms_ordering) * 0.8)]#size of index is [50 by 0]
# index_for_storms_epochs_randomised_test_set = index_sorted_storms_ordering[round(len(index_sorted_storms_ordering) * 0.8):]#size of index is [12 by 0]
# #These indices pertain to rows of all_storms_epochs_set_sorted.
# #This produces 50 training storms and 12 test set storms of the total 62 
# # storms with are within the span of the BGS magnetometer data, and which 
# # are not overlapping at a pre-storm-peak extraction Dst threshold of
# # -30nT, and a post-storm peak extraction span of one day, and for which 
# # the storms in the Pulkkinen et al. 2011 paper have been removed.
# #Note: an important difference to the earlier, non-randomised training 
# # programs is that these storm epochs will no longer be temporally sorted.
# #
# #Store the the extraction fiducials.
# index_for_storms_epochs_all_randomised_training_sets[i_random_storm_combination,:] = index_for_storms_epochs_randomised_training_set
# index_for_storms_epochs_all_randomised_test_sets[i_random_storm_combination,:] = index_for_storms_epochs_randomised_test_set
# # ----------------------
# #Run this code AFTER the loop over i_random_storm_combination: 
# #Save out the extraction fiducials for all 100 randomised instances.
# with open(os.path.join(project_directory,'GGF_Training_Model_Dataset_of_Indices_for_Ensemble_Storm_Epoch_Selection.pkl'), 'wb') as f:
#     pickle.dump([\
#                   index_for_storms_epochs_all_randomised_training_sets,\
#                   index_for_storms_epochs_all_randomised_test_sets\
#                   ], f)
# #Technique, including loading saved data and saving multiple variables, is from
# #https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python.

#Preallocate storage.
all_randomised_storms_sets_PEs = np.empty([3,3,100])#np array of floats, size [stations by components by random storm sample iterations]
all_randomised_storms_sets_model_coeffs = np.empty([3,24,3,6,100])#size [stations by local time bins by components by model parameters by randomised training instances].

#Loop over 100 random iterations of the training and test storm sets.
for i_random_storm_combination in range (100):
    print('Iteration ' + str(i_random_storm_combination))
    #%% Define the training and test set storm epochs.
    
    #Extract the indices for the pre-randomised training and test sets for this randomised iteration.
    index_for_storms_epochs_randomised_training_set = np.int64(index_for_storms_epochs_all_randomised_training_sets[i_random_storm_combination,:])#size of extraction is [50 by 0].
    index_for_storms_epochs_randomised_test_set = np.int64(index_for_storms_epochs_all_randomised_test_sets[i_random_storm_combination,:])#size of extraction is [12 by 0].
    
    #Use the randomised indices to produce the epochs for the randomised 
    # training and test sets.
    storms_epochs_randomised_training_set = all_storms_epochs_set_sorted[index_for_storms_epochs_randomised_training_set,:]#size [training storms by 4]
    storms_epochs_randomised_test_set = all_storms_epochs_set_sorted[index_for_storms_epochs_randomised_test_set,:]#size [test storms by 4]
    
    #%% Concatenate training data, for each storm in this randomised training set.
    #The data are presently in lists of individual np arrays (one array per 
    # storm), and need to be combined into one array for all storms in this 
    # randomised training set. Here, we loop over the training set storms for 
    # this randomly-selected storms subset, extract the data for each storm, 
    # and concatenate the data in one large set of all training storm data.
    #Importantly, the data are stored in lists which have the same order as 
    # the rows of all_storms_epochs_set_sorted, so we use the same randomised 
    # indices applied to that variable to extract the relevant storms.
    
    #Enable preallocation of storage using the size of the training storms.
    training_storms_total_data_count = 0
    for i_training_storm in range(len(storms_epochs_randomised_training_set)):
        training_storms_total_data_count = training_storms_total_data_count + BGS_data_size_each_storm[index_for_storms_epochs_randomised_training_set[i_training_storm]]
    #End loop over training storms.
    
    #Preallocate storage.
    training_storms_epochs = np.empty([training_storms_total_data_count,3],dtype='datetime64[us]')#np array of datetime64[us], size [minutes in all concatenated training storms by 3 stations].
    training_storms_BGS_data = np.empty([training_storms_total_data_count,3,3])#will be np array of floats, size [minutes in all concatenated training storms by 3 stations by 3 components]
    training_storms_lagged_epsilon_data = np.empty([training_storms_total_data_count,len(index_lags)])#will be np array of floats, size [minutes in all concatenated training storms by number of lags]
    training_storms_local_time_hours = np.empty([training_storms_total_data_count,3])#will be np array of floats, size [minutes in all concatenated training storms by 3 stations]
    training_storms_local_time_minutes = np.empty([training_storms_total_data_count,3])#will be np array of floats, size [minutes in all concatenated training storms by 3 stations]
    training_storms_local_time_seconds = np.empty([training_storms_total_data_count,3])#will be np array of floats, size [minutes in all concatenated training storms by 3 stations]
    #Pre-populate with NaNs.
    training_storms_BGS_data[:] = np.nan
    training_storms_lagged_epsilon_data[:] = np.nan
    training_storms_local_time_hours[:] = np.nan
    training_storms_local_time_minutes[:] = np.nan
    training_storms_local_time_seconds[:] = np.nan
    
    #Loop over training storms, via their randomised indices, to extract and 
    # store their data from the corresponding list-storage. The iterator in 
    # this loop is a series of non-ordered randomised fiducials which index 
    # the list-elements of BGS_data_all_stations_all_storms, and related 
    # variables.
    storage_counter = 0
    for single_training_storm_index_fiducial in index_for_storms_epochs_randomised_training_set:
        #Extract and store the data for this storm. The station data includes 
        # all stations and components ,and the epsilon data includes all 
        # specified lags.
        training_storms_epochs[storage_counter:storage_counter+np.shape(BGS_data_dates_all_stations_all_storms[single_training_storm_index_fiducial])[0],:] = BGS_data_dates_all_stations_all_storms[single_training_storm_index_fiducial]#extracted element is np array of datetime64[us], size [minutes in storm by 3 stations].
        training_storms_BGS_data[storage_counter:storage_counter+np.shape(BGS_data_all_stations_all_storms[single_training_storm_index_fiducial])[0],:,:] = BGS_data_all_stations_all_storms[single_training_storm_index_fiducial]#extracted element is np array of floats, size [minutes in storm by stations by components].
        training_storms_lagged_epsilon_data[storage_counter:storage_counter+np.shape(RTSW_ACE_epsilon_lagged_all_storms[single_training_storm_index_fiducial])[0],:] = RTSW_ACE_epsilon_lagged_all_storms[single_training_storm_index_fiducial]#extracted element is np array of floats, size [minutes in storm by number of lag types].
        training_storms_local_time_hours[storage_counter:storage_counter+np.shape(BGS_local_time_hours_all_stations_all_storms[single_training_storm_index_fiducial])[0],:] = BGS_local_time_hours_all_stations_all_storms[single_training_storm_index_fiducial]#extracted element is np array of floats, size [minutes in storm by 3 stations]
        training_storms_local_time_minutes[storage_counter:storage_counter+np.shape(BGS_local_time_minutes_all_stations_all_storms[single_training_storm_index_fiducial])[0],:] = BGS_local_time_minutes_all_stations_all_storms[single_training_storm_index_fiducial]#extracted element is np array of floats, size [minutes in storm by 3 stations]
        training_storms_local_time_seconds[storage_counter:storage_counter+np.shape(BGS_local_time_seconds_all_stations_all_storms[single_training_storm_index_fiducial])[0],:] = BGS_local_time_seconds_all_stations_all_storms[single_training_storm_index_fiducial]#extracted element is np array of floats, size [minutes in storm by 3 stations]
        
        #Check data storage size: the station data and epsilon data must be of the same length.
        if(np.shape(BGS_data_all_stations_all_storms[single_training_storm_index_fiducial])[0] != np.shape(RTSW_ACE_epsilon_lagged_all_storms[single_training_storm_index_fiducial])[0]):
            print('Error in size of data extraction: station and epsilon data are of different sizes.')
            sys.exit()
        #End conditional: data check.
        
        #Advance the data storage counter.
        storage_counter = storage_counter + np.shape(BGS_data_all_stations_all_storms[single_training_storm_index_fiducial])[0]
    #End loop over training storms.
    
    
    #Local time computation for each element of the training data.
    #Convert BGS station local times to ordinal day fraction.
    training_storms_local_time_day_fraction = np.empty(np.shape(training_storms_local_time_hours))#np array of floats, size [minutes in all concatenated training storms by 3 stations]
    for i_col in range(np.shape(training_storms_local_time_hours)[1]):
        for i_row in range(np.shape(training_storms_local_time_hours)[0]):
            training_storms_local_time_day_fraction[i_row,i_col] = datetime.timedelta(\
                hours=training_storms_local_time_hours[i_row,i_col],\
                minutes=training_storms_local_time_minutes[i_row,i_col],\
                seconds=training_storms_local_time_seconds[i_row,i_col]).total_seconds()/seconds_per_day
        #End loop over rows (i.e. minutes in concatenated BGS data series).
    #End loop over columns (i.e. stations in concatenated BGS data series).
    #Approach from https://stackoverflow.com/questions/19813303/converting-time-to-fraction-of-days-in-python.
    
    
    #Extreme epsilon amplitude removal: set epsilon data to NaN if it exceeds 2.5e13.
    #Find the instances where the OMNI epsilon data exceeds 2.5e13 in ampliltude.
    index_epsilon_too_extreme_for_linear_modelling = np.nonzero(training_storms_lagged_epsilon_data > 2.5e13)[0]
    
    #Remove those instances from the epsilon data.
    training_storms_lagged_epsilon_data[index_epsilon_too_extreme_for_linear_modelling,:] = np.nan
    
    #%% Compute regression model: Aurroal oval boundary-independent, DOY-dependent regression.
    
    #Preallocate storage for model coefficients and misfit values.
    model_training_reg_coefs = np.empty([3,len(LT_bin_starts_day_fraction),3,6])#np array of floats, size [3 stations, 24 LTs, 3 components, 6 model parameters (columns of X matrix)].
    #Loop over stations.
    for i_station in range(3):
        #Loop over local time bins for data selection.
        for i_LT in range(len(LT_bin_starts_day_fraction)):
            #Find station data within this LT bin.
            if(LT_bin_starts_day_fraction[i_LT,0] > LT_bin_ends_day_fraction[i_LT,0]):
                index_part_1 = np.nonzero(training_storms_local_time_day_fraction[:,i_station] >= LT_bin_starts_day_fraction[i_LT,0])
                index_part_2 = np.nonzero(training_storms_local_time_day_fraction[:,i_station] < LT_bin_ends_day_fraction[i_LT,0])
                index_single_LT_bin_single_station = np.transpose(np.concatenate((index_part_1,index_part_2),axis=1))[:,0]
            else:
                index_single_LT_bin_single_station = np.transpose(np.nonzero((training_storms_local_time_day_fraction[:,i_station] >= LT_bin_starts_day_fraction[i_LT,0]) & \
                                                                             (training_storms_local_time_day_fraction[:,i_station] < LT_bin_ends_day_fraction[i_LT,0]))[0])#np array of ints, size [Dst-determined storm span minutes within this LT bin by 0]
            #End conditional: since we have overlapping LT bins, the ones which straddle the date-line need special treatment.
            #This index pertains to the rows of the 1-minute data concatenated over all training set storms.
            
            #Determine the (fractional) day of year for all data in this LT bin.
            dates_single_station_single_LT_bin = training_storms_epochs[index_single_LT_bin_single_station,i_station]
            day_fractions_single_station_single_LT_bin = training_storms_local_time_day_fraction[index_single_LT_bin_single_station,i_station]
            LT_bin_subset_DOYs = np.empty(np.shape(dates_single_station_single_LT_bin))
            for i_t_subset in range(len(LT_bin_subset_DOYs)):
                LT_bin_subset_DOYs[i_t_subset] = \
                    dates_single_station_single_LT_bin[i_t_subset].astype(datetime.datetime).timetuple().tm_yday \
                    + day_fractions_single_station_single_LT_bin[i_t_subset]
            #End loop over times (minutes) within this LT bins over all concatenated training set storms.
            
            #Loop over BGS data components.
            for i_component in range(3):
                #Extract a subset of the station data for this station and component.
                Y_data = np.array(training_storms_BGS_data[index_single_LT_bin_single_station,i_station,i_component])#np array of floats, size [Dst-determined storm span minutes within this LT bin by 0]
                
                #Make X matrix with the following columns.
                # -- sine of DOY
                # -- cosine of DOY
                # -- epsilon
                # -- (sine of DOY)*epsilon
                # -- (cosine of DOY)*epsilon
                #To which we will later add a bias unit.
                #And for which the epsilon is based on a subset of OMNI data 
                # which matches the epochs extracted for the BGS data in this 
                # LT bin. Indeed we use the BGS extraction index: equality of 
                # the OMNI and BGS dates was tested earlier.
                X_data = np.array(np.concatenate((\
                    np.sin(((LT_bin_subset_DOYs - 79) * (2 * np.pi))/365.25)[:,np.newaxis],\
                    np.cos(((LT_bin_subset_DOYs - 79) * (2 * np.pi))/365.25)[:,np.newaxis],\
                    training_storms_lagged_epsilon_data[index_single_LT_bin_single_station,index_lag][:,np.newaxis],\
                    np.sin(((LT_bin_subset_DOYs - 79) * (2 * np.pi))/365.25)[:,np.newaxis] * training_storms_lagged_epsilon_data[index_single_LT_bin_single_station,index_lag][:,np.newaxis],\
                    np.cos(((LT_bin_subset_DOYs - 79) * (2 * np.pi))/365.25)[:,np.newaxis] * training_storms_lagged_epsilon_data[index_single_LT_bin_single_station,index_lag][:,np.newaxis],\
                    ),axis=1))#np array of floats, size [Dst-determined storm span minutes within this LT bin by 5].
                #End indenting for this variable.
                #The variable 'X_data' is x in this equation of y = mx + a:
                #y = a + b*(sin(DOY)) + c*(cos(DOY)) + d*epsilon + e*(sin(DOY) * epsilon) + f*(cos(DOY) * epsilon)
                #Which can be rewritten as:
                #y = a + b*(sin(DOY)) + c*(cos(DOY)) + (d + e*sin(DOY) + f*cos(DOY))*epsilon
                #Note: we add on the parameter for the variable 'a' later, as the bias unit.
                
                #Find rows of X_data with NaN elements.
                index_X_is_NaN = np.transpose(np.nonzero(np.any(np.isnan(X_data),axis=1)))#np array of ints, size [NaNs elements of Dst-determined storm span minutes within this LT bin by 1].
                
                #Remove rows of X_data and Y_data for which there is no solar wind data.
                X_data = np.delete(X_data,index_X_is_NaN,axis=0)#new size [non-NaN elements of Dst-determined storm span minutes within this LT bin by 5]
                Y_data = np.delete(Y_data,index_X_is_NaN,axis=0)#new size [non-NaN elements of Dst-determined storm span minutes within this LT bin by 0]
                
                #Add singleton second dimension to Y_data.
                Y_data = Y_data[:,np.newaxis]#new size [non-NaN elements of Dst-determined storm span minutes within this LT bin by 1]
                
                #Preserve unscaled X_data.
                #X_data_unscaled = np.array(X_data)
                
                #Scale each column of X so that it has zero mean and standard 
                # deviation of 1.
                X_means = []#preservation of the X column means. np array of floats, will be of size [X matrix non-bias-unit columns by 0]
                X_std_devs = []#preservation of the mean-removed X column standard deviations. np array of floats, will be of size [X matrix non-bias-unit columns by 0]
                for i_col in range(np.shape(X_data)[1]):
                    #Remove the mean from each column of X.
                    X_means = np.append(X_means,np.mean(X_data[:,i_col]))
                    X_data[:,i_col,np.newaxis] = X_data[:,i_col,np.newaxis] - np.tile(np.mean(X_data[:,i_col]), (np.shape(X_data)[0], 1))
                    
                    #Divide each column of X by its standard deviation.
                    X_std_devs = np.append(X_std_devs,np.std(X_data[:,i_col]))
                    X_data[:,i_col,np.newaxis] = X_data[:,i_col,np.newaxis] / np.tile(np.std(X_data[:,i_col]), (np.shape(X_data)[0], 1))
                #End loop over non-bias-unit columns of X matrix.
                
                #Add in a bias unit as the first column of X_data.
                X_data = np.concatenate((np.ones([np.shape(X_data)[0],1]),X_data),axis = 1)#new size [non-NaN elements of Dst-determined storm span minutes within this LT bin by 2].
                
                #Compute scaled regression coefficients.
                #This is (X^T X)^-1 (X^T Y).
                reg_coef_scaled = np.transpose(np.linalg.lstsq(np.matmul(np.transpose(X_data),X_data), np.matmul(np.transpose(X_data),Y_data), rcond=None)[0])#np array of floatts, size [1 by columns of X matrix].
                #np.linalg.lstsq returns 4 arrays: the least-squares solution, the 
                # residuals, the rank, and the singular values. We pick the first.
                
                #Re-scale the regression coefficients using the
                # mean and standard deviation of the X matrix columns.
                #Approach from https://stats.stackexchange.com/questions/184209/multiple-regression-how-to-calculate-the-predicted-value-after-feature-normali,
                # re-worked in program C:\Users\robore\BAS_Files\Research\Code\SAGE\Testing_regression_scaling.py.
                #Assign empty array for scaled regression coefficients.
                regression_coefficients = np.empty(np.shape(reg_coef_scaled))#np array of floats, size [1 by number of X matrix columns].
                #Make initial assignation of intercept value, which will be altered later.
                regression_coefficients[0,0] = reg_coef_scaled[0][0]
                for i_col in range(1,np.shape(reg_coef_scaled)[1]):
                    #Rescale intercept using this regression coefficient.
                    regression_coefficients[0,0] = regression_coefficients[0,0] - ((reg_coef_scaled[0][i_col] * X_means[i_col-1]) / X_std_devs[i_col-1])
                    
                    #Rescale and store this regression coefficient.
                    regression_coefficients[0,i_col] = reg_coef_scaled[0][i_col] / X_std_devs[i_col-1]
                #End loop over non-bias-unit columns of X matrix.
                
                #Store regression outputs.
                for i_col in range(np.shape(X_data)[1]):
                    model_training_reg_coefs[i_station,i_LT,i_component,i_col] = regression_coefficients[0,i_col]
                #End loop over each regression coefficient.
            #End loop over BGS data components.
        #End loop over LT bins.
    #End loop over stations.
    
    #%% Store outputs for this random combination of storms.
    
    #Store the model coefficients.
    all_randomised_storms_sets_model_coeffs[:,:,:,:,i_random_storm_combination] = model_training_reg_coefs
    
#End loop over random storm combinations.

#%% Save out the model coefficients for all randomised training instances.

with open(os.path.join(project_directory,'GGF_Training_Model_Dataset_of_Stored_Model_Coefficients.pkl'),'wb') as f:
    pickle.dump([all_randomised_storms_sets_model_coeffs], f)
#End indenting for this command.

print('Program run complete. Model coefficients trained and saved.')
