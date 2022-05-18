# -*- coding: utf-8 -*-
"""
Created on Thu May 5 14:15:47 2022

Program name: 
 Substorm_Training_Model.py

Program objectives:
 This program reads in archives of solar wind and geomagnetic index data, and 
 based on a list of substorm onset epochs, trains a Convolutional Neural 
 Network (CNN) model to predict the likelihood (from 0 (least likely) to 1 
 (most likely)) of substorm onset in the following hour, when the trained 
 model is fed real-time solar wind data measured at the L1 Lagrange point. The 
 model architecture is described in the following paper: Maimaiti, 
 M., Kunduri, B., Ruohoniemi, J. M., Baker, J. B. H., & House, L. L. (2019). 
 A deep learning-based approach to forecast the onset of magnetic substorms. 
 Space Weather, 17, 1534– 1552. https://doi.org/10.1029/2019SW002251. 
 A key difference between the model described in that paper and the one trained
 here is that we use a different substorm list.  The Maimati et al. model is 
 trained on the list of substorms described in Newell, P. T., and Gjerloev, J. 
 W. (2011), Evaluation of SuperMAG auroral electrojet indices as indicators of 
 substorms and auroral power, J. Geophys. Res., 116, A12211, 
 doi:10.1029/2011JA016779. Conversely, the model trained in this program uses a
 processed set of the substorms identified by the Substorm Onsets and Phases 
 from Indices of the Electrojet (SOPHIE) technique, described in this paper: 
 Forsyth, C., Rae, I. J., Coxon, J. C., Freeman, M. P., Jackman, C. M., 
 Gjerloev, J., and Fazakerley, A. N. (2015), A new technique for determining 
 Substorm Onsets and Phases from Indices of the Electrojet (SOPHIE), J. 
 Geophys. Res. Space Physics, 120, 10,592– 10,606, doi:10.1002/2015JA021343.

Program details: 
 This code was not all written by Robert Shore (robore@bas.ac.uk, who is the 
 author of this metadata), hence the exact procedure of this program is not 
 known (although it could be deduced from the full code). Since the code is 
 sparsely commented, there follows a top-level summary of how the program 
 achieves the objectives stated above. The program assigns a set of half-hourly
 epochs between 1996/1/1 and 2014/12/31, each of which either starts on the 
 hour, or on the half-hour point. Each one of these epochs defines the start of 
 an hour-long span which either contains, or does not contain, a substorm. 
 Substorm incidence (and non-incidence) is given by the list of substorm 
 onsets, described below. For each half-hourly epoch, the preceding 2 hours of 
 solar wind data (from OMNI) is stored. The Convolutional Neural Network 
 architecture looks for patterns within each of these 2-hour spans of data, 
 which are 'learned' in order to allow it to predict whether the following hour
 contains a substorm or not. Hence, when applied to real time data, the program 
 is able to forecast the likelihood of substorm onset. Note that the program 
 divides the span from 1996/1/1 to 2014/12/31 into training, cross-validation, 
 and test sets. The training set spans 01/01/1996 to 24/01/2006. The 
 cross-validation set spans 24/01/2006 to 10/03/2010. The test set spans 
 10/03/2010 to 31/12/2014.

Input data requirements: each stored in project_directory/data: 
 - omn_mean_std.npy.
    - Description: a file of the mean and standard deviations of the OMNI solar
      wind data from 1996-01-01 to 2014-12-31, for the solar wind parameters 
      used to train the substorm onset likelihood forecast model.
    - Source: produced by this program, and stored in 'project_directory/data'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: NumPy array file format.
 - omni_sw_imf.sqlite.
    - Description: a large database of OMNI solar wind data at 1-min cadence, 
      spanning at least the period 1996-01-01 to 2014-12-31.
    - Source: the SQLite database was provided to the authors of this study by 
      Bharat Kunduri (bharatr@vt.edu) in a personal communication. However, 
      1-min OMNI data are publicly available from 
      https://omniweb.gsfc.nasa.gov/.
    - Is it included here: no. The data are provided on the BAS servers (at the
      location detailed in 'Instructions for running this program'), or can
      be downloaded from the above link and formatted using the details below.
    - Format: The SQLite database is a C binary file.  However, the Python 
      SQLite3 package is used here to convert the file into a pandas dataframe. 
      The resulting dataframe has 9269062 rows, comprising 1-min records from 
      1996-01-01 00:00:00 to 2014-12-31 00:00:00. The dataframe has the 
      following columns: [datetime, Bx, By, Bz, theta, Vx, Np, Pdyn, Temp, 
      Beta, MachNum, Timeshift], where datetime has the format 
      'YYYY-MM-DD hh:mm:ss, B(xyz) are the three GSM components of the 
      interplanetary magnetic field (IMF) (units of nanoTesla), theta is the 
      clock angle (the angle formed by the IMF vector in the Y-Z GSM plane), 
      Vx is the solar wind velocity in the GSM x-direction (units of km/s), Np 
      is the proton density (units of n/cc), Pdyn is the solar wind dynamic 
      pressure, Temp is either proton or electron temperature (unknown and not 
      used here), Beta is the plasma Beta (the ratio of the plasma pressure to 
      the magnetic pressure), MachNum is the Mach number of the solar wind, and 
      Timeshift is the shift applied to the OMNI data (measured at the L1 
      Lagrange point) to mimic a sampling point at the magnetopause bow shock 
      nose (units of seconds).
 - smu_sml_sme.sqlite
    - Description: a large database of SML, SMU and SME geomagnetic index data 
      at 1-min cadence, spanning at least the period 1996-01-01 to 2014-12-31.
    - Source: the SQLite database was provided to the authors of this study by 
      Bharat Kunduri (bharatr@vt.edu) in a personal communication. However, 
      1-min SML/SMU/SME data are publicly available from 
      https://supermag.jhuapl.edu/indices/. The derivation of the indices is 
      described here
      https://supermag.jhuapl.edu/indices/?fidelity=low&layers=SME.UL&tab=description.
    - Is it included here: no. The data are provided on the BAS servers (at the
      location detailed in 'Instructions for running this program'), or can
      be downloaded from the above link and formatted using the details below.
    - Format: The SQLite database is a C binary file.  However, the Python 
      SQLite3 package is used here to convert the file into a pandas dataframe. 
      The resulting dataframe has 9992161 rows, comprising 1-min records from 
      1996-01-01 00:00:00 to 2014-12-31 00:00:00. The dataframe has the 
      following columns: [datetime, ae, al, au], where datetime has the format 
      'YYYY-MM-DD hh:mm:ss, ae is the SME index, al is the SML index, and au is
      the SMU index, each with units of nanoTesla.
- SOPHIE_expansion_phase_onsets_EPT90iso_without_dSMLequalsdSMU_1to6hrWaitingTimes.csv.
    - Description: a file of substorm onset epochs originally sourced from the
      Supporting Information of this paper: Forsyth, C., Rae, I. J., Coxon, J.
      C., Freeman, M. P., Jackman, C. M., Gjerloev, J., and Fazakerley, A. N. 
      (2015), A new technique for determining Substorm Onsets and Phases from 
      Indices of the Electrojet (SOPHIE), J. Geophys. Res. Space Physics, 120, 
      10,592– 10,606, doi:10.1002/2015JA021343. Specifically, the file of 
      onsets sourced from there was 'jgra52264-sup-0003-supinfo.txt', which is 
      the set of onsets for which the rate of change of SML is above the 90th 
      percentile of all rates of change of that index (called the expansion 
      phase threshold, or 'EPT'). These EPT90 onset epochs were subjected to 
      additional processing steps, in order to reduce the substorm list to 
      those which are maximally likely to be actual substorm events (rather 
      than mis-identified nightside activity, or directly-driven events). Those
      processing steps were as follows.
       - The SOPHIE algorithm identifies the starting epochs of all possible 
         growth phases, substorm onsets (i.e. expansion phases), and recovery 
         phases. These are flagged 1 for possible growth, 2 for expansion, 3 
         for recovery. Of these, we retained only SOPHIE expansion phase onset 
         epochs which are directly preceded by a possible growth phase. In 
         other words, we retain only the epochs of the start of the '2' phases
         which were preceded by a '1' phase.
       - SOPHIE expansion phase onsets for which the rate of change of SMU is 
         similar to that of SML in the expansion phase are removed.  This step 
         is taken because the removed expansion phases are more likely to be 
         directly-driven events (i.e. intensifications of the 2-cell system) 
         and less likely to be substorms. Full details of the rate-of-change-
         based identification are in Forsyth et al. 2015.
       - After removal of the SOPHIE list onsets detailed in the above points, 
         the waiting times (inter-event intervals) for the remaining onsets are 
         computed. Our studies (not included here) have shown that onsets with 
         waiting times between 1 and 6 hours offer the highest predictability 
         by the onset forecast model. Hence, at this point we remove all onset 
         epochs with waiting times less than 1 hour or greater than 6 hours.
       - The remaining expansion phase onset epochs were then saved out to the 
         csv file, the format of which is given below.
    - Source: as described above. Stored in 'project_directory/data'.
    - Is it included here: yes. These data do not need to be downloaded 
      because the file is provided in the same directory as this program.
    - Format: the csv file has one header rows and 11,009 data rows. The header
      gives the format of the three comma-separated columns, which are 
      [Date_UTC,MLAT,MLT], where 'Date_UTC' gives the epoch of substorm onset 
      (i.e. the start of the expansion phase) in the format 'DD/MM/YYYY hh:mm',
      e.g. '24/01/1996 12:32'. 'MLAT' and 'MLT' are respectively the magnetic 
      latitude and magnetic local time of the onset location. These columns are
      expected by the program, but these values are not provided by the SOPHIE
      list, and hence all values in these two columns are zero, and the program
      does not make use of the onset location.

Subroutines: 
 Each of the following subroutine programs is stored in directory 
 'project_directory/data_pipeline': these are run automatically by program 
 'Substorm_Training_Model.py' and are not meant to be run independently.
  - batch_utils.py
  - create_onset_data.py
  - omn_utils.py
  - dnn_classifiers.py

Program outputs: the following outputs are stored in directory 
 'project_directory/trained_model':
 - loss_acc.png
    - Description: an image of the chart of model loss and accuracy metrics, 
      computed at each training 'epoch' (i.e. each training iteration) for both
      the training and cross-validation sets.
    - Source: produced by program 'Substorm_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: PNG image file.
 - resnet_all_data_pred.csv
    - Description: comma-separated value file of the model's predictions of 
      substorm incidence and non-incidence in the hour following each epoch.
    - Source: produced by program 'Substorm_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: the file has one header row (which describes the column format) 
      and 29,158 data rows, each of which has the following column format: 
      [epoch,label,label,del_minutes,pred_label,prob_0,prob_1], where epoch is 
      the start time of each hour-long interval (described above in 'Program 
      details'), and has the format 'YYYY-MM-DD hh:mm:ss' 
      (e.g. 2010-03-14 12:30:00). The two duplicated 'label' columns indicate 
      whether the hour following the epoch time contains a substorm (0 for no 
      substorm, 1 for substorm). The 'del_minutes' column indicates the number
      of minutes between the epoch of the start of the hour-long interval and 
      the substorm onset epoch within that hour-long interval. If there is no 
      substorm, 'del_minutes' has a value of -1. The 'pred_label' column 
      indicates whether the model predicts there will be a substorm 
      (pred_label=1) or not substorm (pred_label=0) in the hour following the 
      epoch time. The 'prob_0' column is the predicted likelihood (from 0; 
      least likely, to 1; most likely) that there will be  no substorm in the 
      hour following the epoch time. The 'prob_1' column is the predicted 
      likelihood (from 0; least likely, to 1; most likely) that there is a 
      substorm in the hour following the epoch time. The larger value of prob_0 
      and prob_1 for a controls the value of pred_label for a given row. 
      Lastly, note that the non-substorm intervals have been downsampled to 
      match the count of substorm intervals, so the interval epochs no longer 
      have a continuous half-hourly cadence.
 - resnet_test_data_pred.csv
    - Description: comma-separated value file of the model's predictions of 
      substorm incidence and non-incidence in the hour following each epoch, 
      for the test data set only.
    - Source: produced by program 'Substorm_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: same as resnet_all_data_pred.csv, but with 4,516 data rows.
 - resnet_train_data_pred.csv
    - Description: comma-separated value file of the model's predictions of 
      substorm incidence and non-incidence in the hour following each epoch, 
      for the training data set only.
    - Source: produced by program 'Substorm_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: same as resnet_all_data_pred.csv, but with 17,830 data rows.
 - resnet_val_data_pred.csv
    - Description: comma-separated value file of the model's predictions of 
      substorm incidence and non-incidence in the hour following each epoch, 
      for the cross-validation data set only.
    - Source: produced by program 'Substorm_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: same as resnet_all_data_pred.csv, but with 6,812 data rows.
 - weights.epoch_[###].val_loss_[#.##].val_accuracy_[#.##].hdf5
    - Description: a series of 40 files of model coefficients trained by the 
      convolutional neural network. The model's loss and accuracy statistics 
      vary each time it is run, hence the values for 'val_loss' and 
      'val_accuracy' in the file name are represented by '[#.##]'. The model 
      outputs one coefficients file each 5 training epochs. The coefficients 
      which should be used for forecasting are those of the 200th epoch (or a
      model from an earlier epoch, if this has better performance statistics). 
      For example, the file saved here is called 
      'weights.epoch_200.val_loss_0.59.val_accuracy_0.70.hdf5'. The other 39 
      files were not retained, but can be produced upon running this program.
    - Source: produced by program 'Substorm_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: Hierarchical Data Format HDF5.

Instructions for running this program:
 This program is intended to be run on a machine with access to the British 
 Antarctic Survey (BAS)'s data servers. The variable 'project_directory' and 
 'data_directory' should be set manually by the user, dependent on where the 
 data are stored. If running outside of BAS, the program should be able to
 be run if the user downloads all the data files mentioned above, and 
 replicates (or alters) the file structure used throughout this program. At 
 the time of writing, the code and data are located in the following 
 directories.
 - On the BAS Linux servers:
    - The code is in /data/psdcomplexity/eimf/SAGE_Model_Handover/Substorm.
    - The data are in /data/psdcomplexity/eimf/SAGE_BGS/From_Bharat/sqlite3.
 - On the BAS machine bslthemesa:
    - The directory /data/psdcomplexity/eimf is linked to via /local/users/robore/shortcut_EIMF_data.
 - On the local hard drive of the BAS laptop used by robore@bas.ac.uk, i.e. 
   where these programs were developed:
    - The code is in C:/Users/robore/BAS_Files/Research/Code/SAGE/SAGE_Model_Handover/Substorm.
    - The data are in C:/Users/robore/BAS_Data/SAGE_BGS/From_Bharat/sqlite3.

@author: robore@bas.ac.uk. Robert Shore, ORCID: orcid.org/0000-0002-8386-1425.
For author's reference: this program was based on the programs in 
 C:/Users/robore/BAS_Files/Research/Code/SAGE/Substorm_SOPHIE_ReTraining.
"""

#%% Load and process the solar wind data.

#Define the top-level directory location.
import os
project_directory = os.path.join(os.sep,'local','users','robore','shortcut_EIMF_data','SAGE_Model_Handover','Substorm')#string: /local/users/robore/shortcut_EIMF_data/SAGE_Model_Handover/Substorm.
data_directory = os.path.join(os.sep,'local','users','robore','shortcut_EIMF_data','SAGE_BGS','From_Bharat','sqlite3')#string: /local/users/robore/shortcut_EIMF_data/SAGE_BGS/From_Bharat/sqlite3.
#Dear user: if running on a BAS machine, please edit the above strings to be
# project_directory = '/data/psdcomplexity/eimf/SAGE_Model_Handover/Substorm'.
# data_directory = '/data/psdcomplexity/eimf/SAGE_BGS/From_Bharat/sqlite3'.

#Import remaining packages.
import sys
sys.path.insert(0, os.path.join(project_directory,'data_pipeline'))
import batch_utils
import time
import numpy as np
import pandas as pd
import datetime as dt

#Set options.
omn_dbdir = data_directory
omn_db_name = "omni_sw_imf.sqlite"
omn_table_name = "imf_sw"
omn_norm_param_file = os.path.join(project_directory,'data','omn_mean_std.npy')
include_omn = True
omnTrainParams = ["Bx", "By", "Bz", "Vx", "Np"]
imfNormalize = True
omn_train = True#If the switch is set to true, it assumes you're trying to train the model and it'll normalise the input IMF data based on their values.  If set to false, it assumes you're trying to predict the model and hence it uses the same mean and standard deviation computed from the data used to train the model, which it loads from the 'omn_norm_param_file'.
shuffleData = False
polarData=True
imageData=True
omnHistory = 120
batch_size = 1
onsetDelTCutoff = 4
onsetFillTimeRes = 30
omnDBRes = 1
binTimeRes = 60
nBins = 1
predList=["bin", "del_minutes"] #If a substorm occurs in the hour following the 2-hour training data span, then the number of minutes between the end of the training interval and the substorm onset epoch is assigned to the 'del_minutes' label in subroutine 'create_onset_data.py'. 
loadPreComputedOnset = False
saveBinData = False 
onsetSaveFile = os.path.join(project_directory,'data','binned_data.feather')
useSML = True 
include_sml = False
sml_normalize = True
sml_train = False
sml_train_params = ["au", "al"]
sml_db_name = "smu_sml_sme.sqlite"
sml_table_name = "smusmlsme"
sml_norm_param_file = os.path.join(omn_dbdir,'sml_mean_std.npy')
omn_time_delay = 10
smlDownsample=False
smlDateRange = [ dt.datetime(1996,1,1), dt.datetime(2014,12,31) ] #
smlStrtStr = smlDateRange[0].strftime("%Y%m%d")
smlEndStr = smlDateRange[1].strftime("%Y%m%d")

#This routine loads in the data.
batchObj = batch_utils.DataUtils(omn_dbdir,\
                    omn_db_name, omn_table_name,\
                    omn_train, omn_norm_param_file, useSML=useSML, imfNormalize=imfNormalize, omnDBRes=omnDBRes,\
                    omnTrainParams=omnTrainParams,\
                    include_omn=include_omn, 
                    sml_train=sml_train, sml_norm_file=sml_norm_param_file,
                    smlDbName=sml_db_name, smlTabName=sml_table_name,
                    sml_normalize=sml_normalize,include_sml=include_sml, 
                    sml_train_params=sml_train_params,
                    batch_size=batch_size, loadPreComputedOnset=loadPreComputedOnset,\
                    onsetDataloadFile = os.path.join(project_directory,'data','binned_data.feather'),\
                    northData=True, southData=False, polarData=polarData,\
                    imageData=imageData, polarFile = os.path.join(project_directory,'data','polar_data.feather'),\
                    imageFile = os.path.join(project_directory,'data','image_data.feather'), onsetDelTCutoff=onsetDelTCutoff,\
                    onsetFillTimeRes=onsetFillTimeRes, binTimeRes=binTimeRes, nBins=nBins,\
                    saveBinData=saveBinData, onsetSaveFile=onsetSaveFile,\
                    shuffleData=shuffleData, omnHistory=omnHistory, smlDateRange=smlDateRange,
                    smlDownsample=smlDownsample, omn_time_delay=omn_time_delay)
#End indenting for this variable.

x = time.time()
onsetData_list = []
omnData_list = []
dtms = []
for _bat in batchObj.batchDict.keys():
    # get the corresponding input (omnData) and output (onsetData)
    # for this particular batch!
    dtm = batchObj.batchDict[_bat][0]
    onsetData = batchObj.onset_from_batch(batchObj.batchDict[_bat], predList=predList)
    omnData = batchObj.omn_from_batch(batchObj.batchDict[_bat])
    if omnData is not None:
        onsetData_list.append(onsetData[0])
        omnData_list.append(omnData)
        dtms.append(dtm)
y = time.time() 
print("inOmn calc--->", y-x)

# Process the output
input_data = np.vstack(omnData_list)
output_data = np.vstack(onsetData_list)

# Save datetimes and output labels
col_dct = {}
lbl = 0
for b in range(nBins):
    col_dct[str(b*binTimeRes) + "_" + str((b+1)*binTimeRes)] = output_data[:, b].astype(int).tolist()
    lbl = lbl + (2**(nBins-1-b)) * output_data[:, b].astype(int)
    
col_dct["label"] = lbl
col_dct["del_minutes"] = output_data[:, -1].tolist()
df = pd.DataFrame(data=col_dct, index=dtms)

#%% Train the convolutional neural network model.

#Import packages.
import keras
#from keras.callbacks import ModelCheckpoint
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
from dnn_classifiers import ResNet, train_model
#from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import glob

#Define options.
batch_size = 16 * 4 * 1
n_epochs = 200
n_resnet_units = 2
metrics = ["accuracy"]
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "au":5, "al":6}
input_cols = [param_col_dict[x] for x in omnTrainParams]

#Define write-out location
out_dir = os.path.join(project_directory,'trained_model')

#Reprocess the data.
X = input_data
y = df.loc[:, "label"].values.reshape(-1, 1)

# Select certain columns
X = X[:, :, input_cols]

########################
## Limit the time history
X = X[:, -omnHistory-1:, :]
########################

#Create the training, cross-validation and test sets.
npoints = X.shape[0]
n_classes = np.unique(y).shape[0]
train_size = 0.60
val_size = 0.22
test_size = 0.18
train_eindex = int(npoints * train_size)
val_eindex = train_eindex + int(npoints * val_size)
x_train = X[:train_eindex, :]
x_val = X[train_eindex:val_eindex, :]
x_test = X[val_eindex:, :]
y_train = y[:train_eindex, :]
y_val = y[train_eindex:val_eindex, :]
y_test = y[val_eindex:, :]

df_train = df.iloc[:train_eindex, :]
df_val = df.iloc[train_eindex:val_eindex, :]
df_test = df.iloc[val_eindex:, :]

##################################
#Balance the two classes in train data
ss_idx = np.where(df_train.label.values == 1)[0]
nonss_idx = np.where(df_train.label.values == 0)[0]
np.random.seed(1)
if(len(ss_idx) > len(nonss_idx)):
    ss_idx = np.random.choice(ss_idx, len(nonss_idx), replace=False)
else:
    nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
#End conditional: ensure that you downsample the larger class.
event_idx = np.concatenate([ss_idx, nonss_idx])
#Keep the order of data points the same as before balancing
event_idx.sort()
df_train = df_train.iloc[event_idx, :]
#Select for certain rows and columns
x_train = x_train[event_idx]
y_train = y_train[event_idx]

#Balance the two classes in val data
ss_idx = np.where(df_val.label.values == 1)[0]
nonss_idx = np.where(df_val.label.values == 0)[0]
np.random.seed(1)
if(len(ss_idx) > len(nonss_idx)):
    ss_idx = np.random.choice(ss_idx, len(nonss_idx), replace=False)
else:
    nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
#End conditional: ensure that you downsample the larger class.
event_idx = np.concatenate([ss_idx, nonss_idx])
#Keep the order of data points the same as before balancing
event_idx.sort()
df_val = df_val.iloc[event_idx, :]
#Select for certain rows and columns
x_val = x_val[event_idx]
y_val = y_val[event_idx]

#Balance the two classes in test data
ss_idx = np.where(df_test.label.values == 1)[0]
nonss_idx = np.where(df_test.label.values == 0)[0]
np.random.seed(1)
if(len(ss_idx) > len(nonss_idx)):
    ss_idx = np.random.choice(ss_idx, len(nonss_idx), replace=False)
else:
    nonss_idx = np.random.choice(nonss_idx, len(ss_idx), replace=False)
#End conditional: ensure that you downsample the larger class.
event_idx = np.concatenate([ss_idx, nonss_idx])
#Keep the order of data points the same as before balancing
event_idx.sort()
df_test = df_test.iloc[event_idx, :]
#Select for certain rows and columns
x_test = x_test[event_idx]
y_test = y_test[event_idx]

df = pd.concat([df_train, df_val, df_test])
X = np.concatenate([x_train, x_val, x_test])
y = np.concatenate([y_train, y_val, y_test])
##################################

#Encode the labels
enc = OneHotEncoder()
unique_labels = np.unique(y, axis=0)
enc.fit(unique_labels)
y_train_enc = enc.transform(y_train).toarray()
y_test_enc = enc.transform(y_test).toarray()
y_val_enc = enc.transform(y_val).toarray()
y_enc = enc.transform(y).toarray()

#Build a ResNet model
optimizer=keras.optimizers.Adam(lr=0.00001)
input_shape = X.shape[1:]

#Define the loss, loss_weights, and class_weights
loss=keras.losses.categorical_crossentropy
class_weights = None

#Train the model
dl_obj = ResNet(input_shape, batch_size=batch_size, n_epochs=n_epochs,
                n_classes=n_classes, n_resnet_units=n_resnet_units, loss=loss,
                optimizer=optimizer,
                metrics=metrics, out_dir=out_dir)

print("Training the model...")
dl_obj.model.summary()
fit_history = train_model(dl_obj.model, x_train, y_train_enc, x_val, y_val_enc,
                          batch_size=batch_size, n_epochs=n_epochs,
                          callbacks=dl_obj.callbacks, shuffle=True,
                          class_weights=class_weights)



# Plot the loss curve and the prediction accuracy throughout the training process.
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
xs = np.arange(n_epochs)
train_loss = fit_history.history["loss"]
val_loss = fit_history.history["val_loss"]
train_acc = fit_history.history["accuracy"]
val_accuracy = fit_history.history["val_accuracy"]
axes[0].plot(xs, train_loss, label="train_loss") 
axes[0].plot(xs, val_loss, label="val_loss") 
axes[1].plot(xs, train_acc, label="train_acc") 
axes[1].plot(xs, val_accuracy, label="val_accuracy") 
axes[0].set_title("Training Loss and Accuracy")
axes[0].set_ylabel("Loss")
axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Epoch")
axes[0].legend()
axes[1].legend()
fig_path = os.path.join(out_dir, "loss_acc")
fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")  

# Evaluate the model on test dataset
print("Evaluating the model...")
test_epoch = n_epochs# The epoch number of the model we want to evaluate
if test_epoch < 10:
    model_name = glob.glob(os.path.join(out_dir, "weights.epoch_0" + str(test_epoch) + "*hdf5"))[0]
else:
    model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name) 

# Make predictions
y_train_pred_enc = test_model.predict(x_train, batch_size=batch_size)
y_val_pred_enc = test_model.predict(x_val, batch_size=batch_size)
y_test_pred_enc = test_model.predict(x_test, batch_size=batch_size)
y_pred_enc = test_model.predict(X, batch_size=batch_size)

# The final activation layer uses softmax
y_train_pred = np.argmax(y_train_pred_enc , axis=1)
y_val_pred = np.argmax(y_val_pred_enc , axis=1)
y_test_pred = np.argmax(y_test_pred_enc , axis=1)
y_pred = np.argmax(y_pred_enc , axis=1)
y_train_true = y_train
y_val_true = y_val
y_test_true = y_test

# Report for train data
print("Prediction report for train input data.")
print(classification_report(y_train_true, y_train_pred))

# Report for validation data
print("Prediction report for validation input data.")
print(classification_report(y_val_true, y_val_pred))

# Report for test data
print("Prediction report for test data.")
print(classification_report(y_test_true, y_test_pred))

# Save the predicted outputs
#out_dir = "./trained_models/MLP_iso"
output_df_file =       os.path.join(out_dir, "resnet_all_data_pred.csv")
output_df_train_file = os.path.join(out_dir, "resnet_train_data_pred.csv")
output_df_val_file =   os.path.join(out_dir, "resnet_val_data_pred.csv")
output_df_test_file =  os.path.join(out_dir, "resnet_test_data_pred.csv")

df.loc[:, "pred_label"] = y_pred
df_train.loc[:, "pred_label"] = y_train_pred
df_val.loc[:, "pred_label"] = y_val_pred
df_test.loc[:, "pred_label"] = y_test_pred

for i in range(y_pred_enc.shape[1]):
    df.loc[:, "prob_"+str(i)] = y_pred_enc[:, i]
    df_train.loc[:, "prob_"+str(i)] = y_train_pred_enc[:, i]
    df_val.loc[:, "prob_"+str(i)] = y_val_pred_enc[:, i]
    df_test.loc[:, "prob_"+str(i)] = y_test_pred_enc[:, i]

df.to_csv(output_df_file)
df_train.to_csv(output_df_train_file)
df_val.to_csv(output_df_val_file)
df_test.to_csv(output_df_test_file)


