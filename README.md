# swimmr_sage
Training and forecast code for real-time predictions of the UK ground geomagnetic field, and substorm onset likelihood

This directory and its subdirectories contian four primary programs which underpin the SWIMMR-SAGE modelling to date (2022-05-13 at the time of writing).

The programs are as follows: 
 - SAGE_Model_Handover/GGF/GGF_Training_Model.py. This program trains the ground geomagnetic field (GGF) forecast models.
 - SAGE_Model_Handover/Substorm/Substorm_Training_Model.py. This program trains the substorm onset likelihood forecast model.
 - SAGE_Model_Handover/Forecast/GGF_RTF.py. This program forecasts ground geomagnetic field in real time, for about an hour into the future.
 - SAGE_Model_Handover/Forecast/GGF_RTFH.py. This program forecasts ground geomagnetic field in real time, for about an hour into the future, and provides a hindcast for 24 hours into the past.

More detailed descriptions of each of these programs is found within the docstrings (i.e. header comments) of the code files themselves. These metadata descriptions also describe how to run the programs.

Files of Python packages are provided in order to facilitate the replication of the environment in which the code was developed and run. They are: 
 - conda_requirements.txt: this file was produced with the command:
   conda list -e > conda_requirements.txt
   It can be used to create a conda virtual environment with:
   conda create --name <env> --file conda_requirements.txt
 - pip_requirements.txt: this file was produced with the command: 
   pip list --format=freeze > pip_requirements.txt
   It can be used to create a pip virtual environment with: 
   python3 -m venv env
   source env/bin/activate
   pip install -r pip_requirements.txt


