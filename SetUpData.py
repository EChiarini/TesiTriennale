#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import re
import shutil


# In[ ]:


SAMPLE_RATE = 13
SAMPLE_FREQUENCY = round(1000/SAMPLE_RATE, 6)
SAMPLE_FREQUENCY_STR = str(SAMPLE_FREQUENCY)
script_dir = os.getcwd() + '/data/'


# In[ ]:


def inizializza():
    df = pd.read_csv(script_dir + "raw_data_all.csv")

    for (userid, activity, position), group in df.groupby(["Userid", "Activity", "position"]):
        user_folder = f"User{userid}"
        user_folder_path = os.path.join(script_dir, user_folder)

        os.makedirs(user_folder_path, exist_ok=True)  # <-- crea X/data/UserX/

        filename = f"User{userid}_{activity}_Sensor-{position}.csv"
        filepath = os.path.join(user_folder_path, filename)  # <-- salva dentro UserX/

        group.to_csv(filepath, index=False)

    print("Tutto ok")


# In[ ]:


# Import data as nested dictionary
def get_data(df_dict, user):
    current_user = user
    #mypath = 'C:/Users/Andrea/Documents/1Uni/Tesi_magistrale/Raccolta_e_analisi_dati/Dati/' + current_user +'/'
    #mypath = 'C:/Users/emili/Desktop/Tirocinio/Activity_Recognition/data/' + current_user +'/'
    mypath = script_dir + current_user + '/'
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and re.compile(current_user).match(f)]
    activites = set([file.split('_')[1] for file in file_list])
    sensors = set([file.split('_')[2].split('-')[1].split('.')[0] for file in file_list])
    df_dict = {activity: {sensor: pd.DataFrame() for sensor in sensors} for activity in activites}
    for file in file_list:
        df = pd.read_csv(mypath + file)

        # check for nan values
        nan_values = df.iloc[:, :17].isnull().values.any()
        if nan_values:
            df1 = df[df.isna().any(axis=1)]
            print("File: " + file + " has nan values\nDataFrame:")
        activity = file.split('_')[1]
        sensor = file.split('_')[2].split('-')[1].split('.')[0]
        df_dict[activity][sensor] = \
            pd.concat([df_dict[activity][sensor], df]).reset_index(drop=True)
    return df_dict


# In[ ]:


# Preprocess data
def preprocess_data(df_dict):
    for activity, df_activity in df_dict.items():
        for sensor, df_sensor in df_activity.items():
            discarded_columns = \
            pd.concat([df_sensor.iloc[:, 0:6], df_sensor.iloc[:, 16]], axis=1)
            # Convert timestamps to DateTime format
            #display(df_sensor.columns.tolist())
            df_sensor['Timestamp'] = \
                pd.to_datetime(
                    df_sensor['UTC_start-end'][0] \
                        + (df_sensor['Timestamp']\
                        -df_sensor['relativeTime_start-end'][0])*1000, 
                            unit='us')
            # Align all dataframes to the same timestamps
            df_sensor = df_sensor.iloc[:,6:16].resample(SAMPLE_FREQUENCY_STR+'ms', on='Timestamp').mean().dropna().reset_index()
            #if (df_sensor.isnull().values.any()):
                #print("Activity: " + activity + " Sensor: " + sensor + " has nan values")
                #display(df_sensor[df_sensor.isna().any(axis=1)])
            # Reindex discarded_columns to match the length of df_sensor
            discarded_columns = discarded_columns.reindex(df_sensor.index)
            # Add back the columns that were discarded
            df_sensor = pd.concat([df_sensor, discarded_columns], axis=1).dropna().reset_index(drop=True)
            # Save the modified dataframe
            df_activity[sensor] = df_sensor

        # Align first and last probes for all the dataframes
        initial_timestamps = [df_sensor['Timestamp'].iloc[0] for df_sensor in df_activity.values()]
        final_timestamps = [df_sensor['Timestamp'].iloc[-1] for df_sensor in df_activity.values()]
        max_initial_timestamp = max(initial_timestamps)
        min_final_timestamp = min(final_timestamps)
        for sensor, df_sensor in df_activity.items():
            # Discard data that is not within the range of the first and last probe
            df_sensor = df_sensor[(df_sensor['Timestamp'] > max_initial_timestamp) & (df_sensor['Timestamp'] < min_final_timestamp)]
            df_activity[sensor] = df_sensor.reset_index(drop=True)
    return df_dict



# ### Group Data With Sliding Window

# In[ ]:


def get_frames(data, frame_size, hop_size):
    r = np.arange(len(data))   
    s = r[::hop_size]   
    z = list(zip(s, s + frame_size))   
    g = lambda hop_size: data.iloc[hop_size[0]:hop_size[1]]   
    return pd.concat(map(g, z), keys=range(len(z)))


# ##### Calculate magnitude between data from the same IMU sensor

# In[ ]:


def calculate_magnitude(df_dict):
    for activity, df_activity in df_dict.items():
        for sensor, df_sensor in df_activity.items():
            df_sensor = pd.concat([
                df_sensor,
                np.sqrt(np.square(df_sensor[['AccX', 'AccY', 'AccZ']]).sum(axis=1)).rename("AccMagnitude"),
                np.sqrt(np.square(df_sensor[['GyroX', 'GyroY', 'GyroZ']]).sum(axis=1)).rename("GyroMagnitude"),
                np.sqrt(np.square(df_sensor[['MagnX', 'MagnY','MagnZ']]).sum(axis=1)).rename("MagnMagnitude"),
            ], axis=1)
            df_activity[sensor] = df_sensor
    return df_dict


# #### Raw data dataframe

# In[ ]:


# Merge raw data in single dataframe
def get_raw_data(df_dict):
    df_raw_data = pd.DataFrame()
    for activity, df_activity in df_dict.items():
        for sensor, df_sensor in df_activity.items():
            df_raw_data = pd.concat([df_raw_data, df_sensor]).reset_index(drop=True)
    return df_raw_data


# ### Calculate 1% Range Over/In/Below Mean

# In[ ]:


def calc_over_in_below_mean(df, perc=0.01):
    # Create an empty dataframe to store the results
    result_df = pd.DataFrame()
    df_mean = df.mean()

    # Calculate the lower and upper limits
    df_lim_inf = df_mean.apply(lambda x: x - x*perc)
    df_lim_sup = df_mean.apply(lambda x: x + x*perc)

    # Create a new dataframe to count the number of values over, in and below the mean
    for col in df.columns:
        result_df[col + 'OverMean'] = pd.Series((df[col] > df_lim_sup[col]).value_counts().get(True, 0))
        result_df[col + 'InMean'] = pd.Series(((df[col] >= df_lim_inf[col]) & (df[col] <= df_lim_sup[col])).value_counts().get(True, 0))
        result_df[col + 'BelowMean'] = pd.Series((df[col] < df_lim_inf[col]).value_counts().get(True, 0))

    result_df = result_df.reset_index(drop=True)

    return result_df


# ### Merge Data in Single DataFrame

# In[ ]:


# Merge all the dataframes into a single dataframe
def merge_data(df_dict):
    df_data = pd.DataFrame()
    for df_activity in df_dict.values():
        for df_sensor in df_activity.values():
            df_data = pd.concat([df_data, df_sensor])
    return df_data


# # Features Extraction

# Features:
# * STD
# * avg
# * min
# * max
# * above/in/below range
# 

# In[ ]:


def calculate_fft_energy(frame, signal_len, df_energy):
    fft_result = np.fft.fft(frame)
    power_spectrum = np.abs(fft_result)**2
    power_spectrum /= signal_len
    energy = np.sum(power_spectrum, axis=0)
    energy = pd.DataFrame([energy], columns=df_energy.columns)
    df_energy = pd.concat([df_energy, energy], ignore_index=True)
    return df_energy


# In[ ]:


frame_size = SAMPLE_RATE*4  # Window:  4 seconds
hop_size = SAMPLE_RATE*2  

# Calculate features for each frame
def calculate_features(df_dict):
  # Overlap: 50%
    for df_activity in df_dict.values():
        for sensor, df_sensor in df_activity.items():
            df_energy = pd.DataFrame(columns=[
                "AccxEnergy", "AccyEnergy", "AcczEnergy", 
                "GyroxEnergy", "GyroyEnergy", "GyrozEnergy", 
                "MagnxEnergy", "MagnyEnergy", "MagnzEnergy", 
                "AccMagnitudeEnergy", "GyroMagnitudeEnergy", "MagnMagnitudeEnergy"
            ])
            # Get frames
            df_sensor = get_frames(df_sensor, frame_size, hop_size)
            # Save columns that are going to be discarded
            discarded_columns = pd.concat([df_sensor.iloc[:, 0], df_sensor.iloc[:, 10:-3]], axis=1).reset_index(drop=True)
            # Discard columns that are not going to be used for grouping
            df_sensor = df_sensor[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'MagnX', 'MagnY', 'MagnZ', 'AccMagnitude', 'GyroMagnitude', 'MagnMagnitude']]
            # Calculate Energy Spectral Density for each frame and separately for each axis
            df_energy = df_sensor.groupby(level=0).apply(lambda x: calculate_fft_energy(x, frame_size, df_energy)).reset_index(drop=True)
            # Calculate mean for each frame
            df_mean = df_sensor.groupby(level=0).mean().reset_index(drop=True)
            df_mean.columns = [col + 'Mean' for col in df_mean.columns]
            # Calculate std
            df_std = df_sensor.groupby(level=0).std().reset_index(drop=True)
            df_std.columns = [col + 'Std' for col in df_std.columns]
            # Calculate min
            df_min = df_sensor.groupby(level=0).min().reset_index(drop=True)
            df_min.columns = [col + 'Min' for col in df_min.columns]
            # Calculate max
            df_max = df_sensor.groupby(level=0).max().reset_index(drop=True)
            df_max.columns = [col + 'Max' for col in df_max.columns]
            # Calculate number of values in, over and below 1% of the mean
            df_over_in_below_mean = df_sensor.groupby(level=0).apply(lambda x: calc_over_in_below_mean(x)).reset_index(drop=True)
            # Add back the columns that were discarded and the calculated energy
            df_activity[sensor] = pd.concat([df_mean, discarded_columns, df_energy, df_std, df_min, df_max, df_over_in_below_mean], axis=1).dropna().reset_index(drop=True)

    df_data = merge_data(df_dict)
    return df_data


# Write grouped data and raw data on file

# In[ ]:


users = [
     'User0',
     'User1',
     'User2',
     'User3',
     'User4',
     'User5',
     'User6',
     'User7',
     'User8',
     'User9',
 ]

inizializza()
for user in tqdm(users):
    df_dict = {}
    df_dict = get_data(df_dict, user)
    #df_dict = preprocess_data(df_dict)
    df_raw_data = get_raw_data(df_dict) #faccio nulla
    df_dict = calculate_magnitude(df_dict)
    df_data = calculate_features(df_dict)
    # Save the data
    os.makedirs(script_dir + "Processed_data", exist_ok=True)
    df_data.to_csv(script_dir + "Processed_data/" + 'grouped_data_' + user + '.csv')


# In[ ]:


def cleanUp():
  for item in os.listdir(script_dir):
    item_path = os.path.join(script_dir, item)

    if os.path.isdir(item_path) and re.fullmatch(r'User\d+', item):
      shutil.rmtree(item_path)


# In[ ]:


cleanUp();

