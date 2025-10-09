import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy import signal
from scipy.signal import butter, filtfilt, resample, resample_poly

BASE_DATA_PATH = os.getcwd() + '/data/selfBACK_raw'
OUTPUT_PATH = os.getcwd() + '/data/selfBACK_processed_data'

SAMPLE_RATE = 25   
WINDOW_SECONDS = 4  
HOP_SECONDS = 2     
FRAME_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS) 
HOP_SIZE = int(SAMPLE_RATE * HOP_SECONDS)    

def downsample_with_antialiasing(df, original_fs=100, target_fs=SAMPLE_RATE, cols=['x', 'y', 'z']):
    factor = original_fs // target_fs
    nyquist_new = target_fs / 2  
    
    
    
    cutoff = nyquist_new * 0.9  
    b, a = butter(N=4, Wn=cutoff, fs=original_fs, btype='low')
    
    df_filtered = df.copy()
    
    
    for col in cols:
        if col in df.columns:
            
            df_filtered[col] = filtfilt(b, a, df[col].values)
    
    
    return df_filtered.iloc[::factor].reset_index(drop=True)

def modified_load_sensor_data(base_path, participant_id, activity, sensor_type):
    filename = f"0{participant_id}.csv"
    file_path = base_path + "/" + sensor_type + "/" + activity + "/" + filename
    try:
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], names=['timestamp', 'x', 'y', 'z'])
        if df.isnull().values.any():
            df.dropna(inplace=True)
        
        
        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        
        df.dropna(subset=['x', 'y', 'z'], inplace=True)
            
        

        df = downsample_with_antialiasing(df, original_fs=100, target_fs=SAMPLE_RATE, cols=['x', 'y', 'z'])
            
        return df
    except Exception as e:
        print(f"Errore {file_path}: {e}")
        return None

def modified_load_wt_data(wt_path, participant_id, activity_wt_filename):
    filename = f"0{participant_id}_{activity_wt_filename}"
    file_path = os.path.join(wt_path+'/wt/', filename)
    
    try:
        df = pd.read_csv(file_path, header=None, names=['x_wrist', 'y_wrist', 'z_wrist', 'x_thigh', 'y_thigh', 'z_thigh'])
        if df.isnull().values.any():
            df.dropna(inplace=True)
        
        
        numeric_cols = ['x_wrist', 'y_wrist', 'z_wrist', 'x_thigh', 'y_thigh', 'z_thigh']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        
        df.dropna(subset=numeric_cols, inplace=True)
            

        df = downsample_with_antialiasing(df, original_fs=100, target_fs=SAMPLE_RATE, cols=numeric_cols)
            
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Errore caricando {file_path}: {e}")
        return None


  

UTENTI = [p for p in range(26, 63) if p not in [32,35,37,38,45]] 
ATTIVITA = [
    "upstairs", "downstairs", "walk_slow", "walk_mod", "walk_fast",
    "jogging", "standing", "sitting", "lying"
]


ATTIVITA_MAP_WT = {
    "upstairs": "upstairs",
    "downstairs": "downstairs",
    "walkslow": "walk_slow",   
    "walkmod": "walk_mod",    
    "walkfast": "walk_fast",  
    "jogging": "jogging",
    "standing": "standing",
    "sitting": "sitting",
    "lying": "lying"
}
ATTIVITA_WT_FILENAMES = list(ATTIVITA_MAP_WT.keys()) 


SENSORI = ['w', 't', 'wt']

COLONNE_W_T = ['timestamp', 'x', 'y', 'z']
COLONNE_WT = ['x_wrist', 'y_wrist', 'z_wrist', 'x_thigh', 'y_thigh', 'z_thigh']





def load_sensor_data(base_path, participant_id, activity, sensor_type):
    filename = f"0{participant_id}.csv"
    file_path = base_path + "/" + sensor_type + "/" + activity + "/" + filename
    try:
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], names=COLONNE_W_T)
        if df.isnull().values.any():
            df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Errore {file_path}: {e}")
        return None





def load_wt_data(wt_path, participant_id, activity_wt_filename):
    filename = f"0{participant_id}_{activity_wt_filename}"
    file_path = os.path.join(wt_path+'/wt/', filename)

    try:
        df = pd.read_csv(file_path, header=None, names=COLONNE_WT)
        if df.isnull().values.any():
            df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Errore caricando {file_path}: {e}")
        return None





def get_frames(data, frame_size, hop_size):
    r = np.arange(len(data))
    s = r[::hop_size]

    window_dentro = s[s + frame_size <= len(data)]

    z = list(zip(window_dentro, window_dentro + frame_size))
    g = lambda indices: data.iloc[indices[0]:indices[1]]

    return pd.concat(map(g, z), keys=range(len(z)))





def calculate_acc_magnitude(df, prefix=""):
    suffix = f"_{prefix}" if prefix else ""
    x_col, y_col, z_col = f"x{suffix}", f"y{suffix}", f"z{suffix}"
    output_col = f"AccMagnitude{suffix}"


    if all(col in df.columns for col in [x_col, y_col, z_col]):
        df[[x_col, y_col, z_col]] = df[[x_col, y_col, z_col]].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=[x_col, y_col, z_col], inplace=True)
        df[output_col] = np.sqrt(np.square(df[[x_col, y_col, z_col]]).sum(axis=1))
    else:
        print(f"Colonne {x_col}, {y_col}, {z_col} non trovate in calculate_acc_magnitude.")

    return df

def calc_over_in_below_mean(df, cols, perc = 0.01):
    result_dict = {}

    df_mean = df[cols].mean()
    df_lim_inf = df_mean - df_mean.abs() * perc
    df_lim_sup = df_mean + df_mean.abs() * perc

    for col in cols:
        result_dict[format_colname(col, "OverMean")] = (df[col] > df_lim_sup[col]).sum()
        result_dict[format_colname(col, "InMean")] = ((df[col] >= df_lim_inf[col]) & (df[col] <= df_lim_sup[col])).sum()
        result_dict[format_colname(col, "BelowMean") ] = (df[col] < df_lim_inf[col]).sum()

    return pd.DataFrame([result_dict])



def calculate_energy(series):
    fft_result = np.fft.fft(series.values)
    power_spectrum = np.abs(fft_result)**2 / len(series)
    return np.sum(power_spectrum)

def format_colname(col, stat):
    if "_" in col:
        first, rest = col.split("_", 1)
        if len(first) == 1:
            return f"Acc{first.upper()}{stat}_{rest}"
        elif first == "AccMagnitude":
            return f"{first}{stat}_{rest}"
        else:
            return f"{first}{stat}_{rest}"
    else:
        if len(col) == 1:
            return f"Acc{col.upper()}{stat}"
        elif col == "AccMagnitude":
            return f"{col}{stat}"
        else:
            return f"{col}{stat}"

def calculate_features(df, feature_cols, energy_cols):
    df_windowed = get_frames(df, FRAME_SIZE, HOP_SIZE)
    grouped = df_windowed.groupby(level=0)

    df_mean = grouped[feature_cols].mean()
    df_mean.columns = [format_colname(col, "Mean") for col in feature_cols]

    df_std = grouped[feature_cols].std()
    df_std.columns = [format_colname(col, "Std") for col in feature_cols]

    df_min = grouped[feature_cols].min()
    df_min.columns = [format_colname(col, "Min") for col in feature_cols]

    df_max = grouped[feature_cols].max()
    df_max.columns = [format_colname(col, "Max") for col in feature_cols]

    df_oib_mean = grouped.apply(lambda x: calc_over_in_below_mean(x, feature_cols))
    df_oib_mean = df_oib_mean.reset_index(level=1, drop=True)

    energy_features_list = []
    for col in energy_cols:
        energy_series = grouped[col].apply(calculate_energy)
        energy_series.name = format_colname(col, "Energy")
        energy_features_list.append(energy_series)
    df_energy = pd.concat(energy_features_list, axis=1)

    all_features = pd.concat([df_mean, df_std, df_min, df_max, df_oib_mean, df_energy], axis=1)

    return all_features.reset_index(drop=True)





os.makedirs(OUTPUT_PATH, exist_ok=True)

for utente in tqdm(UTENTI):
    single_sensor_feature_list = []
    all_sensors_feature_list = []
    for sensore in SENSORI:
        if sensore == 'w' or sensore == 't':
            feature_cols_single = ['x', 'y', 'z', 'AccMagnitude']
            energy_cols_single = ['AccMagnitude'] 
            for attivita in ATTIVITA:
                
                df = modified_load_sensor_data(BASE_DATA_PATH, utente, attivita, sensore)
                if df is not None and not df.empty:
                    df = calculate_acc_magnitude(df) 
                    if 'AccMagnitude' in df.columns:
                        df_features = calculate_features(df, feature_cols_single, energy_cols_single)
                        df_features['Userid'] = utente
                        df_features['Activity'] = attivita
                        df_features['position'] = "wrist" if sensore == 'w' else "thigh"
                        single_sensor_feature_list.append(df_features)

        elif sensore == 'wt':
            feature_cols_wt = ['x_wrist', 'y_wrist', 'z_wrist', 'x_thigh', 'y_thigh', 'z_thigh', 'AccMagnitude_wrist', 'AccMagnitude_thigh']
            energy_cols_wt = ['AccMagnitude_wrist', 'AccMagnitude_thigh']
            for attivita_wt in ATTIVITA_WT_FILENAMES:
                
                df = modified_load_wt_data(BASE_DATA_PATH, utente, attivita_wt)
                if df is not None and not df.empty:
                    df = calculate_acc_magnitude(df, prefix="wrist")
                    df = calculate_acc_magnitude(df, prefix="thigh")

                    if 'AccMagnitude_wrist' in df.columns and 'AccMagnitude_thigh' in df.columns:
                        df_features = calculate_features(df, feature_cols_wt, energy_cols_wt)
                        df_features['Userid'] = utente
                        df_features['Activity'] = ATTIVITA_MAP_WT[attivita_wt]
                        df_features['position'] = "all sensors"
                        all_sensors_feature_list.append(df_features)

    
    if single_sensor_feature_list:
        df_ss_features = pd.concat(single_sensor_feature_list, ignore_index=True)
        
        id_cols = ['Userid', 'Activity', 'position']
        extracted_feature_cols = [col for col in df_ss_features.columns if col not in id_cols]
        final_cols_ss = id_cols + extracted_feature_cols
        df_ss_features = df_ss_features[final_cols_ss]

        output_filename_single = f"{OUTPUT_PATH}/grouped_data_User{utente}.csv"
        df_ss_features.to_csv(output_filename_single, index=False)
    else:
        print(f" No features {utente}.")

    
    if all_sensors_feature_list:
        df_as_features = pd.concat(all_sensors_feature_list, ignore_index=True)
        
        id_cols = ['Userid', 'Activity', 'position'] 
        extracted_feature_cols = [col for col in df_as_features.columns if col not in id_cols]
        final_cols_as = id_cols + extracted_feature_cols
        df_as_features = df_as_features[final_cols_as]

        output_filename_all = f"{OUTPUT_PATH}/grouped_data_User{utente}_combined.csv"
        df_as_features.to_csv(output_filename_all, index=False)
    else:
        print(f" No features {utente}.")


