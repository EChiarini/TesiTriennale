import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from scipy.signal import butter, filtfilt

BASE_DATA_PATH = os.getcwd() + '/data/SDALLE_raw/'
OUTPUT_PATH = os.getcwd() + '/data/SDALLE_processed_data/'

SAMPLE_RATE = 37 
WINDOW_SECONDS = 4
HOP_SECONDS = 2
FRAME_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SIZE = int(SAMPLE_RATE * HOP_SECONDS)

def downsample_with_antialiasing(df, original_fs=148, target_fs=SAMPLE_RATE, cols=None):
    if cols is None:
        cols = df.columns
    factor = original_fs // target_fs
    nyquist_new = target_fs / 2
    cutoff = nyquist_new * 0.9
    b, a = butter(N=4, Wn=cutoff, fs=original_fs, btype='low')
    
    df_filtered = df.copy()
    for col in cols:
        if col in df.columns:
            df_filtered[col] = filtfilt(b, a, df[col].values)
    return df_filtered.iloc[::factor].reset_index(drop=True)

def load_trial_data_ds(file_path):
    column_names = [f"{modality}_{sensor}" for sensor in SENSORS for modality in MODALITIES]
    df = pd.read_csv(file_path, header=None, skiprows=8, on_bad_lines='skip')
    if df.shape[1] > len(column_names):
        df = df.iloc[:, :len(column_names)]
    df.columns = column_names
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    df = downsample_with_antialiasing(df, original_fs=148, target_fs=SAMPLE_RATE, cols=column_names)

    return df


UTENTI = range(1, 10)
ATTIVITA = ['Jogging', 'Stairs_up', 'Stairs_down', 'Walking']

SENSORS = [
    'Rectus_Femoris_left', 'Rectus_Femoris_right', 'Vastus_Medialis_left',
    'Vastus_Medialis_right', 'Vastus_Lateralis_Left', 'Vastus_Lateralis_right',
    'Semitendinosus_left', 'Semitendinosus_right'
]

MODALITIES = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']





def load_trial_data(file_path):
    column_names = [f"{modality}_{sensor}" for sensor in SENSORS for modality in MODALITIES]

    try:
        
        df = pd.read_csv(file_path, header=None, skiprows=8, on_bad_lines='skip')

        
        if df.shape[1] > len(column_names):
            df = df.iloc[:, :len(column_names)]

        
        df.columns = column_names
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        return df
    except Exception as e:
        print(f"Errore caricando o processando {file_path}: {e}")
        return pd.DataFrame()





def get_frames(data, frame_size, hop_size):
    if len(data) < frame_size: return pd.DataFrame()
    r = np.arange(len(data)); s = r[::hop_size]
    window_dentro = s[s + frame_size <= len(data)]
    z = list(zip(window_dentro, window_dentro + frame_size))
    g = lambda indices: data.iloc[indices[0]:indices[1]]
    if not z: return pd.DataFrame()
    return pd.concat(map(g, z), keys=range(len(z)))

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
    if col[0] == 'g' and len(col) < 3:
        col = col[1:]
        if "_" in col:
            first, rest = col.split("_", 1)
            if len(first) == 1:
                return f"Gyro{first.upper()}{stat}_{rest}"
            elif first == "GyroMagnitude":
                return f"{first}{stat}_{rest}"
            else:
                return f"{first}{stat}_{rest}"
        else:
            if len(col) == 1:
                return f"Gyro{col.upper()}{stat}"
            elif col == "GyroMagnitude":
                return f"{col}{stat}"
            else:
                return f"{col}{stat}"
    else:
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

    all_stat_features = []
    all_stat_features.append(grouped[feature_cols].mean().add_suffix('Mean'))
    all_stat_features.append(grouped[feature_cols].std().add_suffix('Std'))
    all_stat_features.append(grouped[feature_cols].min().add_suffix('Min'))
    all_stat_features.append(grouped[feature_cols].max().add_suffix('Max'))

    df_mean = grouped[feature_cols].mean()
    df_mean.columns = [format_colname(col, "Mean") for col in feature_cols]

    df_std = grouped[feature_cols].std()
    df_std.columns = [format_colname(col, "Std") for col in feature_cols]

    df_min = grouped[feature_cols].min()
    df_min.columns = [format_colname(col, "Min") for col in feature_cols]

    df_max = grouped[feature_cols].max()
    df_max.columns = [format_colname(col, "Max") for col in feature_cols]

    all_features = pd.concat(all_stat_features, axis=1)

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

for utente_id in tqdm(UTENTI, desc="Processing Users"):
    user_single_sensor_features = []
    user_combined_sensor_features = []

    for attivita in ATTIVITA:
        activity_path = os.path.join(BASE_DATA_PATH, f"Subject_{utente_id}", attivita)
        trial_files = glob.glob(os.path.join(activity_path, "Trial_*.csv"))

        for trial_file in trial_files:
            
            df_trial = load_trial_data_ds(trial_file)
            
            if df_trial.empty:
                continue

            
            acc_mag_cols_combined = []
            for sensor_name in SENSORS:
                x_col, y_col, z_col = f"AccX_{sensor_name}", f"AccY_{sensor_name}", f"AccZ_{sensor_name}"
                mag_col = f"AccMagnitude_{sensor_name}"
                df_trial[mag_col] = np.sqrt(np.square(df_trial[[x_col, y_col, z_col]]).sum(axis=1))
                acc_mag_cols_combined.append(mag_col)

            gyro_mag_cols_combined = []
            for sensor_name in SENSORS:
                gx_col, gy_col, gz_col = f"GyroX_{sensor_name}", f"GyroY_{sensor_name}", f"GyroZ_{sensor_name}"
                gyro_mag_col = f"GyroMagnitude_{sensor_name}"
                df_trial[gyro_mag_col] = np.sqrt(np.square(df_trial[[gx_col, gy_col, gz_col]]).sum(axis=1))
                gyro_mag_cols_combined.append(gyro_mag_col)

            
            acc_cols_all_sensors = [f"Acc{ax}_{s}" for s in SENSORS for ax in ['X', 'Y', 'Z']]
            
            gyro_cols_all_sensors = [f"Gyro{ax}_{s}" for s in SENSORS for ax in ['X', 'Y', 'Z']]

            
            feature_cols_combined = acc_cols_all_sensors + gyro_cols_all_sensors + acc_mag_cols_combined + gyro_mag_cols_combined
            energy_cols_combined = acc_mag_cols_combined + gyro_mag_cols_combined

            
            df_features_combined = calculate_features(df_trial, feature_cols_combined, energy_cols=energy_cols_combined)

            if not df_features_combined.empty:
                df_features_combined['Userid'] = utente_id
                df_features_combined['Activity'] = attivita
                df_features_combined['position'] = 'all position' 
                user_combined_sensor_features.append(df_features_combined)

            
            for sensor_name in SENSORS:
                
                single_sensor_cols = {
                    f"AccX_{sensor_name}": 'x',
                    f"AccY_{sensor_name}": 'y',
                    f"AccZ_{sensor_name}": 'z',
                    f"GyroX_{sensor_name}": 'gx',
                    f"GyroY_{sensor_name}": 'gy',
                    f"GyroZ_{sensor_name}": 'gz'
                }

                df_single_sensor = df_trial[list(single_sensor_cols.keys())].copy()
                df_single_sensor.rename(columns=single_sensor_cols, inplace=True)

                df_single_sensor['AccMagnitude'] = np.sqrt(np.square(df_single_sensor[['x', 'y', 'z']]).sum(axis=1))
                df_single_sensor['GyroMagnitude'] = np.sqrt(np.square(df_single_sensor[['gx', 'gy', 'gz']]).sum(axis=1))
                df_features_single = calculate_features(df_single_sensor, ['x', 'y', 'z', 'gx', 'gy', 'gz', 'AccMagnitude', 'GyroMagnitude'],['AccMagnitude', 'GyroMagnitude'])

                if not df_features_single.empty:
                    df_features_single['Userid'] = utente_id
                    df_features_single['Activity'] = attivita
                    df_features_single['position'] = sensor_name 
                    user_single_sensor_features.append(df_features_single)

    
    
    if user_single_sensor_features:
        df_final_single = pd.concat(user_single_sensor_features, ignore_index=True)
        output_filename_single = f"{OUTPUT_PATH}/grouped_data_User{utente_id}.csv"
        df_final_single.to_csv(output_filename_single, index=False)

    
    if user_combined_sensor_features:
        df_final_combined = pd.concat(user_combined_sensor_features, ignore_index=True)
        output_filename_combined = f"{OUTPUT_PATH}/grouped_data_User{utente_id}_combined.csv"
        df_final_combined.to_csv(output_filename_combined, index=False)

