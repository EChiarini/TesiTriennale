import pandas as pd
import math
import numpy as np
import os
import re
import time
import sys
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from genericpath import isfile
from ntpath import join
from os import listdir
import glob
import cProfile, pstats 
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_features_for_each_sensor_optimized(df_data_input, positions_list):
    if not positions_list or df_data_input.empty:
        return pd.DataFrame()

    grouping_keys = ['Userid', 'label']
    
    meta_cols_to_exclude_from_rename = ['label', 'position', 'Userid','Activity']

    all_processed_group_dfs = []
    
    for (uid, lbl), group_df in df_data_input.groupby(grouping_keys, observed=False):
        processed_dfs_for_group_concat = []
        labels_columns_for_mask_group = []

        min_len_for_group_alignment = float('inf')
        temp_pos_dfs_for_group = {}

        
        actual_positions_in_group = [p for p in positions_list if p in group_df['position'].unique()]
        if len(actual_positions_in_group) != len(positions_list):
            
            continue 

        for position_val in actual_positions_in_group: 
            df_pos_filtered_group = group_df[group_df['position'] == position_val]
            temp_pos_dfs_for_group[position_val] = df_pos_filtered_group
            if len(df_pos_filtered_group) < min_len_for_group_alignment:
                min_len_for_group_alignment = len(df_pos_filtered_group)

        if min_len_for_group_alignment == 0 or min_len_for_group_alignment == float('inf'):
            
            continue

        for position_val in actual_positions_in_group:
            df_pos_segment_group = temp_pos_dfs_for_group[position_val].head(min_len_for_group_alignment).reset_index(drop=True)

            cols_to_rename_group = [col for col in df_pos_segment_group.columns if col not in meta_cols_to_exclude_from_rename]
            rename_dict_group = {col: f"{col}_{position_val}" for col in cols_to_rename_group}

            df_features_renamed_group = df_pos_segment_group.rename(columns=rename_dict_group)[list(rename_dict_group.values())]

            label_col_name_for_mask_group = f'label_{position_val}' 
            df_label_for_mask_group = df_pos_segment_group[['label']].rename(columns={'label': label_col_name_for_mask_group})
            labels_columns_for_mask_group.append(label_col_name_for_mask_group)

            current_df_part_group = pd.concat([df_features_renamed_group, df_label_for_mask_group], axis=1)
            processed_dfs_for_group_concat.append(current_df_part_group)

        if not processed_dfs_for_group_concat:
            continue

        df_group_final_wide = pd.concat(processed_dfs_for_group_concat, axis=1)

        
        existing_labels_cols_group = [col for col in labels_columns_for_mask_group if col in df_group_final_wide.columns]
        if not existing_labels_cols_group: continue 

        if len(existing_labels_cols_group) > 1:
            label_data_group = df_group_final_wide[existing_labels_cols_group]
            mask_group = label_data_group.nunique(axis=1, dropna=True) <= 1
            df_group_filtered_wide = df_group_final_wide[mask_group]
        else: 
            df_group_filtered_wide = df_group_final_wide

        if df_group_filtered_wide.empty: continue

        
        df_group_filtered_wide = df_group_filtered_wide.rename(columns={existing_labels_cols_group[0]: 'label'})
        if len(existing_labels_cols_group) > 1:
            cols_to_drop_labels_group = [col for col in existing_labels_cols_group[1:] if col in df_group_filtered_wide.columns]
            if cols_to_drop_labels_group:
                 df_group_filtered_wide = df_group_filtered_wide.drop(columns=cols_to_drop_labels_group)

        df_group_filtered_wide['Userid'] = uid
        all_processed_group_dfs.append(df_group_filtered_wide)

    if not all_processed_group_dfs:
        return pd.DataFrame()

    final_combined_df = pd.concat(all_processed_group_dfs, ignore_index=True)

    return final_combined_df.dropna()





def duplicaRighePesi(df_moved, weight):
    if df_moved.empty: return df_moved
    df_repeated = df_moved.loc[np.repeat(df_moved.index, weight)].copy()
    df_repeated.reset_index(drop=True, inplace=True)
    return df_repeated





def get_train_test_data(df_data_input, user=None, random_state=42, weight = -1, moltiplico_arg = False, 
                        features_list=None, all_positions_list=None, row_to_move = 0):

    df_train = df_data_input[df_data_input['Userid'] != user].reset_index(drop=True)
    df_test = df_data_input[df_data_input['Userid'] == user].reset_index(drop=True)

    if len(all_positions_list) > 1:
        df_train = get_features_for_each_sensor_optimized(df_train[features_list + ['position', 'label', 'Userid','Activity']], all_positions_list)
        df_test  = get_features_for_each_sensor_optimized(df_test[features_list + ['position', 'label', 'Userid','Activity']], all_positions_list)
    df_sampling_pool, df_testFISSO = train_test_split(df_test, test_size=0.2, random_state=random_state,stratify=df_test['label'])


    
    moved_indices = []
    if row_to_move > 0:
        for label_value in df_sampling_pool['label'].unique():
            df_test_label = df_sampling_pool[df_sampling_pool['label'] == label_value]
            indices_to_move = df_test_label.sample(n=row_to_move, random_state=random_state).index.tolist()
            moved_indices.extend(indices_to_move)

    total_rows_moved = len(moved_indices)

    if moved_indices:
        df_moved = df_sampling_pool.loc[moved_indices].copy()
        if moltiplico_arg and weight > 1:
            df_moved = duplicaRighePesi(df_moved, weight)
        df_train = pd.concat([df_train, df_moved], ignore_index=True).reset_index(drop=True)

    if len(all_positions_list) > 1:
        X_train = df_train.drop(columns=['label','Userid'])
        X_test = df_testFISSO.drop(columns=['label','Userid'])
    else:
        X_train = df_train[features_list]
        X_test = df_testFISSO[features_list]
    y_train = df_train['label']
    y_test  = df_testFISSO['label']


    return X_train, X_test, y_train, y_test, total_rows_moved





def train_model(X_train, X_test, y_train, random_state, dove_peso):
    class_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    final_sample_weights = dove_peso * class_weights

    xgb = XGBClassifier( n_estimators=150, random_state=random_state, n_jobs=-1)
    xgb.fit(X_train, y_train, sample_weight = final_sample_weights)
    y_pred = xgb.predict(X_test)

    return y_pred





def process_single_random_state(rand_state, current_position_list, df_data_arg, weight_list_arg,
                                lista_secondi_arg, ROW_TIME_arg, OVERLAP_arg, moltiplico_arg=False):
    single_state_results = []
    all_sensors_flag = len(current_position_list) > 1
    current_pos_key_str = 'all sensors' if all_sensors_flag else current_position_list[0]

    all_original_features = [item for item in df_data_arg.columns if
                             item not in ['Timestamp', 'Userid', 'UserAge', 'UserSex', 'UserHeight', 'UserWeight', 'Activity',
                                          'position', 'label', 'MagnxEnergy', 'MagnyEnergy', 'MagnzEnergy', 'MagnMagnitude',
                                          'MagnMagnitudeMean', 'MagnMagnitudeMin', 'MagnMagnitudeMax', 'MagnMagnitudeStd',
                                          'MagnMagnitudeEnergy']]
    all_original_features = [item for item in all_original_features if not re.match(r'.*MagnMagnitude.*', item)]
    selected_original_features = [item for item in all_original_features if re.match(r'.*Magnitude.*', item)]

    df_position = df_data_arg[df_data_arg['position'].isin(current_position_list)]
    labels = df_data_arg['Activity'].unique()

    if weight_list_arg is None:
        crowd_results = []
        for k_user in df_position['Userid'].unique():
            X_train, X_test, y_train, y_test, num_dati_spostati = \
                            get_train_test_data(
                                df_position,
                                user=k_user, random_state=rand_state,
                                features_list=selected_original_features,
                                all_positions_list=current_position_list
                            )

            sample_weight = [1] * len(X_train)
            y_pred = train_model(X_train, X_test, y_train, random_state=rand_state,dove_peso=sample_weight)

            if len(y_test) == 0 or len(y_pred) == 0:
                print("errore che non vuoi avere") 
                macro_f1 = np.nan; macro_precision = np.nan; macro_recall = np.nan
            else:
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                macro_f1 = class_report.get('macro avg', {}).get('f1-score', np.nan)
                macro_precision = class_report.get('macro avg', {}).get('precision', np.nan)
                macro_recall = class_report.get('macro avg', {}).get('recall', np.nan)

            current_metrics = {
                'k_user': k_user,
                'randomState': rand_state,
                'position': current_pos_key_str,
                'f1-score': macro_f1,
                'precision': macro_precision, 
                'recall': macro_recall
            }
            crowd_results.append(current_metrics)

        return crowd_results
    else:
        minimo_gruppo = df_position.groupby(['Userid', 'label', 'position']).size().min()
        minimo_disponibile = math.floor(minimo_gruppo * 0.8)
        for peso_arg in weight_list_arg:
            for secondi_nuovo_train_arg in lista_secondi_arg:
                target_n_total = 1 + (secondi_nuovo_train_arg - ROW_TIME_arg) / (ROW_TIME_arg * (1 - OVERLAP_arg))
                num_ideale_per_classe = max(1, target_n_total)
                final_num_to_move = int(min(num_ideale_per_classe, minimo_disponibile))
                tempoEffettivo = ROW_TIME_arg * (1 + (final_num_to_move - 1) * (1 - OVERLAP_arg))
                
                results_this_minute_all_k = []
                for k_user in df_position['Userid'].unique():
                    X_train, X_test, y_train, y_test, num_dati_spostati = \
                        get_train_test_data(
                            df_position,
                            user=k_user, random_state=rand_state,
                            weight=int(peso_arg), moltiplico_arg=moltiplico_arg,
                            features_list=selected_original_features,
                            all_positions_list=current_position_list,
                            row_to_move = final_num_to_move
                        )
                    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                        print("errore che non vuoi avere")
                        continue

                    if not moltiplico_arg and peso_arg > 1:
                        len_train = len(X_train) - num_dati_spostati
                        sample_weight = [1] * len_train + [peso_arg] * num_dati_spostati
                    else:
                        sample_weight = [1] * len(X_train)

                    start = time.perf_counter()
                    y_pred = train_model(X_train, X_test, y_train, random_state=rand_state,dove_peso=sample_weight)
                    end = time.perf_counter()
                    durata = end - start

                    if len(y_test) == 0 or len(y_pred) == 0:
                        print("errore che non vuoi avere") 
                        macro_f1 = np.nan; macro_precision = np.nan; macro_recall = np.nan
                    else:
                        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        macro_f1 = class_report.get('macro avg', {}).get('f1-score', np.nan)
                        macro_precision = class_report.get('macro avg', {}).get('precision', np.nan)
                        macro_recall = class_report.get('macro avg', {}).get('recall', np.nan)

                    current_metrics = {
                        'k_user': k_user,
                        'timeUsed': int(tempoEffettivo),
                        'weight': int(peso_arg),
                        'time': round(durata, 2),
                        'randomState': rand_state,
                        'position': current_pos_key_str,
                        'f1-score': macro_f1,
                        'precision': macro_precision, 
                        'recall': macro_recall
                    }

                    for label_idx, label in zip(y_train.unique(), labels):
                        current_metrics[f'f1_{label}'] =  class_report[str(label_idx)]['f1-score']
                        current_metrics[f'precision_{label}'] = class_report[str(label_idx)]['precision']
                        current_metrics[f'recall_{label}'] = class_report[str(label_idx)]['recall']

                    results_this_minute_all_k.append(current_metrics)

                single_state_results.extend(results_this_minute_all_k)

        return single_state_results





def k_fold_cross_validation_parallel(
    position_list_arg,
    df_data_arg,
    weight_list_param=None,
    random_state_list_global=None,
    ROW_TIME_global=None,
    OVERLAP_global=None,
    minute_list_global=None,
    moltiplico_global = False
    ):

    num_cores = os.cpu_count()

    parallel_outputs = Parallel(n_jobs=num_cores)(
        delayed(process_single_random_state)(
            rs,
            position_list_arg,
            df_data_arg,
            weight_list_param,
            minute_list_global,
            ROW_TIME_global,
            OVERLAP_global,
            moltiplico_global
        ) for rs in random_state_list_global
    )
    all_results_list = [item for sublist in parallel_outputs for item in sublist]

    return pd.DataFrame(all_results_list)





RANDOM_STATE_LIST = [int(i*10) for i in range(1,11)]
SECONDS_LIST = [4,10,20]
WEIGHT_LIST = [1,5,50,100,500,1000]
OVERLAP = 0.5
ROW_TIME = 4
SAVE = True
COMBINED_POSITIONS = False

DATASET = sys.argv[1]
print(f"ESECUZIONE DATASET {DATASET}")

if DATASET == 'selfBACK':
  NOMI_FILE = {
    'cartella_dati': 'selfBACK_processed_data'
  }
elif DATASET == 'SDALLE':
  NOMI_FILE = {
    'cartella_dati': 'SDALLE_processed_data'
  }

mypath_carica = os.getcwd() + '/data/' + NOMI_FILE['cartella_dati'] + '/'
file_pattern = os.path.join(mypath_carica, 'grouped_data*.csv')
all_files = glob.glob(file_pattern)

file_list = [f for f in all_files if '_combined' not in os.path.basename(f)]

df_data = pd.DataFrame()
for file in file_list:
    df_temp = pd.read_csv(file, header=0)
    if df_temp.columns[0].lower() in ['unnamed: 0', 'unnamed: 0.1']:
      df_temp = df_temp.iloc[:, 1:]
    df_data = pd.concat([df_data, df_temp], ignore_index=True)

def set_labels(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Activity'])
    return df, label_encoder.classes_
df_data, labels_activity_names = set_labels(df_data)
def strip_Spaces(df):
    df.columns = df.columns.str.strip()
    return df
df_data = strip_Spaces(df_data)

if len(all_files) - len(file_list) > 0:
  COMBINED_POSITIONS = True
  file_list = [f for f in all_files if '_combined' in os.path.basename(f)]
  df_data_combined = pd.DataFrame()
  for file in file_list:
      df_temp = pd.read_csv(file, header=0)
      if df_temp.columns[0].lower() in ['unnamed: 0', 'unnamed: 0.1']:
        df_temp = df_temp.iloc[:, 1:]
      df_data_combined = pd.concat([df_data_combined, df_temp], ignore_index=True)
  df_data_combined, labels_activity_names_combined = set_labels(df_data_combined)
  df_data_combined = strip_Spaces(df_data_combined)





print("baseline")

list_df_baseline_parts = []

all_available_positions_from_df = list(df_data['position'].unique())

if len(all_available_positions_from_df) > 1:

    if COMBINED_POSITIONS:
        position_list = df_data_combined['position'].unique().tolist()
        data = df_data_combined
    else:
        position_list = all_available_positions_from_df
        data = df_data

    df_all_sensors_model = k_fold_cross_validation_parallel(
        position_list, data,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=SECONDS_LIST
    )

    list_df_baseline_parts.append(df_all_sensors_model)

for pos_single in all_available_positions_from_df:
    df_pos_single_baseline = k_fold_cross_validation_parallel(
        [pos_single], df_data, 
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        minute_list_global=SECONDS_LIST
    )
    list_df_baseline_parts.append(df_pos_single_baseline)

pesoBaseData = pd.concat(list_df_baseline_parts, ignore_index=True)





print("base righe pesate")

list_df_model_base_parts = []

if len(all_available_positions_from_df) > 1:

    if COMBINED_POSITIONS:
        position_list = df_data_combined['position'].unique().tolist()
        data = df_data_combined
    else:
        position_list = all_available_positions_from_df
        data = df_data

    df_all_sensors_model = k_fold_cross_validation_parallel(
        position_list, data,
        weight_list_param=WEIGHT_LIST,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=SECONDS_LIST
    )

    list_df_model_base_parts.append(df_all_sensors_model)


for pos_single in all_available_positions_from_df:
    df_pos_single_model = k_fold_cross_validation_parallel(
        [pos_single], df_data,
        weight_list_param=WEIGHT_LIST,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=SECONDS_LIST
    )
    list_df_model_base_parts.append(df_pos_single_model)

baseData = pd.concat(list_df_model_base_parts, ignore_index=True)






print("base righe moltiplicate")

list_df_model_base_parts_MOLTIPLICO = []

if len(all_available_positions_from_df) > 1:

    if COMBINED_POSITIONS:
        position_list = df_data_combined['position'].unique().tolist()
        data = df_data_combined
    else:
        position_list = all_available_positions_from_df
        data = df_data

    df_all_sensors_model = k_fold_cross_validation_parallel(
        position_list, data,
        weight_list_param=WEIGHT_LIST,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=SECONDS_LIST, moltiplico_global = True
    )

    list_df_model_base_parts_MOLTIPLICO.append(df_all_sensors_model)


for pos_single in all_available_positions_from_df:
    df_pos_single_model = k_fold_cross_validation_parallel(
        [pos_single], df_data,
        weight_list_param=WEIGHT_LIST,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=SECONDS_LIST, moltiplico_global = True
    )
    list_df_model_base_parts_MOLTIPLICO.append(df_pos_single_model)

baseData_MOLTIPLICO = pd.concat(list_df_model_base_parts_MOLTIPLICO, ignore_index=True)






MYPATH = os.getcwd() + '/data/data_total/'





df_list = []

df_base = baseData
df_base['dataset'] = 'Peso'
df_list.append(df_base)

df_base_RFT = baseData_MOLTIPLICO
df_base_RFT['dataset'] = 'Moltiplicazione'
df_list.append(df_base_RFT)


df_all_data = pd.concat(df_list, ignore_index=True)
PESI = sorted(df_all_data['weight'].unique())

df_baseline = pesoBaseData


mypath_data_total = os.getcwd() + '/data/data_temp/'
os.makedirs(mypath_data_total, exist_ok=True)
df_all_data.to_csv(mypath_data_total + 'allData.csv', index=False)
df_baseline.to_csv(mypath_data_total + 'baseline.csv', index=False)



















































































