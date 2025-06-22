#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import cProfile, pstats # per vedere quanti ci metto
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# # CREAZIONE MODELLI

# In[ ]:


def get_features_for_each_sensor_optimized(df_data_input, positions_list):
    if not positions_list or df_data_input.empty:
        return pd.DataFrame()

    grouping_keys = ['Userid', 'label']
    # Colonne da escludere dalla ridenominazione delle feature
    meta_cols_to_exclude_from_rename = ['label', 'position', 'Userid','Activity']

    all_processed_group_dfs = []
    # Itera su ogni gruppo (Userid, label)
    for (uid, lbl), group_df in df_data_input.groupby(grouping_keys, observed=False):
        processed_dfs_for_group_concat = []
        labels_columns_for_mask_group = []

        min_len_for_group_alignment = float('inf')
        temp_pos_dfs_for_group = {}

        #trova la lunghezza minima tra le posizioni
        actual_positions_in_group = [p for p in positions_list if p in group_df['position'].unique()]
        if len(actual_positions_in_group) != len(positions_list):
            #non vuoi entrare qui
            continue 

        for position_val in actual_positions_in_group: # Usa actual_positions_in_group
            df_pos_filtered_group = group_df[group_df['position'] == position_val]
            temp_pos_dfs_for_group[position_val] = df_pos_filtered_group
            if len(df_pos_filtered_group) < min_len_for_group_alignment:
                min_len_for_group_alignment = len(df_pos_filtered_group)

        if min_len_for_group_alignment == 0 or min_len_for_group_alignment == float('inf'):
            #non vuoi entrare qui
            continue

        for position_val in actual_positions_in_group:
            df_pos_segment_group = temp_pos_dfs_for_group[position_val].head(min_len_for_group_alignment).reset_index(drop=True)

            cols_to_rename_group = [col for col in df_pos_segment_group.columns if col not in meta_cols_to_exclude_from_rename]
            rename_dict_group = {col: f"{col}_{position_val}" for col in cols_to_rename_group}

            df_features_renamed_group = df_pos_segment_group.rename(columns=rename_dict_group)[list(rename_dict_group.values())]

            label_col_name_for_mask_group = f'label_{position_val}' # Usato solo per il check mask
            df_label_for_mask_group = df_pos_segment_group[['label']].rename(columns={'label': label_col_name_for_mask_group})
            labels_columns_for_mask_group.append(label_col_name_for_mask_group)

            current_df_part_group = pd.concat([df_features_renamed_group, df_label_for_mask_group], axis=1)
            processed_dfs_for_group_concat.append(current_df_part_group)

        if not processed_dfs_for_group_concat:
            continue

        df_group_final_wide = pd.concat(processed_dfs_for_group_concat, axis=1)

        # Applica la maschera per le label
        existing_labels_cols_group = [col for col in labels_columns_for_mask_group if col in df_group_final_wide.columns]
        if not existing_labels_cols_group: continue # Non dovrebbe succedere

        if len(existing_labels_cols_group) > 1:
            label_data_group = df_group_final_wide[existing_labels_cols_group]
            mask_group = label_data_group.nunique(axis=1, dropna=True) <= 1
            df_group_filtered_wide = df_group_final_wide[mask_group]
        else: # Solo una colonna label, nessun filtro mask necessario
            df_group_filtered_wide = df_group_final_wide

        if df_group_filtered_wide.empty: continue

        # Rinomina la prima colonna label in 'label' e droppa le altre (ridondanti)
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


# In[ ]:


def duplicaRighePesi(df_moved, weight): #solo per varianza
    if df_moved.empty: return df_moved
    df_moved['is_original'] = True
    repeated_part = df_moved.loc[np.repeat(df_moved.index, int(weight) - 1)].copy()
    repeated_part['is_original'] = False
    df_moved = pd.concat([df_moved, repeated_part], ignore_index=True)
    feature_cols = df_moved.columns.difference(['label', 'is_original', 'Userid', 'Timestamp', 'Activity', 'position'])
    feature_cols = df_moved[feature_cols].select_dtypes(include=[np.number]).columns
    if not feature_cols.empty:
        df_moved[feature_cols] = df_moved[feature_cols].astype(float)
        df_moved.loc[~df_moved['is_original'], feature_cols] *= np.random.uniform(0.99, 1.01, size=df_moved.loc[~df_moved['is_original'], feature_cols].shape)
    df_moved = df_moved.drop('is_original', axis=1, errors='ignore')
    return df_moved


# In[ ]:


def get_train_test_data(df_data_input, user=None, random_state=42, weight = -1, varianza = False, 
                        features_list=None, all_positions_list=None, row_to_move = 0):

    df_train = df_data_input[df_data_input['Userid'] != user].reset_index(drop=True)
    df_test = df_data_input[df_data_input['Userid'] == user].reset_index(drop=True)

    if len(all_positions_list) > 1:
        df_train = get_features_for_each_sensor_optimized(df_train[features_list + ['position', 'label', 'Userid','Activity']], all_positions_list)
        df_test  = get_features_for_each_sensor_optimized(df_test[features_list + ['position', 'label', 'Userid','Activity']], all_positions_list)
    df_sampling_pool, df_testFISSO = train_test_split(df_test, test_size=0.2, random_state=random_state,stratify=df_test['label'])

    #sposto le righe
    num_to_move = 0
    moved_indices = []
    if row_to_move > 0:
        for label_value in df_sampling_pool['label'].unique():
            df_test_label = df_sampling_pool[df_sampling_pool['label'] == label_value]
            indices_to_move = df_test_label.sample(n=row_to_move, random_state=random_state).index.tolist()
            moved_indices.extend(indices_to_move)

    total_rows_moved = len(moved_indices)

    if moved_indices:
        df_moved = df_sampling_pool.loc[moved_indices].copy()
        if varianza and weight > 1:
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


# In[ ]:


def train_model(X_train, X_test, y_train, random_state, dove_peso, RFT):
    class_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    final_sample_weights = dove_peso * class_weights

    if RFT:
        xgb = RandomForestClassifier( n_estimators=150, random_state=random_state, n_jobs=-1)
    else:
        xgb = XGBClassifier( n_estimators=150, random_state=random_state, n_jobs=-1)
    xgb.fit(X_train, y_train, sample_weight = final_sample_weights)
    y_pred = xgb.predict(X_test)

    return y_pred


# In[ ]:


def process_single_random_state(rand_state, current_position_list, df_data_arg, weight_list_arg,
                                lista_minuti_arg, varianza_arg, ROW_TIME_arg, OVERLAP_arg, RFT):
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
            y_pred = train_model(X_train, X_test, y_train, random_state=rand_state,dove_peso=sample_weight, RFT=RFT)

            if len(y_test) == 0 or len(y_pred) == 0:
                print("errore che non vuoi avere") #crasha tutto
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
        for peso_arg in weight_list_arg:
            for minuti_nuovo_train_arg in lista_minuti_arg:
                minimo_gruppo = df_position.groupby(['Userid', 'label', 'position']).size().min()
                minimo_disponibile = math.floor(minimo_gruppo * 0.8)
                target_n_total = 1 + (minuti_nuovo_train_arg * 60 - ROW_TIME_arg) / (ROW_TIME_arg * (1 - OVERLAP_arg))
                target_n_total = math.ceil(target_n_total)
                num_labels = df_position['label'].nunique()
                num_ideale_per_classe = math.ceil(target_n_total / num_labels)
                final_num_to_move = min(num_ideale_per_classe, minimo_disponibile)

                results_this_minute_all_k = []
                for k_user in df_position['Userid'].unique():
                    X_train, X_test, y_train, y_test, num_dati_spostati = \
                        get_train_test_data(
                            df_position,
                            user=k_user, random_state=rand_state,
                            weight=int(peso_arg), varianza=varianza_arg,
                            features_list=selected_original_features,
                            all_positions_list=current_position_list,
                            row_to_move = final_num_to_move
                        )
                    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                        print("errore che non vuoi avere")
                        continue

                    if not varianza_arg and peso_arg > 1:
                        len_train = len(X_train) - num_dati_spostati
                        sample_weight = [1] * len_train + [peso_arg] * num_dati_spostati
                    else:
                        sample_weight = [1] * len(X_train)

                    start = time.perf_counter()
                    y_pred = train_model(X_train, X_test, y_train, random_state=rand_state,dove_peso=sample_weight, RFT=RFT)
                    end = time.perf_counter()
                    durata = end - start

                    tempoEffettivo = ROW_TIME_arg * (1 + (num_dati_spostati - 1) * (1 - OVERLAP_arg))
                    tempoEffettivo = tempoEffettivo - (tempoEffettivo % 10) if tempoEffettivo % 10 <= 15 else tempoEffettivo


                    if len(y_test) == 0 or len(y_pred) == 0:
                        print("errore che non vuoi avere") #crasha tutto
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


# In[ ]:


def k_fold_cross_validation_parallel(
    position_list_arg,
    df_data_arg,
    weight_list_param=None,
    varianza_param=False,
    random_state_list_global=None,
    ROW_TIME_global=None,
    OVERLAP_global=None,
    minute_list_global=None,
    RFT = False
    ):

    num_cores = os.cpu_count()

    parallel_outputs = Parallel(n_jobs=num_cores)(
        delayed(process_single_random_state)(
            rs,
            position_list_arg,
            df_data_arg,
            weight_list_param,
            minute_list_global,
            varianza_param,
            ROW_TIME_global,
            OVERLAP_global,
            RFT = RFT
        ) for rs in random_state_list_global
    )
    all_results_list = [item for sublist in parallel_outputs for item in sublist]

    return pd.DataFrame(all_results_list)


# In[ ]:


RANDOM_STATE_LIST = [1]
MINUTE_LIST = [1,5]
WEIGHT_LIST = [1,5]
OVERLAP = 0.5
ROW_TIME = 4
SAVE = True
COMBINED_POSITIONS = False

DATASET = sys.argv[1]
print(f"ESECUZIONE DATASET {DATASET}")

if DATASET == 'MultiPositionWearable':
  NOMI_FILE = {
    'cartella_dati': 'MultiPositionWearable_processed_data'
  }
elif DATASET == 'selfBACK':
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


# ## BASELINE XGB

# In[ ]:


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
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST
    )

    list_df_baseline_parts.append(df_all_sensors_model)

for pos_single in all_available_positions_from_df:
    df_pos_single_baseline = k_fold_cross_validation_parallel(
        [pos_single], df_data, 
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        minute_list_global=MINUTE_LIST
    )
    list_df_baseline_parts.append(df_pos_single_baseline)

pesoBaseData = pd.concat(list_df_baseline_parts, ignore_index=True)


# ## BASELINE RFT

# In[ ]:


list_df_baseline_parts_RFT = []

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
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST, RFT = True
    )

    list_df_baseline_parts_RFT.append(df_all_sensors_model)

for pos_single in all_available_positions_from_df:
    df_pos_single_baseline = k_fold_cross_validation_parallel(
        [pos_single], df_data, 
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        minute_list_global=MINUTE_LIST, RFT = True
    )
    list_df_baseline_parts_RFT.append(df_pos_single_baseline)

pesoBaseData_RFT = pd.concat(list_df_baseline_parts_RFT, ignore_index=True)


# ## BASE XGB

# In[ ]:


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
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST
    )

    list_df_model_base_parts.append(df_all_sensors_model)


for pos_single in all_available_positions_from_df:
    df_pos_single_model = k_fold_cross_validation_parallel(
        [pos_single], df_data,
        weight_list_param=WEIGHT_LIST,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST
    )
    list_df_model_base_parts.append(df_pos_single_model)

baseData = pd.concat(list_df_model_base_parts, ignore_index=True)


# ## BASE RFT

# In[ ]:


list_df_model_base_parts_RFT = []

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
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST, RFT = True
    )

    list_df_model_base_parts_RFT.append(df_all_sensors_model)


for pos_single in all_available_positions_from_df:
    df_pos_single_model = k_fold_cross_validation_parallel(
        [pos_single], df_data,
        weight_list_param=WEIGHT_LIST,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST, RFT = True
    )
    list_df_model_base_parts_RFT.append(df_pos_single_model)

baseData_RFT = pd.concat(list_df_model_base_parts_RFT, ignore_index=True)


# ## VARIANZA XGB

# In[ ]:


list_df_model_varianza_parts = []

if len(all_available_positions_from_df) > 1:
    if COMBINED_POSITIONS:
        position_list = df_data_combined['position'].unique().tolist()
        data = df_data_combined
    else:
        position_list = all_available_positions_from_df
        data = df_data

    df_all_sensors_model = k_fold_cross_validation_parallel(
        position_list, data,
        weight_list_param=WEIGHT_LIST, varianza_param=True,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST
    )
    list_df_model_varianza_parts.append(df_all_sensors_model)

for pos_single in all_available_positions_from_df:
    df_pos_single_varianza = k_fold_cross_validation_parallel(
        [pos_single], df_data,
        weight_list_param=WEIGHT_LIST, varianza_param=True,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST
    )
    list_df_model_varianza_parts.append(df_pos_single_varianza)

varianzaData = pd.concat(list_df_model_varianza_parts, ignore_index=True)


# ## VARIANZA RFT

# In[ ]:


list_df_model_varianza_parts_RFT = []
if len(all_available_positions_from_df) > 1:
    if COMBINED_POSITIONS:
        position_list = df_data_combined['position'].unique().tolist()
        data = df_data_combined
    else:
        position_list = all_available_positions_from_df
        data = df_data

    df_all_sensors_model = k_fold_cross_validation_parallel(
        position_list, data,
        weight_list_param=WEIGHT_LIST, varianza_param=True,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST, RFT = True
    )
    list_df_model_varianza_parts_RFT.append(df_all_sensors_model)

for pos_single in all_available_positions_from_df:
    df_pos_single_varianza = k_fold_cross_validation_parallel(
        [pos_single], df_data,
        weight_list_param=WEIGHT_LIST, varianza_param=True,
        random_state_list_global=RANDOM_STATE_LIST, ROW_TIME_global=ROW_TIME,
        OVERLAP_global=OVERLAP, minute_list_global=MINUTE_LIST, RFT = True
    )
    list_df_model_varianza_parts_RFT.append(df_pos_single_varianza)

varianzaData_RFT = pd.concat(list_df_model_varianza_parts_RFT, ignore_index=True)


# # GRAFICI

# In[ ]:


MYPATH = os.getcwd() + '/data/data_total/'


# In[ ]:


df_list = []

df_base = baseData
df_base['dataset'] = 'XGB_BASE'
df_list.append(df_base)

df_varianza = varianzaData
df_varianza['dataset'] = 'XGB_VARIANZA'
df_list.append(df_varianza)

df_base_RFT = baseData_RFT
df_base_RFT['dataset'] = 'RFT_BASE'
df_list.append(df_base_RFT)

df_varianza_RFT = varianzaData_RFT
df_varianza_RFT['dataset'] = 'RFT_VARIANZA'
df_list.append(df_varianza_RFT)

df_all_data = pd.concat(df_list, ignore_index=True)
PESI = sorted(df_all_data['weight'].unique())

df_baseline = pesoBaseData
df_baseline_RFT = pesoBaseData_RFT


# In[ ]:


def format_seconds_to_minutes(val,pos):
    if val < 60:
        return f"{val:.0f} sec"
    else:
        minutes = int(val // 60)
        seconds = int(val % 60)
        return f"{minutes}:{seconds:02d} min"


# In[ ]:


for position_to_plot in df_all_data['position'].unique():
    df_pos_main = df_all_data[df_all_data['position'] == position_to_plot]
    df_pos_main = df_pos_main.sort_values(by='weight')

    df_baseline_pos = df_baseline[df_baseline['position'] == position_to_plot]
    df_baseline_pos_RFT = df_baseline_RFT[df_baseline_RFT['position'] == position_to_plot]

    baseline_f1_value = df_baseline_pos['f1-score'].mean() if not df_baseline_pos.empty else np.nan
    baseline_f1_value_RFT = df_baseline_pos_RFT['f1-score'].mean() if not df_baseline_pos_RFT.empty else np.nan
    df_plot_data_avg = df_pos_main.groupby(['weight', 'dataset', 'timeUsed'])[['f1-score', 'time']].mean().reset_index()

    df_plot_data_avg['Tempo Usato'] = df_plot_data_avg['timeUsed'].apply(
        lambda x: format_seconds_to_minutes(x, None)
    )

    fig, ax1 = plt.subplots(figsize=(18, 9))

    unique_datasets = df_plot_data_avg['dataset'].unique()
    dataset_colors = dict(zip(unique_datasets, sns.color_palette("tab10", n_colors=len(unique_datasets))))

    sns.lineplot(
        data=df_plot_data_avg,
        x='weight',
        y='f1-score',
        hue='dataset',
        style='Tempo Usato',
        palette=dataset_colors,
        marker='o',
        markersize=7,
        linewidth=2.0,
        ax=ax1
    )

    if np.isfinite(baseline_f1_value):
        ax1.axhline(y=baseline_f1_value, color='black', linestyle=':', linewidth=2, 
                    label=f'Baseline F1 ({baseline_f1_value:.3f})')
        ax1.axhline(y=baseline_f1_value_RFT, color='black', linestyle=':', linewidth=2, 
                    label=f'Baseline F1 ({baseline_f1_value_RFT:.3f})')

    ax1.set_ylabel('f1-score', color='tab:blue', fontsize=14)
    baseline_min = min(baseline_f1_value,baseline_f1_value_RFT)
    ax1.set_ylim(baseline_min-0.01, 1.01)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    ax1.set_xlabel("Peso (Weight)", fontsize=14)
    ax1.set_xscale('log')
    ax1.set_xticks(PESI)
    ax1.set_xticklabels(PESI)
    ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, alpha=0.4)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title='Modello / TimeUsed', loc='best')

    plt.title(f'Posizione: {position_to_plot}', fontsize=18, pad=20)

    fig.tight_layout()

    output_dir = os.path.join(os.getcwd(), 'images','differenza tra modelli', DATASET)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'plot_{position_to_plot}.png')
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

