#!/usr/bin/env python
# coding: utf-8

# ### Dipendenze

# In[1]:


from xgboost import XGBClassifier
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from tqdm import tqdm


# In[2]:


random_state_list = [int(i*11) for i in range(1,16)]
#random_state_list = [123,42,456]
ROW_TIME = 4 #secondi di dati riassunti in una riga del dataframe


mypath = os.getcwd() + '/data/data_total/'
os.makedirs(mypath[:-1], exist_ok=True)


# ### Carica Dati per Modello

# In[56]:


mypath_carica = os.getcwd() + '/data/Processed_data/'
file_pattern = 'grouped_data.*'

file_list = [
    f for f in listdir(mypath_carica)
    if (isfile(join(mypath_carica, f)) and
               re.compile(file_pattern).match(f))]
df_data = pd.DataFrame()
for file in file_list:
    df = pd.read_csv(mypath_carica + file, header=0).iloc[:,1:]
    df_data = pd.concat([df_data, df]).reset_index(drop=True)
def set_labels(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Activity'])
    return df, label_encoder.classes_
df_data, labels = set_labels(df_data)


# ### Bilanciamento Dati
# Tutti gli utenti avranno lo stesso numero di dati di quello che ne ha meno

# In[57]:


def balance_user_labels(df, random_state=42):
    min_count = df.groupby(['Userid', 'label', 'position']).size().min()

    def sample_group(group):
        return group.sample(n=min_count, random_state=random_state)

    balanced_df = df.groupby(['Userid', 'label', 'position']).apply(sample_group).reset_index(drop=True)

    return balanced_df

df_data = balance_user_labels(df_data)


# ### Per quanto uso multipli sensori

# In[58]:


def get_features_for_each_sensor(df_data, positions):
    df_final = pd.DataFrame()  # This will be the final DataFrame including features and labels
    labels_columns = []  # To keep track of the names of the 'label' columns for each position

    df_data = df_data.reset_index(drop=True)

    for position in positions:
        # Prepare feature columns for the current position
        df_data_pos = df_data[df_data['position'] == position].drop(columns=['label', 'position']).rename(
                columns=lambda x: x + '_' + position).reset_index(drop=True)

        # Prepare label column for the current position
        label_col_name = f'label_{position}'
        df_labels = df_data[df_data['position'] == position]['label'].reset_index(drop=True).to_frame(name=label_col_name)
        labels_columns.append(label_col_name)

        # Concatenate feature and label columns
        df_combined = pd.concat([df_data_pos, df_labels], axis=1)
        df_final = pd.concat([df_final, df_combined], axis=1)
    # Filter rows where all label columns have the same value
    mask = df_final.apply(lambda row: all(row[col] == row[labels_columns[0]] for col in labels_columns), axis=1)
    #first_label_col = labels_columns[0]
    #mask = df_final.apply(lambda row: all(row.to_dict()[col] == row.to_dict()[first_label_col] for col in labels_columns), axis=1)
    df_filtered = df_final[mask]
    # Optionally, you might want to drop redundant label columns and keep just one
    df_filtered = df_filtered.drop(columns=labels_columns[1:]).rename(columns={labels_columns[0]: 'label'})

    return df_filtered.dropna()


# In[59]:


def duplicaRighePesi(df_moved, weight, varianza):
    if not varianza:
        df_moved = df_moved.loc[np.repeat(df_moved.index, int(weight))].reset_index(drop=True)
    elif varianza and weight > 1:
        df_moved['is_original'] = True
        repeated_part = df_moved.loc[np.repeat(df_moved.index, int(weight) - 1)].copy()
        repeated_part['is_original'] = False
        df_moved = pd.concat([df_moved, repeated_part], ignore_index=True)

        feature_cols = df_moved.columns.difference(['label', 'is_original'])
        feature_cols = df_moved[feature_cols].select_dtypes(include=[np.number]).columns
        df_moved[feature_cols] = df_moved[feature_cols].astype(float)

        df_moved.loc[~df_moved['is_original'], feature_cols] *= np.random.uniform(0.99, 1.01, size=df_moved.loc[~df_moved['is_original'], feature_cols].shape)

        df_moved = df_moved.drop('is_original', axis=1)

    return df_moved


# In[60]:


def get_train_test_data(df_data, user=None, random_state=42, percentage=None, weight = None, varianza = False):
    positions = list(df_data['position'].unique())
    all_features = [item for item in df_data.columns if
                    item not in ['Timestamp', 'Userid', 'UserAge', 'UserSex', 'UserHeight', 'UserWeight', 'Activity',
                                 'position', 'label', 'MagnxEnergy', 'MagnyEnergy', 'MagnzEnergy', 'MagnMagnitude',
                                 'MagnMagnitudeMean', 'MagnMagnitudeMin', 'MagnMagnitudeMax', 'MagnMagnitudeStd',
                                 'MagnMagnitudeEnergy']]
    all_features = [item for item in all_features if not re.match(r'.*MagnMagnitude.*', item)]
    magnitude_features = [item for item in all_features if re.match(r'.*Magnitude.*', item)]
    features = magnitude_features


    df_data = df_data[df_data['position'].isin(positions)]
    for position in positions:
        if position not in list(df_data['position'].unique()):
            print(f'Position {position} not found in the dataset')
            return None

    df_train = df_data[df_data['Userid'] != user].reset_index(drop=True)
    df_test = df_data[df_data['Userid'] == user].reset_index(drop=True)
    df_test, df_testFISSO = train_test_split(df_test, test_size=0.2, random_state=random_state,stratify=df_test['label'])  # 80/20 split

    #sposto le righe
    moved_indices = []
    for label_value in df_test['label'].unique():
        df_test_label = df_test[df_test['label'] == label_value]
        for position_value in df_test_label['position'].unique():
            df_test_label_position = df_test_label[df_test_label['position'] == position_value]
            num_to_move = int(len(df_test_label_position) * percentage)
            if num_to_move > 0:
                indices_to_move = df_test_label_position.sample(n=num_to_move, random_state=random_state).index.tolist()
                moved_indices.extend(indices_to_move)

    righe_mosse = len(moved_indices)

    if len(positions) > 1:
        righe_mosse = righe_mosse / len(positions)
        df_train = get_features_for_each_sensor(df_train[features + ['position', 'label']], positions)
        df_testFISSO  = get_features_for_each_sensor(df_testFISSO[features + ['position', 'label']], positions)

    if moved_indices:
        df_moved = df_test.loc[moved_indices].copy()
        if len(positions) > 1:
            df_moved = get_features_for_each_sensor(df_moved[features + ['position', 'label']], positions)
        df_moved = duplicaRighePesi(df_moved, weight, varianza)
        df_train = pd.concat([df_train, df_moved], ignore_index=True).reset_index(drop=True)

    if len(positions) > 1:
        X_train = df_train.drop(columns=['label'])
        X_test = df_testFISSO.drop(columns=['label'])
    else:
        X_train = df_train[features]
        X_test = df_testFISSO[features]
    y_train = df_train['label']
    y_test  = df_testFISSO['label']

    #print("ed io gli passo:",moved_indices)
    return X_train, X_test, y_train, y_test, righe_mosse


# ### Alleno Modello

# In[61]:


def train_model(X_train, X_test, y_train, y_test, random_state):
    xgb = XGBClassifier(
        n_estimators=150,
        random_state=random_state,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    #accuracy = xgb.score(X_test, y_test)
    #return accuracy, y_pred
    return y_pred


# ### Divido i dati

# In[62]:


def k_fold_cross_validation(position, df_data, weight_list=None, varianza = False, lista_percentuali = None):
    global df_f1_score
    all_sensors = len(position) > 1
    labels = df_data['Activity'].unique()

    if weight_list is None:
        weight_list = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000]
    if lista_percentuali is None:
        lista_percentuali = [i / 100 for i in range(0, 101, 5)]

    for rand_state in tqdm(random_state_list):
        #print(" random state:", rand_state)
        for peso in tqdm(weight_list):
            for percentuale_nuovo_train in tqdm(lista_percentuali):
                #print(",".join(position)+" stato "+str(rand_state)+" "+str(peso)+"w "+str(int(percentuale_nuovo_train*100))+"%")
                for k in df_data['Userid'].unique():
                    X_train, X_test, y_train, y_test, num_dati_spostati = get_train_test_data(df_data[df_data['position'].isin(position)], user=k, random_state = rand_state, percentage=float(percentuale_nuovo_train), weight = float(peso), varianza = varianza)

                    start = time.perf_counter()
                    y_pred = train_model(X_train, X_test, y_train, y_test,random_state = rand_state)
                    end = time.perf_counter()
                    durata = end - start

                    class_report = classification_report(y_test, y_pred, output_dict=True,zero_division=0)
                    for label_idx, label in zip(y_train.unique(), labels):
                        df = pd.DataFrame()
                        df['label'] = [label]
                        df['timeUsed'] = [num_dati_spostati * ROW_TIME]
                        df['percentage'] = [int(percentuale_nuovo_train*100)]
                        df['weight'] = [float(peso)]
                        df['time'] = [round(durata, 2)]
                        df['randomState'] = [rand_state]
                        df['position'] = ['both sensors'] if all_sensors else position

                        key_formats = [str(label_idx), str(float(label_idx)), str(int(label_idx))]
                        for key in key_formats:
                            try:
                                df['f1-score'] = [class_report[key]['f1-score']]
                                df['precision'] = [class_report[key]['precision']]
                                df['recall'] = [class_report[key]['recall']]
                                break
                            except KeyError:
                                continue

                        df_f1_score = pd.concat([df_f1_score, df], axis=0).reset_index(drop=True)

                if baseCalcolata:
                    if all_sensors or position[0] in ['left wrist', 'right pocket']:
                        df_appena_calcolato = df_f1_score[df_f1_score['weight'] == peso]
                        df_appena_calcolato = df_appena_calcolato[df_appena_calcolato['percentage'] == int(percentuale_nuovo_train*100)]

                        if all_sensors:
                            pos_key = 'both sensors'
                            f1_s_max = f1_s_max_both
                        else:
                            pos_key = position[0]
                            f1_s_max = f1_s_max_lw if pos_key == 'left wrist' else f1_s_max_rp

                        f1_s_mifermo = prendiMax(df_appena_calcolato, pos_key, rand_state)
                        #print(f"confronto {f1_s_mifermo[rand_state]} e {f1_s_max[rand_state]}")

                        if f1_s_mifermo[rand_state] >= f1_s_max[rand_state]:
                            print(f"  stop a {int(percentuale_nuovo_train * 100)}%({num_dati_spostati * ROW_TIME}) per peso {peso} "
                                  f"in quanto {f1_s_mifermo[rand_state]} è maggiore del max a peso 1 ({f1_s_max[rand_state]})")
                            break


# In[63]:


def prendiMax(df, position, random_states):
    if not isinstance(random_states, list):
        random_states = [random_states]

    df_pos = df[(df['position'] == position) & (df['randomState'].isin(random_states))]
    grouped = df_pos.groupby(['randomState', 'timeUsed'])['f1-score'].mean()
    max_medie = grouped.groupby('randomState').max()

    return max_medie.to_dict()


# ### Caso Base
# Peso 1, fa da ottimizzatore per i veri modelli con tutti i vari pesi facendoli fermare quando superano il massimo di questo

# In[64]:


baseCalcolata = False

df_f1_score = pd.DataFrame()

# CREO CASO BASE
print("both sensors")
k_fold_cross_validation(['right pocket','left wrist'], df_data, weight_list=[1])
print("right pocket")
k_fold_cross_validation(['right pocket'], df_data, weight_list=[1])
print("left wrist")
k_fold_cross_validation(['left wrist'], df_data, weight_list=[1])
df_base = df_f1_score[df_f1_score['weight'] == 1] #opzionale in teoria

f1_s_max_both = prendiMax(df_base, 'both sensors',random_state_list)
f1_s_max_lw = prendiMax(df_base, 'left wrist',random_state_list)
f1_s_max_rp = prendiMax(df_base, 'right pocket',random_state_list)

print("f1-score both sensors:",f1_s_max_both)
print("f1-score left wrist:",f1_s_max_lw)
print("f1-score right pocket:",f1_s_max_rp)

pesoBaseData = df_f1_score.copy()

baseCalcolata = True


# In[65]:


pesoBaseData.to_csv(mypath + 'modelXGBtotal_baseline.csv')


# In[66]:


df_f1_score = pd.DataFrame()

print("both sensors")
k_fold_cross_validation(['right pocket','left wrist'], df_data)
print("right pocket")
k_fold_cross_validation(['right pocket'], df_data)
print("left wrist")
k_fold_cross_validation(['left wrist'], df_data)

baseData = df_f1_score.copy()
baseData = pd.concat([baseData, pesoBaseData]).reset_index(drop=True)


# ### Salvo modello base

# In[67]:


baseData.to_csv(mypath + 'modelXGBtotal_base.csv')


# ### Calcolo modello con varianza
# Varianza definita come il moltiplicare ogni riga ripetuta per un valore compreso tra 0.99 e 1.01

# In[68]:


df_f1_score = pd.DataFrame()

print("both sensors")
k_fold_cross_validation(['right pocket','left wrist'], df_data, varianza = True)
print("right pocket")
k_fold_cross_validation(['right pocket'], df_data, varianza = True)
print("left wrist")
k_fold_cross_validation(['left wrist'], df_data, varianza = True)

varianzaData = df_f1_score.copy()
varianzaData = pd.concat([varianzaData, pesoBaseData]).reset_index(drop=True)


# ### Salvo modello con varianza

# In[69]:


varianzaData.to_csv(mypath + 'modelXGBtotal_varianza.csv')

