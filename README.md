# Tesi/Tirocinio Triennale Chiarini Emiliano
### SetUpData.ipynb - MultiPosition Wearable dataset
Link: https://zenodo.org/records/11219118  
Caratteristiche:
- 10 Utenti
- 8 Posizioni
- 3 Sensori ( Accelerometro, Giroscopio, Magnetometro )
- 5 Attivita'
- 13 Hz di campionamento, 4 sec di finestre, 50% overlap
### SelfBACKAdapter.ipynb - selfBack
Link: https://archive.ics.uci.edu/dataset/521/selfback  
Caratteristiche:
- 33 Utenti
- 2 Posizioni
- 1 Sensore ( Accelerometro )
- 9 Attivita'
- 100 Hz di campionamento, 4 sec di finestre, 50% overlap
### SDALLEAdapter.ipynb - SDALLE
Link: https://zenodo.org/records/14841611
Caratteristiche:
- 9 Utenti
- 8 Posizioni
- 2 Sensori ( Accelerometro, Giroscopio )
- 4 Attivita'
- 148 Hz di campionamento, 4 sec di finestre, 50% overlap
### AllCreateModelsInOneFile.ipynb
Genera f1-score relativi ai modelli  
Usa le magnitudo ( legacy ), però vengono calcolate anche altre metrice per eventuali scopi futuri  
Eventuali parametri da modificare sono dentro il file `Configurazioni.txt`
#### Dataset MultiPosition wearable
Dentro `Configurazioni.txt` ha il nome di `MultiPositionWearable`  
Non usa i dati relativi al magnetometro  
Mettere il file scaricato(raw_data_all.csv) dentro la cartella `MultiPositionWearable_raw`
#### Dataset selfBack
Dentro `Configurazioni.txt` ha il nome di `selfBACK`  
Mettere le varie cartelle scaricate(t,w e wt) dentro la cartella `selfBACK_raw`
#### Dataset SDALLE
Dentro `Configurazioni.txt` ha il nome di `SDALLE`  
Mettere i vari file scaricati dentro la cartella `SDALLE_raw`
### DisplayResults.ipynb
Genera grafici