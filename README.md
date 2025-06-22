# Tesi/Tirocinio Triennale Chiarini Emiliano
## Motivazione scelta del modello XGB su RFT
Far partire il file `Difference_XGB_RFT.py` tramite il comando (in questo caso io ho usato powershell)
```powershell
foreach ($DATASET in "MultiPositionWearable", "selfBACK", "SDALLE") {
    python Difference_XGB_RFT.py $DATASET
}
```
I risultati della comparazione saranno visibili nella cartella `/images/differenza tra modelli/NOME_DATASET/plot_X.png`  
Nel caso si vogliano modificare i parametri, modificare:
```python
(386) RANDOM_STATE_LIST = [ ... ]
(387) MINUTE_LIST = [ ... ]
(388) WEIGHT_LIST = [ ... ]
```

## Creazione ed Analisi Modelli
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
Eventuali parametri da modificare sono dentro il file `/Configurazioni.txt`
#### Dataset MultiPosition wearable
Dentro `Configurazioni.txt` ha il nome di `MultiPositionWearable`  
Non usa i dati relativi al magnetometro  
Mettere il file scaricato(raw_data_all.csv) dentro la cartella `/data/MultiPositionWearable_raw/`
#### Dataset selfBack
Dentro `Configurazioni.txt` ha il nome di `selfBACK`  
Mettere le varie cartelle scaricate(t,w e wt) dentro la cartella `/data/selfBACK_raw/`
#### Dataset SDALLE
Dentro `Configurazioni.txt` ha il nome di `SDALLE`  
Mettere i vari file scaricati dentro la cartella `/data/SDALLE_raw/`
### DisplayResults.ipynb
Genera le immagini dei grafici della cartella `/images/`