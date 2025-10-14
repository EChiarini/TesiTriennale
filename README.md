# Tesi/Tirocinio Triennale Chiarini Emiliano
## Motivazione scelta del modello XGB rispetto al modello RFT
Far partire il file `Difference_XGB_RFT.py` tramite il comando (in questo caso io ho usato powershell)
```powershell
foreach ($DATASET in "selfBACK", "SDALLE") {
    python Difference_XGB_RFT.py $DATASET
}
```
I risultati della comparazione saranno visibili nella cartella `/images/differenza tra modelli/NOME_DATASET/plot_X.png`  
Nel caso si vogliano modificare i parametri, modificare:
```python
(386) RANDOM_STATE_LIST = [ ... ]
(387) SECONDS_LIST = [ ... ]
(388) WEIGHT_LIST = [ ... ]
```
## Motivazione scelta di pesare una riga rispetto a moltiplicarla N volte
Far partire il file `Difference_MUL_PESO.py` tramite il comando (in questo caso io ho usato powershell)
```powershell
foreach ($DATASET in "selfBACK", "SDALLE") {
    python Difference_MUL_PESO.py $DATASET
}
```
I risultati della comparazione saranno visibili nella cartella `/images/diff_peso_mul/NOME_DATASET/plot_X.png`  
Nel caso si vogliano modificare i parametri, modificare:
```python
(356) RANDOM_STATE_LIST = [ ... ]
(357) SECONDS_LIST = [ ... ]
(358) WEIGHT_LIST = [ ... ]
```

## Creazione ed Analisi Modelli
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
#### Dataset selfBack
Dentro `Configurazioni.txt` ha il nome di `selfBACK`  
Mettere le varie cartelle scaricate(t,w e wt) dentro la cartella `/data/selfBACK_raw/`  
Range di `SECONDS_LIST` in `/Configurazioni.txt` : `4 - 40`, a multipli di 2
#### Dataset SDALLE
Dentro `Configurazioni.txt` ha il nome di `SDALLE`  
Mettere i vari file scaricati dentro la cartella `/data/SDALLE_raw/`  
Range di `SECONDS_LIST` in `/Configurazioni.txt` : `4 - 20`, a multipli di 2
### DisplayResults.ipynb
Genera le immagini dei grafici nella cartella `/images/andamento_modelli/dataset di riferimento`  
Per ora genera i grafici relativi al guadagno assoluto e relativo(rispetto al peso 1) 