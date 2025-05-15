# Tesi/Tirocinio Triennale Chiarini Emiliano
### SetUpData.ipynb - MultiPosition wearable dataset
Link: https://zenodo.org/records/11219118  
Caratteristiche:
- 10 Utenti
- 8 Posizioni
- 3 Sensori ( Accelerometro, Giroscopio, Magnetometro )
- 5 Attivita'
- 13 Hz di campionamento, 4 sec di finestre, 2 sec di hop
### SelfBACKAdapter.ipynb - selfBack
Link: https://archive.ics.uci.edu/dataset/521/selfback  
Caratteristiche:
- 33 Utenti
- 2 Posizioni
- 1 Sensore ( Accelerometro )
- 9 Attivita'
- 100 Hz di campionamento, 4 sec di finestre, 2 sec di hop
### ?Adapter.ipynb - ?
Link: ?  
Caratteristiche:
- ? Utenti
- ? Posizioni
- ? Sensore ( ? )
- ? Attivita'
- ? Hz di campionamento, 4 sec di finestre, 2 sec di hop
### AllCreateModelsInOneFile.ipynb
Genera f1-score relativi ai modelli  
Bilancia il dataset con il sotto-insieme di 'utente-posizione-attivita' minore  
Usa le magnitudo ( legacy ), però vengono calcolate anche altre metrice per eventuali scopi futuri  
Assegnamento fisso del 20% di ogni utente alla fase di testing
#### Dataset MultiPosition wearable
Non usa i dati relativi al magnetometro  
#### Dataset selfBack
#### Dataset ?
### DisplayResults.ipynb
Genera grafici