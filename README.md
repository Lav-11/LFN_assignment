# Riassunto delle Modifiche
Di seguito sono elencate le principali modifiche e aggiunte apportate alla codebase:

- **Moduli Comuni**: Sono stati aggiunti i file `dataset.py` e `utility.py` per tutte quelle funzionalità comuni utilizzate in più notebook.
  - **`dataset.py`**: Contiene la riscrittura delle logiche di preprocessing, la nuova implementazione del negative sampling (spiegata nel notebook `gnn_implementation_inductive`) e la funzione per lo splitting del dataset.
  - **`utility.py`**: Aggiunge due funzioni utili per la manipolazione delle date e per l'analisi statistica approfondita degli split (inclusi grafici sulla topologia e sovrapposizione utenti).
- **Gestione Dipendenze ed Esecuzione**: È stato incluso il file `requirements.txt`. Per installare le dipendenze, eseguire `pip install -r requirements.txt`.
- **Negative Sampling Migliorato**: Nel notebook `gnn_implementation_inductive`, la logica di `negative_sampling` è stata aggiornata. Ora utilizza una **global forbidden map** per verificare la validità dei campioni negativi, assicurando che non esistano come archi positivi in nessuno degli split (training, validation o test), a differenza della versione precedente che controllava solo lo split corrente.
- **Training Sliding Window**: È stato introdotto il file `gnn_sliding_window_inductive` che implementa una strategia di training basata su finestre temporali (sliding window).

## Future Work
Un obiettivo fondamentale per i prossimi sviluppi è la **standardizzazione dell'algoritmo di training e della procedura di valutazione**.

Attualmente, le variazioni nelle modalità di training o di calcolo delle metriche possono rendere difficile il confronto diretto tra i vari modelli. Stabilire una pipeline comune e rigorosa per il training e l'evaluation permetterà di confrontare i modelli in modo equo, garantendo che le differenze nelle performance siano dovute effettivamente alle capacità predittive del modello e non a discrepanze nel setup sperimentale.
