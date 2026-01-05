# Riassunto delle Modifiche

## Commit: Standardized training GNN (January 5, 2026)
Di seguito sono elencate le principali modifiche e aggiunte apportate alla codebase:

- **Moduli Comuni**: Sono stati aggiunti i file `dataset.py`, `utility.py` e `training.py` per tutte quelle funzionalità comuni utilizzate in più notebook.
  - **`dataset.py`**: Contiene la riscrittura delle logiche di preprocessing, la nuova implementazione del negative sampling (spiegata nel notebook `gnn_implementation_inductive`) e la funzione per lo splitting del dataset.
  - **`utility.py`**: Aggiunge due funzioni utili per la manipolazione delle date e per l'analisi statistica approfondita degli split (inclusi grafici sulla topologia e sovrapposizione utenti).
  - **`training.py`**: Implementa l'algoritmo di training valido unicamente per link prediction + vote polarity (`train_hierarchical`) e di valutazione (`eval_hierarchical`) in modo univoco. Questo standardizza la pipeline di addestramento, permettendo un confronto diretto e coerente tra diversi esperimenti e notebook.
- **Gestione Dipendenze ed Esecuzione**: È stato incluso il file `requirements.txt`. Per installare le dipendenze, eseguire `pip install -r requirements.txt`.
- **Negative Sampling Migliorato**: Nel notebook `gnn_implementation_inductive`, la logica di `negative_sampling` è stata aggiornata. Ora utilizza una **global forbidden map** per verificare la validità dei campioni negativi, assicurando che non esistano come archi positivi in nessuno degli split (training, validation o test), a differenza della versione precedente che controllava solo lo split corrente.
- **Training Sliding Window**: È stato introdotto il file `gnn_sliding_window_inductive` che implementa una strategia di training basata su finestre temporali (sliding window).


## Commit: Polarity only transductive GNN (January 5, 2026)
Di seguito sono elencate le principali modifiche e aggiunte apportate alla codebase:

- **Estensione Training (Polarity-Only)**: Il modulo `training.py` è stato arricchito con le funzioni `train_polarity_only` ed `eval_polarity`. Queste permettono di addestrare e valutare il modello focalizzandosi esclusivamente sulla predizione della polarità (3 classi), disaccoppiando questo task dalla link prediction gerarchica.
- **Approccio Transduttivo**: È stato aggiunto il notebook `gnn_implementation_transductive.ipynb`. Questo introduce un'implementazione GNN transduttiva che lavora sull'intero grafo statico mascherato, offrendo un'alternativa all'approccio induttivo e sliding-window precedentemente implementato.

#### Todo
- Verificare che GraphSAGE e il message passing funzionino correttamente per il grafo diretto.