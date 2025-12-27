import gzip
import locale
from datetime import datetime
from typing import List, Tuple, Dict, Union

import torch
import random
from collections import defaultdict

locale.setlocale(locale.LC_TIME, 'C')

class WikiRfAPreprocessor:
    """
    Gestisce il caricamento e il preprocessing del dataset Wiki-RfA.
    L'obiettivo è estrarre archi (u, v) con attributi temporali e di voto
    per costruire il grafo G=(V,E) definito nella proposta di progetto.
    """

    def __init__(self, file_path: str):
        # Inizializza il dataset con il percorso del file compresso.
        self.file_path = file_path
        self.samples: List[Tuple[str, str, int, int]] = []
        self.errors = {"EMPTY_ERROR": [], "SELF_LOOP": [], "ERROR_OTHERS": []}

    def _parse_date(self, date_str: str) -> Union[int, str]:
        """ Converte la stringa data in intero YYYYMMDD. """
        if len(date_str) == 0:
            return "EMPTY_ERROR"
        
        formats_to_try = [
            "%H:%M, %d %B %Y",  # Es: "23:13, 19 April 2013" (%B = mese completo)
            "%H:%M, %d %b %Y"   # Es: "23:25, 15 Jan 2005"   (%b = mese abbreviato)
        ]
        
        for fmt in formats_to_try:
            try:
                # Trasform the string in an abject datetime
                dt_obj = datetime.strptime(date_str, fmt)
                # Casting and reformatting
                return int(dt_obj.strftime("%Y%m%d"))
            except ValueError:
                # If it fails, pass to the next format
                continue
        
        return "ERROR_OTHERS"
    
    def _is_valid_block(self, block: Dict[str, str]) -> bool:
        """ Verifica la presenza dei campi essenziali SRC, TGT, VOT, DAT. """
        required_keys = {"SRC", "TGT", "VOT", "DAT"}
        return required_keys.issubset(block.keys())

    def _process_block(self, block: Dict[str, str]) -> Union[Tuple, str]:
        """ 
        Processa un blocco di linee e restituisce un dizionario con i campi chiave-valore.
        Altrimenti restituisce la tipologia di errore.
        """
        # 1. Extract date and convert to integer YYYYMMDD
        date_str = block["DAT"].strip()
        int_date = self._parse_date(date_str)
        
        if isinstance(int_date, str): # Se è tornato un errore (stringa)
            return int_date
        
        # 2. Veryfy no self-loops
        if block["SRC"] == block["TGT"]:
            return "SELF_LOOP"

        # 3. Return processed block
        return (block["SRC"], block["TGT"], int(block["VOT"]), int_date)

    def load(self, verbose: bool = False) -> None:
        """Legge il file compresso e popola la lista self.samples."""
        print(f"Loading dataset from {self.file_path}...")
        
        current_block = {}
        
        with gzip.open(self.file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                
                # Stampa le prime righe se richiesto (debug)
                if verbose and line_idx < 10:
                    print(f"Raw line {line_idx}: {line}")

                if not line:
                    # Fine del blocco: processiamo i dati accumulati
                    if self._is_valid_block(current_block):
                        result = self._process_block(current_block)
                        
                        if isinstance(result, str):
                            # Store skipped edges for error analysis
                            self.errors[result].append((current_block["SRC"], current_block["TGT"], current_block["DAT"]))
                        else:
                            self.samples.append(result)
                    
                    current_block = {} # Reset
                else:
                    # Accumulo dati nel blocco corrente
                    try:
                        key, value = line.split(":", 1)
                        current_block[key] = value.strip()
                    except ValueError:
                        continue # Gestione linee malformate

        print("\n" + "-" * 7 + " Dataset Loaded " + "-" * 7)
        self.print_stats()

    def print_stats(self):
        """Stampa statistiche riassuntive del dataset."""
        print(f"Total valid samples loaded: {len(self.samples)}")
        if self.samples:
            print(f"Example sample (SRC, TGT, VOT, DATE): {self.samples[0]}")
        
        print("\nDiscarded edges breakdown:")
        for error_type, error_list in self.errors.items():
            print(f"  {error_type}: {len(error_list)}")
            for edge in error_list[:4]:  # Print first 4 errors of each type
                print(f"    Skipped edge: {edge}")


class CandidateCentricSampler:
    """
    Gestisce il campionamento dei negativi (NoVote) preservando la distribuzione
    dei candidati (Candidate-Centric).
    
    Logica:
    Per ogni arco positivo (u, t) nel set corrente, campioniamo 'ratio' archi negativi (u', t)
    tali che l'arco (u', t) non esista in NESSUNA partizione del dataset (Train/Val/Test).
    """

    def __init__(self, all_pos_tensors):
        """
        Args:
            all_pos_tensors: lista di tensori [X_train, X_val, X_test].
                Serve a costruire la mappa globale dei 'voti proibiti'.
        """
        self.global_forbidden = self._build_global_forbidden(all_pos_tensors)

    def _build_global_forbidden(self, tensor_list):
        """Crea un dizionario {target_id: set(voter_ids)} unendo tutti gli split."""
        forbidden = defaultdict(set)
        for tensor in tensor_list:
            for u, t in tensor.tolist():
                forbidden[int(t)].add(int(u))
        return forbidden

    def sample_negatives(self, target_tensor, num_nodes, ratio=1, seed=42):
        """
        Esegue il sampling per un dato tensore di input (es. X_train).
        Restituisce (X_neg, y_neg).
        """
        rng = random.Random(seed)
        targets = target_tensor[:, 1].tolist() # Prendiamo i candidati dagli archi positivi
        
        X_neg_list = []
        
        # Per ogni voto reale ricevuto dal candidato t, generiamo 'ratio' non-voti
        for t in targets:
            t_idx = int(t)
            forbidden_voters = self.global_forbidden[t_idx]
            
            for _ in range(ratio):
                while True:
                    # Campioniamo un utente casuale
                    u_neg = rng.randrange(num_nodes)
                    
                    # Accettiamo se l'utente non ha mai votato per questo candidato
                    # (nemmeno nel futuro/test set) e non è un self-loop
                    if u_neg not in forbidden_voters and u_neg != t_idx:
                        X_neg_list.append([u_neg, t_idx])
                        break
        
        X_neg = torch.tensor(X_neg_list, dtype=torch.long)
        y_neg = torch.zeros(X_neg.size(0), dtype=torch.long) # 0 = NoVote
        
        return X_neg, y_neg
