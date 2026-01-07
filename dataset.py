# Import standard libraries
import gzip, sys, locale, random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Union
from collections import defaultdict
# Computations
import torch
import numpy as np
import pandas as pd
# Import this project modules
REPO_ROOT = Path('.').resolve()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utility import sum_months

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


def train_val_test_split_by_date(X: torch.Tensor, y: torch.Tensor, dates: np.ndarray, val_months:int=6, test_months:int=6):
    """
    Splits the dataset into train, validation, and test sets based on dates.
    Args:
        X (torch.Tensor): Tensor of shape (num_samples, 2) with edges (u, v).
        y (torch.Tensor): Tensor of shape (num_samples,) with labels (1=Vote, 0=NoVote).
        dates (np.ndarray): Tensor of shape (num_samples,) with dates in YYYYMMDD format.
        val_months (int): Number of months for the validation set.
        test_months (int): Number of months for the test set.
    Returns:
        splits (tuple): A tuple containing three tuples:
        - (X_train, y_train, dates_train)
        - (X_val, y_val, dates_val)
        - (X_test, y_test, dates_test)
    """
    last_date = max(dates)

    # Calculate split dates
    val_start_date = sum_months(last_date, - (val_months + test_months))
    test_start_date = sum_months(last_date, - test_months)

    # Create boolean masks
    mask_train = dates < val_start_date
    mask_val = (dates >= val_start_date) & (dates < test_start_date)
    mask_test = dates >= test_start_date
    
    # Split the data
    X_train, y_train, dates_train = X[mask_train], y[mask_train], dates[mask_train]
    X_val, y_val, dates_val = X[mask_val], y[mask_val], dates[mask_val]
    X_test, y_test, dates_test = X[mask_test], y[mask_test], dates[mask_test]

    return (X_train, y_train, dates_train), (X_val, y_val, dates_val), (X_test, y_test, dates_test)

def extract_node_features(edge_index: torch.Tensor, edge_dates: np.ndarray, num_nodes: int) -> torch.Tensor:
    """
    Compute structural and temporal features for nodes in a dynamic graph.
    
    Args:
        edge_index (torch.Tensor): [2, E] Source (Voter) -> Target (Candidate)
        edge_dates (np.ndarray): [E] Dates in YYYYMMDD format (int)
        num_nodes (int): Total number of nodes in the graph
    Returns:
        torch.Tensor: [num_nodes, 6] Normalized feature matrix
    """
    
    # Conversion: YYYYMMDD to Timestamp Linear (days)
    dates_pd = pd.to_datetime(edge_dates, format='%Y%m%d')
    ref_date = pd.to_datetime(edge_dates.max(), format='%Y%m%d')
    
    df = pd.DataFrame({
        'src': edge_index[0].cpu().numpy(),
        'tgt': edge_index[1].cpu().numpy(),
        'date': dates_pd
    })
    
    # Initialize features to 0 ----> [In-Deg, Out-Deg, Tenure, Recency, Span, Freq]
    features = np.zeros((num_nodes, 6), dtype=np.float32)
    
    # ---------------------------------------------------------
    # 1. Structural Features (Degrees)
    # ---------------------------------------------------------
    in_degree = df.groupby('tgt').size()
    out_degree = df.groupby('src').size()
    
    features[in_degree.index, 0] = in_degree.values  # In-Degree (Votes Received)
    features[out_degree.index, 1] = out_degree.values # Out-Degree (Votes Given)

    # ---------------------------------------------------------
    # 2. Temporal Features (Tenure, Recency, Span)
    # ---------------------------------------------------------
    # Group by SRC (Voter) to calculate T_first_vote and T_last_vote
    grp_src = df.groupby('src')['date'].agg(['min', 'max'])
    
    voter_indices = grp_src.index.values
    t_first = grp_src['min']
    t_last = grp_src['max']
    
    # (.dt.days converts Timedelta to int)
    features[voter_indices, 2] = (ref_date - t_first).dt.days.values   # Tenure (Anzianità): T_now - T_first
    features[voter_indices, 3] = (ref_date - t_last).dt.days.values    # Recency (Recenza): T_now - T_last
    features[voter_indices, 4] = (t_last - t_first).dt.days.values     # Activity Span: T_last - T_first
    
    # ---------------------------------------------------------
    # 3. Derived Features (Frequency)
    # ---------------------------------------------------------
    # Add 1 day to span to avoid division by zero     
    safe_span = features[voter_indices, 4] + 1.0 
    votes_given = features[voter_indices, 1]
    
    features[voter_indices, 5] = votes_given / safe_span

    # ---------------------------------------------------------
    # 4. Logarithmic Normalization (Critical for Neural Networks)
    # ---------------------------------------------------------   
    return torch.tensor(np.log1p(features), dtype=torch.float)