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
    Handles loading and preprocessing of the Wiki-RfA dataset.
    The goal is to extract edges (u, v) with temporal and vote attributes
    to build the graph G=(V,E) defined in the project proposal.
    """

    def __init__(self, file_path: str):
        # Initialize the dataset with the path to the compressed file.
        self.file_path = file_path
        self.samples: List[Tuple[str, str, int, int]] = []
        self.errors = {"EMPTY_ERROR": [], "SELF_LOOP": [], "ERROR_OTHERS": []}

    def _parse_date(self, date_str: str) -> Union[int, str]:
        """Converts the date string to an integer in YYYYMMDD format."""
        if len(date_str) == 0:
            return "EMPTY_ERROR"
        
        formats_to_try = [
            "%H:%M, %d %B %Y",  # e.g., "23:13, 19 April 2013" (%B = full month name)
            "%H:%M, %d %b %Y"   # e.g., "23:25, 15 Jan 2005"   (%b = abbreviated month name)
        ]
        
        for fmt in formats_to_try:
            try:
                # Transform the string into a datetime object
                dt_obj = datetime.strptime(date_str, fmt)
                # Cast and reformat
                return int(dt_obj.strftime("%Y%m%d"))
            except ValueError:
                # If it fails, move to the next format
                continue
        
        return "ERROR_OTHERS"
    
    def _is_valid_block(self, block: Dict[str, str]) -> bool:
        """Checks for the presence of the essential fields SRC, TGT, VOT, DAT."""
        required_keys = {"SRC", "TGT", "VOT", "DAT"}
        return required_keys.issubset(block.keys())

    def _process_block(self, block: Dict[str, str]) -> Union[Tuple, str]:
        """ 
        Processes a block of lines and returns a tuple with key fields.
        Otherwise, returns the corresponding error type.
        """
        # 1. Extract date and convert to integer YYYYMMDD
        date_str = block["DAT"].strip()
        int_date = self._parse_date(date_str)
        
        if isinstance(int_date, str):  # If an error was returned (string)
            return int_date
        
        # 2. Verify no self-loops
        if block["SRC"] == block["TGT"]:
            return "SELF_LOOP"

        # 3. Return processed block
        return (block["SRC"], block["TGT"], int(block["VOT"]), int_date)

    def load(self, verbose: bool = False) -> None:
        """Reads the compressed file and populates the self.samples list."""
        print(f"Loading dataset from {self.file_path}...")
        
        current_block = {}
        
        with gzip.open(self.file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                
                # Print the first lines if requested (debug)
                if verbose and line_idx < 10:
                    print(f"Raw line {line_idx}: {line}")

                if not line:
                    # End of block: process the accumulated data
                    if self._is_valid_block(current_block):
                        result = self._process_block(current_block)
                        
                        if isinstance(result, str):
                            # Store skipped edges for error analysis
                            self.errors[result].append((current_block["SRC"], current_block["TGT"], current_block["DAT"]))
                        else:
                            self.samples.append(result)
                    
                    current_block = {}  # Reset
                else:
                    # Accumulate data in the current block
                    try:
                        key, value = line.split(":", 1)
                        current_block[key] = value.strip()
                    except ValueError:
                        continue  # Handle malformed lines

        print("\n" + "-" * 7 + " Dataset Loaded " + "-" * 7)
        self.print_stats()

    def print_stats(self):
        """Prints summary statistics for the dataset."""
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
    Handles negative sampling (NoVote) while preserving the candidate distribution
    (Candidate-Centric).
    
    Logic:
    For each positive edge (u, t) in the current split, we sample 'ratio' negative edges (u', t)
    such that the edge (u', t) does not exist in ANY dataset partition (Train/Val/Test).
    """

    def __init__(self, all_pos_tensors):
        """
        Args:
            all_pos_tensors: list of tensors [X_train, X_val, X_test].
                Used to build the global 'forbidden votes' map.
        """
        self.global_forbidden = self._build_global_forbidden(all_pos_tensors)

    def _build_global_forbidden(self, tensor_list):
        """Creates a dict {target_id: set(voter_ids)} by merging all splits."""
        forbidden = defaultdict(set)
        for tensor in tensor_list:
            for u, t in tensor.tolist():
                forbidden[int(t)].add(int(u))
        return forbidden

    def sample_negatives(self, target_tensor, num_nodes, ratio=1, seed=42):
        """
        Performs sampling for a given input tensor (e.g., X_train).
        Returns (X_neg, y_neg).
        """
        rng = random.Random(seed)
        targets = target_tensor[:, 1].tolist()  # Take candidates from positive edges
        
        X_neg_list = []
        
        # For each real vote received by candidate t, generate 'ratio' no-votes
        for t in targets:
            t_idx = int(t)
            forbidden_voters = self.global_forbidden[t_idx]
            
            for _ in range(ratio):
                while True:
                    # Sample a random user
                    u_neg = rng.randrange(num_nodes)
                    
                    # Accept if the user never voted for this candidate
                    # (not even in the future/test set) and it's not a self-loop
                    if u_neg not in forbidden_voters and u_neg != t_idx:
                        X_neg_list.append([u_neg, t_idx])
                        break
        
        X_neg = torch.tensor(X_neg_list, dtype=torch.long)
        y_neg = torch.zeros(X_neg.size(0), dtype=torch.long)  # 0 = NoVote
        
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
    
    # Conversion: YYYYMMDD to a linear timestamp (days)
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
    
    features[in_degree.index, 0] = in_degree.values   # In-Degree (Votes Received)
    features[out_degree.index, 1] = out_degree.values # Out-Degree (Votes Given)

    # ---------------------------------------------------------
    # 2. Temporal Features (Tenure, Recency, Span)
    # ---------------------------------------------------------
    # Group by SRC (Voter) to compute T_first_vote and T_last_vote
    grp_src = df.groupby('src')['date'].agg(['min', 'max'])
    
    voter_indices = grp_src.index.values
    t_first = grp_src['min']
    t_last = grp_src['max']
    
    # (.dt.days converts Timedelta to int)
    features[voter_indices, 2] = (ref_date - t_first).dt.days.values  # Tenure: T_now - T_first
    features[voter_indices, 3] = (ref_date - t_last).dt.days.values   # Recency: T_now - T_last
    features[voter_indices, 4] = (t_last - t_first).dt.days.values    # Activity span: T_last - T_first
    
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
