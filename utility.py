
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import pandas as pd
import seaborn as sns

def sum_months(date: int, months: int) -> int:
    """
    Utility function to sum or subtract months to a date in YYYYMMDD format.
    Args:
        date (int): Date in YYYYMMDD format.
        months (int): Number of months to add (can be negative).
    Returns:
        new_date (int): New date in YYYYMMDD format after adding the months.    
    """
    date_temp, date_days = divmod(date, 100)
    date_year, date_month = divmod(date_temp, 100)

    total_months = date_month + months
    add_year, new_month = divmod(total_months, 12)
    if total_months % 12 == 0:
        add_year -= 1
        new_month = 12

    new_date = str(date_year + add_year) + (str(new_month).zfill(2)) + (str(date_days).zfill(2))
    new_date = int(new_date)

    return new_date


def analyze_split_statistics(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Performs a detailed analysis of the topology and data distribution across temporal splits.
    Args:
        X_train, y_train (torch.Tensor): Training set edges and labels.
        X_val, y_val (torch.Tensor): Validation set edges and labels.
        X_test, y_test (torch.Tensor): Test set edges and labels.
    Returns:
        None (prints and plots the analysis).
    """
    
    # --- CONFIGURAZIONE ---
    class_map = {1: "Oppose (-1)", 2: "Neutral (0)", 3: "Support (1)"}
    
    splits = {
        "Train": (X_train, y_train),
        "Val":   (X_val, y_val),
        "Test":  (X_test, y_test)
    }

    print(f"\n{' DATASET SPLITTING ANALYSIS ':=^70}")

    # ---------------------------------------------------------
    # 1. Topology & Density Stats
    # ---------------------------------------------------------
    print("\n[1] Topology & Density (Graph Connectivity)")

    stats_data = []
    nodes_sets = {}
    edges_sets = {}

    for name, (X, y) in splits.items():
        # Nodi unici
        nodes = set(X.flatten().tolist())
        nodes_sets[name] = nodes
        
        # Archi (Tuples) - Grafo Diretto: (u,v) != (v,u)
        edges = set(map(tuple, X.tolist()))
        edges_sets[name] = edges
        
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        # Se < 1.0 il grafo è molto frammentato (foresta di alberi o nodi isolati)
        avg_deg = n_edges / n_nodes if n_nodes > 0 else 0
        
        stats_data.append({
            "Split": name,
            "Edges": n_edges,
            "Nodes": n_nodes,
            "Avg Degree": f"{avg_deg:.2f}"
        })

    df_stats = pd.DataFrame(stats_data)
    print(df_stats.set_index("Split"))

    # ---------------------------------------------------------
    # 2. User Overlap Analysis
    # ---------------------------------------------------------
    print("\n[2] User Overlap & Cold Start Diagnostics (Inductive vs Transductive)")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1 - Venn Diagram
    v = venn3([nodes_sets["Train"], nodes_sets["Val"], nodes_sets["Test"]], 
              set_labels=('Train', 'Validation', 'Test'), 
              ax=axes[0])
    axes[0].set_title("Set Intersections (User Overlap)", fontsize=14, fontweight='bold')

    # Plot 2 - Cold Start Analysis (Stacked Bar)
    split_names = ['Val', 'Test']
    new_counts = [len(nodes_sets[n] - nodes_sets["Train"]) for n in split_names]
    seen_counts = [len(nodes_sets[n].intersection(nodes_sets["Train"])) for n in split_names]
        
    p1 = axes[1].bar(split_names, seen_counts, label='Seen in Train (Transductive)', color='#a8dadc')
    p2 = axes[1].bar(split_names, new_counts, bottom=seen_counts, label='New Nodes (Inductive)', color='#e63946')
    
    axes[1].set_ylabel('Number of Users')
    axes[1].set_title("Cold Start Diagnostics: Known vs New Nodes", fontsize=14, fontweight='bold')
    axes[1].legend()
    
    for i, (seen, new) in enumerate(zip(seen_counts, new_counts)):
        total = seen + new
        if total > 0:
            pct_new = (new / total) * 100
            axes[1].text(i, total + (total*0.02), f"{pct_new:.1f}% New", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 3. Edge Duplicates
    # ---------------------------------------------------------
    print("\n[3] Edge Duplicates Analysis")
    duplicated_edges = {
        "Train-Val": edges_sets["Train"] & edges_sets["Val"],
        "Train-Test": edges_sets["Train"] & edges_sets["Test"],
        "Val-Test": edges_sets["Val"] & edges_sets["Test"]
    }
    
    for pair, links in duplicated_edges.items():
        print(f"Found {len(links)} duplicate edges between {pair}")

    # ---------------------------------------------------------
    # 4. Class Distribution Analysis (Label Shift)
    # ---------------------------------------------------------
    print("\n[4] Class Distribution Analysis (Vote Polarity)")
    fig2, ax_dist = plt.subplots(figsize=(16, 6))
    
    df_list = []
    for name, (_, y) in splits.items():
        temp_df = pd.DataFrame({'Label': y.numpy()})
        temp_df['Split'] = name
        temp_df['Class Name'] = temp_df['Label'].map(class_map)
        df_list.append(temp_df)
    df_all = pd.concat(df_list)

    # Compute percentages to normalize the bars
    props = df_all.groupby("Split")['Class Name'].value_counts(normalize=True).rename("Percentage").reset_index()
    props['Percentage'] *= 100
    
    # Order the splits logically
    props['Split'] = pd.Categorical(props['Split'], categories=['Train', 'Val', 'Test'], ordered=True)
    
    sns.barplot(data=props, x='Split', y='Percentage', hue='Class Name', ax=ax_dist, palette="viridis")
    
    ax_dist.set_title("Label Shift Check (Class Distribution)", fontsize=14, fontweight='bold')
    ax_dist.set_ylabel("Percentage (%)")
    ax_dist.set_ylim(0, 100)
    ax_dist.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
