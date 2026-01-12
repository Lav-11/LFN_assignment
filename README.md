# Learning From Networks Project
## Predicting voting behavior and community structure in Wikipedia requests for adminship

This project studies Wikipedia’s *Requests for Adminship* (RfA) as a directed network of user interactions, where edges represent votes cast by a user (voter) on another user (candidate). The goal is twofold:
1. **Predict voting behavior**: given a voter–candidate pair, predict whether a vote will occur (*link existence*) and, if so, its **polarity** (*Support / Neutral / Oppose*)
2. **Discover latent communities**: analyze learned node embeddings to identify meaningful user groups and participation roles through unsupervised clustering

We compare **embedding-based baselines** (e.g., Node2Vec + MLP) against **Graph Neural Networks** (GraphSAGE variants), and we evaluate models under both:
- a **transductive** setting (all nodes are known; training/testing differ by edges/time), and
- an **inductive** setting (the message-passing graph contains only training edges; validation/test may include unseen nodes)


---

## Authors

* [@umberto-bianchin](https://www.github.com/umberto-bianchin): umberto.bianchin@studenti.unipd.it
* [@Andre1372](https://github.com/Andre1372): andrea.marigo.3@studenti.unipd.it
* [@Lav-11](https://github.com/Lav-11): luca.lavezzi@studenti.unipd.it


---

## Notebooks

### `Node2Vec_implementation.ipynb`
- Trains **Node2Vec** embeddings on the training voting graph
- Uses the learned node embeddings as input features for a downstream **MLP classifier** for vote prediction

### `gnn_implementation_transductive.ipynb`
- Implements a **GraphSAGE-based GNN** for vote prediction in the **transductive** scenario
- Learns node representations via message passing and predicts vote polarity
- Focuses on “full-graph known” experimentation and model selection in a temporally-split evaluation

### `gnn_implementation_inductive.ipynb`
- Implements the **Hierarchical (multi-task) GNN** for the **inductive** scenario
- Uses a shared GraphSAGE encoder and two heads:
  - **Link head**: predicts *NoVote vs Voted* (participation).
  - **Polarity head**: predicts *Oppose / Neutral / Support* conditional on voting
- Uses neighbor sampling for scalable mini-batch training and evaluates both heads separately (plus the reconstructed 4-class output)

### `clustering_embeddings.ipynb`
- Loads trained GNN embeddings and performs **unsupervised clustering** to study community structure
- Runs **KMeans** with model selection (e.g., silhouette vs. k) and a robustness check with **MeanShift** (optionally after PCA)
- Visualizes the embedding space (e.g., **t-SNE**) and produces **cluster profiling** tables (sizes, in/out activity, vote composition) for interpretation

