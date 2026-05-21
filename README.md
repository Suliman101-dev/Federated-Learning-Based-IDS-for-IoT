# Federated Learning for IoT Intrusion Detection

FT-Transformer trained with Flower across 5 edge clients on the BoT-IoT dataset — no raw data leaves the device.

---

## What this is

This project builds a network intrusion detection system for IoT environments using Federated Learning. Instead of sending raw traffic data to a central server, each edge client trains locally on its own data shard and only shares model weight updates. The server aggregates them using FedAvg and repeats for 5 global rounds.

The model is an **FT-Transformer** — a transformer architecture built specifically for tabular data that embeds each network-flow feature individually before passing them through attention layers.

**Final result: 99.73% accuracy, AUC 1.0000 on a held-out test set of 270,869 samples.**

---

## Dataset

[BoT-IoT](https://www.kaggle.com/datasets/azdinsahir/bot-iot-processed) — 1.35M network flow records across 5 categories:
`DDoS` · `DoS` · `Normal` · `Reconnaissance` · `Theft`

The dataset is partitioned across clients using Dirichlet sampling (α = 0.5) to simulate realistic non-IID distributions — each client ends up with a different class imbalance, just like real IoT sensors would.

---

## Project Structure

```
├── data-sharding.ipynb     # Data prep: preprocessing, partitioning, saving shards
├── server.ipynb            # FL server: FedAvg aggregation + centralised evaluation
├── edge_1.ipynb            # Edge client (duplicate and change client_id for each)
└── client_data/            # Generated after running data-sharding (not tracked)
    ├── client_0_X.npy
    ├── client_0_y.npy
    ├── ...
    ├── test_X.npy
    ├── test_y.npy
    └── meta.json
```

---

## Requirements

```bash
pip install flwr torch scikit-learn numpy pandas kagglehub
```

Tested on Python 3.10+. If running on Google Colab, all dependencies can be installed inline inside the notebooks.

---

## How to Run

The notebooks must be run in this exact order. The server blocks until all clients connect, so read through before starting.

### Step 1 — Prepare the data

Open and run `data-sharding.ipynb` from top to bottom.

This will:
- Download the BoT-IoT dataset from Kaggle via `kagglehub`
- Preprocess features (drop irrelevant columns, label encode, StandardScaler)
- Split off a global test set (20% stratified)
- Partition the remaining data into 5 client shards using Dirichlet α = 0.5
- Save everything to `/content/client_data/`

Key parameters you can change at the top of the execution cell:
```python
NUM_CLIENTS = 5
ALPHA       = 0.5   # lower = more non-IID
SEED        = 42
TEST_SPLIT  = 0.2
```

Download the `client_data/` folder before closing — you'll need it for the next two steps.

---

### Step 2 — Start the server

Open `server.ipynb` and run it. The server will print:

```
Starting server on 0.0.0.0:9090
```

and wait. **Keep this session open.** It will not proceed until all 5 clients connect.

---

### Step 3 — Start the clients

Open 5 separate sessions of `edge_1.ipynb`. In each one, change the `client_id` variable at the top of the execution cell:

```python
client_id = 0   # change to 1, 2, 3, 4 for the other sessions
```

Make sure `data_dir` points to where you saved `client_data/` and `server_address` matches the server:

```python
data_dir       = '/content/client_data'
server_address = 'localhost:9090'
```

Once all 5 clients connect, training starts automatically.

---

### Step 4 — Collect results

After 5 rounds complete, the server saves:

```
fl_outputs/
├── global_model.pth          # final aggregated model weights
├── server_metrics.json       # round-by-round test set metrics
├── fl_metrics.json           # per-client training metrics
└── fl_training_curves.png    # 4-panel convergence plot
```

---

## Model Architecture

The **Feature Tokenizer Transformer (FT-Transformer)** treats each input feature as its own token rather than flattening everything into one vector.

| Component | Config |
|---|---|
| Input features | 36 |
| Feature tokenizer | 36 × Linear(1 → 128) |
| [CLS] token | Learnable, shape (1, 1, 128) |
| Transformer layers | 6 |
| Attention heads | 8 |
| FFN dimension | 512 |
| Dropout | 0.1 |
| Classification head | LayerNorm → ReLU → Dropout → Linear(128, 5) |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |

---

## Results

| Round | Accuracy | F1 | AUC |
|---|---|---|---|
| 0 (init) | 0.56% | 0.0001 | 0.6683 |
| 1 | 99.07% | 0.9870 | 0.9989 |
| 3 | 99.75% | 0.9971 | 0.9999 |
| 5 | 99.73% | 0.9970 | 1.0000 |

Accuracy exceeds 99% after a single round despite non-IID data across all clients.

---

## Limitations

- All clients are simulated on a single machine / Colab, not real edge hardware
- 5 clients and 5 rounds is a small-scale setup
- No differential privacy on weight updates
- Static dataset — not a real-time traffic stream
