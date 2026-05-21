"""
shard_data.py
-------------
Shards the BoT-IoT preprocessed data into N client partitions with mild Non-IID.

Usage:
    python shard_data.py --data_path /path/to/botiot.csv \
                         --num_clients 5 \
                         --output_dir ./client_data \
                         --alpha 0.5

Non-IID control:
    alpha (Dirichlet concentration):
        - Lower  (0.1)  → very non-iid  (each client sees few classes)
        - Medium (0.5)  → mild non-iid  ← recommended for this imbalanced dataset
        - Higher (5.0)  → near-iid
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# Preprocessing (mirrors your main project)
# ─────────────────────────────────────────────────────────────

def preprocess(df, target='category'):
    columns_to_keep = [
        'pkSeqID', 'stime', 'flgs', 'flgs_number', 'proto', 'proto_number',
        'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state',
        'state_number', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min',
        'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate',
        'TnBPSrcIP', 'TnBPDstIP', 'TnP_PSrcIP', 'TnP_PDstIP', 'TnP_PerProto',
        'TnP_Per_Dport', 'AR_P_Proto_P_SrcIP', 'AR_P_Proto_P_DstIP',
        'N_IN_Conn_P_DstIP', 'N_IN_Conn_P_SrcIP', 'AR_P_Proto_P_Sport',
        'AR_P_Proto_P_Dport', 'Pkts_P_State_P_Protocol_P_DestIP',
        'Pkts_P_State_P_Protocol_P_SrcIP', 'attack', 'category', 'subcategory'
    ]
    df = df[[c for c in columns_to_keep if c in df.columns]].copy()
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    le = LabelEncoder()
    y = le.fit_transform(df[target])
    classes = le.classes_
    print(f"\nTarget: {target} | Classes: {list(classes)}")

    drop_cols = ['pkSeqID', 'stime', 'ltime', 'saddr', 'sport', 'daddr', 'dport',
                 'attack', 'category', 'subcategory']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.fillna(0)

    cat_cols = [c for c in ['flgs', 'proto', 'state'] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for col in cat_cols:
        cle = LabelEncoder()
        X[col] = cle.fit_transform(X[col].astype(str))

    X_arr = X.values.astype(np.float32)
    num_idx = [X.columns.tolist().index(c) for c in num_cols]
    scaler = StandardScaler()
    X_arr[:, num_idx] = scaler.fit_transform(X_arr[:, num_idx])

    print(f"Dataset shape: {X_arr.shape}")
    return X_arr, y, classes


# ─────────────────────────────────────────────────────────────
# Non-IID Dirichlet Sharding
# ─────────────────────────────────────────────────────────────

def dirichlet_partition(y, num_clients, alpha, seed=42):
    """
    Partition indices using a Dirichlet distribution over class labels.
    alpha controls non-iid-ness: lower = more non-iid.
    """
    np.random.seed(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    # For each class, draw proportions for clients
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)

        # Draw proportions from Dirichlet
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Split indices according to proportions
        splits = (proportions * len(cls_idx)).astype(int)
        # Fix rounding so we don't lose samples
        splits[-1] = len(cls_idx) - splits[:-1].sum()

        start = 0
        for c, size in enumerate(splits):
            client_indices[c].extend(cls_idx[start:start + size].tolist())
            start += size

    # Shuffle within each client
    for c in range(num_clients):
        np.random.shuffle(client_indices[c])

    return client_indices


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def plot_distribution(y, client_indices, classes, output_dir):
    num_clients = len(client_indices)
    n_classes = len(classes)

    dist = np.zeros((num_clients, n_classes), dtype=int)
    for c, idxs in enumerate(client_indices):
        for cls in range(n_classes):
            dist[c, cls] = np.sum(y[idxs] == cls)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Stacked bar chart
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for cls in range(n_classes):
        axes[0].bar(range(num_clients), dist[:, cls],
                    bottom=bottom, label=classes[cls], color=colors[cls])
        bottom += dist[:, cls]
    axes[0].set_title('Class Distribution per Client', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Sample Count')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xticks(range(num_clients))

    # Heatmap of proportions
    prop = dist / dist.sum(axis=1, keepdims=True)
    sns.heatmap(prop, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=classes, yticklabels=[f'Client {i}' for i in range(num_clients)],
                ax=axes[1])
    axes[1].set_title('Class Proportion per Client', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Class')

    plt.tight_layout()
    path = os.path.join(output_dir, 'data_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Distribution plot saved → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Shard BoT-IoT for Federated Learning')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the BoT-IoT CSV file')
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Number of FL clients')
    parser.add_argument('--target', type=str, default='category',
                        choices=['attack', 'category', 'subcategory'],
                        help='Classification target')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha (0.1=very non-iid, 0.5=mild, 5.0=near-iid)')
    parser.add_argument('--output_dir', type=str, default='./client_data',
                        help='Output directory for client shards')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  BoT-IoT FL Data Sharding")
    print("=" * 60)
    print(f"  Clients  : {args.num_clients}")
    print(f"  Target   : {args.target}")
    print(f"  Alpha    : {args.alpha}  (Dirichlet non-IID)")
    print(f"  Output   : {args.output_dir}")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = pd.read_csv(args.data_path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Preprocess
    X, y, classes = preprocess(df, target=args.target)

    # Partition
    print(f"\nPartitioning with Dirichlet(alpha={args.alpha}) ...")
    client_indices = dirichlet_partition(y, args.num_clients, args.alpha, args.seed)

    # Save each client's shard
    print("\nClient shard summary:")
    print(f"{'Client':<10} {'Samples':<10} {'Breakdown'}")
    print("-" * 60)

    for cid, idxs in enumerate(client_indices):
        idxs = np.array(idxs)
        X_c = X[idxs]
        y_c = y[idxs]

        # Save
        np.save(os.path.join(args.output_dir, f'client_{cid}_X.npy'), X_c)
        np.save(os.path.join(args.output_dir, f'client_{cid}_y.npy'), y_c)

        breakdown = {classes[cls]: int((y_c == cls).sum())
                     for cls in range(len(classes))}
        breakdown_str = ', '.join(f'{k}:{v}' for k, v in breakdown.items())
        print(f"  Client {cid:<4} {len(idxs):<10} {breakdown_str}")

    # Save metadata
    np.save(os.path.join(args.output_dir, 'classes.npy'), classes)
    meta = {
        'num_clients': args.num_clients,
        'target': args.target,
        'n_features': X.shape[1],
        'n_classes': len(classes),
        'alpha': args.alpha,
        'classes': list(classes)
    }
    import json
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nMetadata saved → {args.output_dir}/meta.json")

    # Visualise
    plot_distribution(y, client_indices, classes, args.output_dir)
    print("\nDone! ✓")


if __name__ == '__main__':
    main()