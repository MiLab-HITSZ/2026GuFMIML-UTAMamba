
import sys
import numpy as np
import torch
import random
from sklearn.metrics import (
    average_precision_score,
    hamming_loss,
    coverage_error,
    label_ranking_loss,
    precision_score,
    recall_score,
    f1_score
)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _stable_sigmoid(x):
    x = np.clip(x, -88.72, 88.72)
    return 1 / (1 + np.exp(-x))

def _sanitize_scores(y_score):
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(y_score, 0.0, 1.0)

def evaluate_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if np.any(y_score > 1.0) or np.any(y_score < 0.0):
        y_prob = _stable_sigmoid(y_score)
    else:
        y_prob = y_score
    y_prob = _sanitize_scores(y_prob)
    y_pred = (y_prob >= threshold).astype(np.int32)
    
    try: ham = hamming_loss(y_true, y_pred)
    except: ham = 0.0
    
    try:
        top_indices = np.argmax(y_prob, axis=1)
        relevant_at_top = y_true[np.arange(len(y_true)), top_indices]
        one_error = 1.0 - np.mean(relevant_at_top)
    except: one_error = 1.0
        
    try: cov = coverage_error(y_true, y_prob)
    except: cov = 0.0
    
    try: rank_loss = label_ranking_loss(y_true, y_prob)
    except: rank_loss = 0.0
        
    try:
        micro_prec = precision_score(y_true, y_pred, average='micro', zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    except: micro_prec, macro_prec = 0.0, 0.0

    try:
        micro_rec = recall_score(y_true, y_pred, average='micro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    except: micro_rec, macro_rec = 0.0, 0.0
        
    try:
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except: micro_f1, macro_f1 = 0.0, 0.0
    
    try:
        aps = []
        for i in range(y_true.shape[1]):
            if np.all(y_true[:, i] == 0) or np.all(y_true[:, i] == 1):
                continue
            aps.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        mAP = np.mean(aps) if aps else 0.0
    except: mAP = 0.0

    try:
        # mAP_per_sample: Calculate average precision per sample and average them
        # This is equivalent to average_precision_score with average='samples'
        mAP_per_sample = average_precision_score(y_true, y_prob, average='samples')
    except: mAP_per_sample = 0.0

    return {
        "mAP": mAP,
        "mAP_per_sample": mAP_per_sample,
        "HammingLoss": ham,
        "OneError": one_error,
        "Coverage": cov,
        "RankingLoss": rank_loss,
        "MicroPrecision": micro_prec,
        "MacroPrecision": macro_prec,
        "MicroRecall": micro_rec,
        "MacroRecall": macro_rec,
        "MicroF1": micro_f1,
        "MacroF1": macro_f1
    }

def _get_labels_numpy(dataset):
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        labels = np.asarray(labels)
        if labels.ndim == 2:
            return labels

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        idxs = np.asarray(dataset.indices, dtype=np.int64)
        if hasattr(base, "labels"):
            base_labels = np.asarray(getattr(base, "labels"))
            if base_labels.ndim == 2 and len(base_labels) >= int(idxs.max(initial=0) + 1):
                return base_labels[idxs]
        out = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            out.append(np.asarray(y, dtype=np.float32))
        return np.stack(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)

    out = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        out.append(np.asarray(y, dtype=np.float32))
    return np.stack(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)

def split_dataset(dataset, num_clients, partition_mode="iid", dirichlet_alpha=0.5, labels_per_client=2, seed=42):
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    partition_mode = (partition_mode or "iid").lower()

    n = len(dataset)
    if n == 0:
        return [Subset(dataset, []) for _ in range(num_clients)], [None for _ in range(num_clients)]

    if partition_mode == "iid":
        indices = rng.permutation(n).tolist()
        base = n // num_clients
        sizes = [base] * num_clients
        for i in range(n - base * num_clients):
            sizes[i] += 1
        splits = []
        start = 0
        for sz in sizes:
            end = start + sz
            splits.append(Subset(dataset, indices[start:end]))
            start = end
        return splits, [None for _ in range(num_clients)]

    labels = _get_labels_numpy(dataset)
    if labels.size == 0:
        return split_dataset(dataset, num_clients, partition_mode="iid", seed=seed)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape={labels.shape}")

    num_classes = labels.shape[1]
    all_idxs = np.arange(n)

    if partition_mode == "dirichlet":
        labels_bin = (labels > 0).astype(np.int32)
        for _ in range(50):
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = all_idxs[labels_bin[:, k] > 0]
                if len(idx_k) == 0:
                    continue
                rng.shuffle(idx_k)
                proportions = rng.dirichlet(np.repeat(float(dirichlet_alpha), num_clients))
                cut_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                split = np.split(idx_k, cut_points)
                for i, part in enumerate(split):
                    if len(part) > 0:
                        idx_batch[i].extend(part.tolist())

            idx_batch = [sorted(set(b)) for b in idx_batch]
            sizes = [len(b) for b in idx_batch]
            if min(sizes) >= 1:
                return [Subset(dataset, b) for b in idx_batch], [None for _ in range(num_clients)]

        return split_dataset(dataset, num_clients, partition_mode="iid", seed=seed)

    if partition_mode == "label":
        labels_bin = (labels > 0).astype(np.int32)
        client_allowed = []
        for _ in range(num_clients):
            allowed = rng.choice(np.arange(num_classes), size=min(int(labels_per_client), num_classes), replace=False)
            client_allowed.append(set(int(x) for x in allowed.tolist()))

        idx_batch = [[] for _ in range(num_clients)]
        for idx in range(n):
            present = set(np.where(labels_bin[idx] > 0)[0].tolist())
            for cid, allowed in enumerate(client_allowed):
                if len(present & allowed) > 0:
                    idx_batch[cid].append(int(idx))

        sizes = [len(b) for b in idx_batch]
        if min(sizes) < 1:
            splits, _ = split_dataset(dataset, num_clients, partition_mode="iid", seed=seed)
            return splits, [sorted(list(s)) for s in client_allowed]

        return [Subset(dataset, b) for b in idx_batch], [sorted(list(s)) for s in client_allowed]

    raise ValueError(f"Unknown partition_mode: {partition_mode}")
