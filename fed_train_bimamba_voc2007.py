import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import sys
import datetime
import argparse
from sklearn.metrics import (
    average_precision_score,
    hamming_loss,
    coverage_error,
    label_ranking_loss,
    precision_score,
    recall_score,
    f1_score,
    zero_one_loss
)
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision import transforms
import copy
import os
import numpy as np
import json
from utils import setup_seed, split_dataset
import config as cfg

# 引入基础组件
import train_mamba_voc2
import train_mamba_voc2_resnet7x7
from train_mamba_voc2_resnet7x7 import (
    VOCMambaModel,
    VOCMultiLabelDataset,
    VOC_CLASSES,
    AsymmetricLoss,
    GraphInformedMambaDecoder
)

# 尝试引入 Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("⚠️ mamba_ssm not found, using LSTM fallback for BiMamba")

# ==========================================
# 0. Bi-Mamba Block
# ==========================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        if HAS_MAMBA:
            # Forward direction Mamba
            self.forward_core = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            # Backward direction Mamba
            self.backward_core = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            # Fallback to Bi-LSTM
            self.core = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True)
            self.proj = nn.Linear(d_model * 2, d_model)
        
        # Fusion layer to combine forward and backward features
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        if HAS_MAMBA:
            # 1. Forward Pass
            x_fwd = self.forward_core(x)
            
            # 2. Backward Pass (flip -> mamba -> flip)
            x_rev = torch.flip(x, dims=[1])
            x_bwd = self.backward_core(x_rev)
            x_bwd = torch.flip(x_bwd, dims=[1])
            
            # 3. Concatenate & Fuse
            x_combined = torch.cat([x_fwd, x_bwd], dim=-1) # (B, L, 2*D)
            x_out = self.fusion(x_combined)
        else:
            # LSTM Fallback
            output, _ = self.core(x) # (B, L, 2*D)
            x_out = self.proj(output)
            
        return x_out # Residual connection is handled in the Decoder loop

# ==========================================
# 1. AlignMamba 核心组件: OT Alignment
# ==========================================

class OTAlignment(nn.Module):
    """
    [AlignMamba Component 1] 基于最优传输 (Optimal Transport) 的局部对齐
    """
    def __init__(self, d_model):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, image_features, label_embeddings):
        # image_features: (B, N_img, D)
        # label_embeddings: (B, N_lbl, D)
        
        img_norm = F.normalize(image_features, p=2, dim=-1)
        lbl_norm = F.normalize(label_embeddings, p=2, dim=-1)
        
        sim_matrix = torch.bmm(lbl_norm, img_norm.transpose(1, 2))
        cost_matrix = 1 - sim_matrix
        
        transport_plan = F.softmax(-cost_matrix / self.temperature, dim=-1)
        aligned_feat = torch.bmm(transport_plan, image_features)
        
        return aligned_feat

# ==========================================
# 2. 辅助模块 (GCN, SE, etc.)
# ==========================================

class DynamicAdjLayer(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.phi = nn.Linear(d_model, d_model // 4)
        self.theta = nn.Linear(d_model, d_model // 4)
        self.soft = nn.Softmax(dim=1)
    def forward(self, label_embeddings):
        phi_out = self.phi(label_embeddings)
        theta_out = self.theta(label_embeddings)
        adj = torch.matmul(phi_out, theta_out.transpose(1, 2))
        return self.soft(adj)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, n, c = x.size()
        y = x.transpose(1, 2)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y.expand_as(x)

def load_glove_embeddings(class_names, d_model, glove_path=None):
    if glove_path is None or not os.path.exists(glove_path):
        return nn.Parameter(torch.randn(1, len(class_names), d_model))
    embeddings_dict = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split(); word = values[0]
                try: embeddings_dict[word] = np.asarray(values[1:], "float32")
                except: continue
    except: return nn.Parameter(torch.randn(1, len(class_names), d_model))
    if not embeddings_dict: return nn.Parameter(torch.randn(1, len(class_names), d_model))
    glove_dim = len(next(iter(embeddings_dict.values())))
    voc_glove_map = {'aeroplane': 'airplane', 'pottedplant': 'plant', 'tvmonitor': 'monitor', 'diningtable': 'table'}
    matrix = []
    for cls in class_names:
        query_word = voc_glove_map.get(cls, cls)
        vec = embeddings_dict.get(query_word, np.random.normal(scale=0.6, size=(glove_dim,)))
        matrix.append(vec)
    matrix = torch.tensor(np.array(matrix)).float()
    if glove_dim != d_model:
        with torch.no_grad(): matrix = nn.Linear(glove_dim, d_model)(matrix)
    return nn.Parameter(matrix.unsqueeze(0))

train_mamba_voc2.load_glove_embeddings = load_glove_embeddings

# ==========================================
# 3. AlignBiMamba Decoder (双向 Mamba)
# ==========================================

class AlignBiMambaDecoder(GraphInformedMambaDecoder):
    def __init__(self, d_model, num_classes, num_layers=2, dropout=0.1):
        super().__init__(d_model, num_classes, num_layers, dropout)
        
        # 1. 动态图
        self.dynamic_adj = DynamicAdjLayer(d_model, num_classes)
        
        # 2. OT Alignment
        self.ot_align = OTAlignment(d_model)
        
        # 3. Fusion Layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # 4. SE Attention
        self.se_block = SEBlock(d_model)
        
        # 5. 双向 Mamba Layers (覆盖父类的 mamba_layers)
        self.mamba_layers = nn.ModuleList([
            BiMambaBlock(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, image_features, label_embeddings, _=None):
        # image_features: (B, 196, D)
        # label_embeddings: (B, 20, D)
        
        # --- Step 1: Label 自身增强 (GCN) ---
        adj = self.dynamic_adj(label_embeddings)
        label_ctx = self.gcn(label_embeddings, adj)
        enhanced_labels = label_embeddings + label_ctx
        
        # --- Step 2: 显式 OT 对齐 ---
        aligned_img_feat = self.ot_align(image_features, enhanced_labels)
        
        # --- Step 3: 融合与序列构建 ---
        fused_labels = self.fusion(torch.cat([enhanced_labels, aligned_img_feat], dim=-1))
        fused_labels = self.norm(fused_labels)
        
        # 构建序列: [Labels, Images]
        # 在双向 Mamba 中，Labels 位于前面：
        # Forward pass: Labels 看不到 Images (只能看到之前的 Labels)
        # Backward pass: Labels 可以看到 Images (因为反向时 Images 在 Labels 之前)
        # 从而实现 Labels 对 Images 的全局感知
        full_sequence = torch.cat([fused_labels, image_features], dim=1)
        full_sequence = self.dropout(full_sequence)

        # --- Step 4: Mamba 扫描 (双向) ---
        x = full_sequence
        for layer in self.mamba_layers:
            x = layer(x) + x # Residual connection
        x = self.norm(x)
        x = self.se_block(x)

        # --- Step 5: 分类 ---
        final_label_features = x[:, :20, :] # 取前 20 个 token (Labels)
        
        logits = (final_label_features * enhanced_labels).sum(dim=-1)
        return logits

# Monkey Patch
train_mamba_voc2.GraphInformedMambaDecoder = AlignBiMambaDecoder
train_mamba_voc2_resnet7x7.GraphInformedMambaDecoder = AlignBiMambaDecoder
train_mamba_voc2_resnet7x7.load_glove_embeddings = load_glove_embeddings
print("✅ 已升级 Decoder: AlignBiMamba (Bi-Directional Mamba + OT + GCN)")

# ==========================================
# 4. 联邦学习流程 (保持不变)
# ==========================================

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

    return {
        "mAP": mAP,
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

def evaluate_model(model, dataloader, device):
    model.eval()
    # Use BCEWithLogitsLoss for consistency
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)
    total_loss = 0.0
    all_targets, all_scores = [], []
    
    print("🔍 [Validation] Evaluating model...")
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs, None)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_scores.append(logits.detach().cpu().numpy())
            
    all_targets = np.concatenate(all_targets, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    metrics = evaluate_metrics(all_targets, all_scores)
    metrics["Loss"] = total_loss / len(dataloader)
    return metrics

class FedServer:
    def __init__(self, global_model, test_dataset, device):
        self.global_model = global_model.to(device)
        self.device = device
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) if test_dataset else None
        
    def aggregate_parameters(self, client_weights):
        print("\n🔵 [Server] Aggregating parameters...")
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys(): global_dict[key] = torch.zeros_like(global_dict[key]).float()
        for w in client_weights:
            for key in global_dict.keys(): global_dict[key] += w[key]
        for key in global_dict.keys(): global_dict[key] = global_dict[key] / len(client_weights)
        self.global_model.load_state_dict(global_dict)
        self.validate_global_model()
        return self.global_model.state_dict()

    def distribute_parameters(self): return copy.deepcopy(self.global_model.state_dict())
    
    def validate_global_model(self):
        if not self.test_loader: return
        metrics = evaluate_model(self.global_model, self.test_loader, self.device)
        print("📈 [Server Validation]")
        print(f"   Loss: {metrics['Loss']:.4f} | mAP: {metrics['mAP']:.4f}")
        print(f"   Hamming: {metrics['HammingLoss']:.4f} | OneErr: {metrics['OneError']:.4f} | Cov: {metrics['Coverage']:.4f}")
        print(f"   RankLoss: {metrics['RankingLoss']:.4f}")
        print(f"   Micro - Prec: {metrics['MicroPrecision']:.4f} | Rec: {metrics['MicroRecall']:.4f} | F1: {metrics['MicroF1']:.4f}")
        print(f"   Macro - Prec: {metrics['MacroPrecision']:.4f} | Rec: {metrics['MacroRecall']:.4f} | F1: {metrics['MacroF1']:.4f}")

class FedClient:
    def __init__(self, client_id, dataset, model_cls, model_args, device, allowed_labels=None):
        self.client_id = client_id
        self.device = device
        self.loader = DataLoader(dataset, batch_size=8, shuffle=True)
        self.model = model_cls(**model_args).to(device)
        self.allowed_labels = allowed_labels
        # Use BCEWithLogitsLoss for better Hamming Loss optimization (matching FMIML-TAG)
        # AsymmetricLoss is better for mAP but can be conservative on Recall.
        self.criterion_cls = torch.nn.BCEWithLogitsLoss().to(self.device)
        # self.criterion_cls = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(device)

    def _compute_loss(self, logits, targets):
        return self.criterion_cls(logits, targets)
        
    def local_train(self, global_params, epochs=1, current_round=0):
        print(f"🟢 [Client {self.client_id}] Starting AlignBiMamba training...")
        self.model.load_state_dict(global_params)
        self.model.train()
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4) 
        
        total_loss = 0.0
        steps = 0
        for epoch in range(epochs):
            for imgs, targets in self.loader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                logits = self.model(imgs, None)
                loss = self._compute_loss(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                # Reduce clip_grad_norm from 5.0 to 1.0 to prevent exploding gradients in Mamba
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                
        print(f"   [Client {self.client_id}] Done. Avg Loss: {total_loss / max(1, steps):.4f}")
        return self.model.state_dict()

def partition_data(dataset, num_clients):
    indices = torch.randperm(len(dataset)).tolist()
    split_size = len(dataset) // num_clients
    return [Subset(dataset, indices[i*split_size : (i+1)*split_size]) for i in range(num_clients)]

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

def run_federated_training():
    parser = argparse.ArgumentParser(description='Federated AlignBiMamba Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--partition_mode', type=str, default=getattr(cfg, 'PARTITION_MODE', 'iid'), choices=['iid', 'dirichlet', 'label'])
    parser.add_argument('--dirichlet_alpha', type=float, default=getattr(cfg, 'DIRICHLET_ALPHA', 0.5))
    parser.add_argument('--labels_per_client', type=int, default=getattr(cfg, 'LABELS_PER_CLIENT', 2))
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _mode_tag = args.partition_mode
    if _mode_tag == 'dirichlet':
        _mode_tag = f"dirichlet_alpha{args.dirichlet_alpha}"
    elif _mode_tag == 'label':
        _mode_tag = f"label_labels{args.labels_per_client}"
    log_filename = os.path.join(cfg.LOG_DIR, f"voc2007_{_mode_tag}_{_ts}.txt")
    sys.stdout = Logger(log_filename)
    
    NUM_CLIENTS = getattr(cfg, 'NUM_CLIENTS', 5)
    ROUNDS = getattr(cfg, 'ROUNDS', 50)
    LOCAL_EPOCHS = getattr(cfg, 'EPOCHS', 5)
    
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        print(f"✅ Using GPU: {torch.cuda.get_device_name(args.gpu)} (ID: {args.gpu})")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ CUDA not available, using CPU")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, 'data')
    DATA_ROOT2 = "/home/gjh/datasets"
    GLOVE_PATH = os.path.join(DATA_ROOT, 'glove.6B.300d.txt')
    ds_name = 'voc'
    print(f"🚀 启动联邦 AlignBiMamba 训练 (ResNet101 + 448px + OT + BiMamba)")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        # ds_train_full = VOCMultiLabelDataset(root=DATA_ROOT, year='2007', image_set='trainval', transform=train_transform)
        # ds_val_full = VOCMultiLabelDataset(root=DATA_ROOT, year='2007', image_set='trainval', transform=val_transform)
        # total_size = len(ds_train_full)
        # train_size = int(0.9 * total_size)
        # generator = torch.Generator().manual_seed(42)
        # indices = torch.randperm(total_size, generator=generator).tolist()
        # train_dataset = Subset(ds_train_full, indices[:train_size])
        # val_dataset = Subset(ds_val_full, indices[train_size:])
        if ds_name == 'scene':
            print("Loading Scene dataset...")
            image_dir = os.path.join('data', 'scene', 'original')
            label_path = os.path.join('data', 'scene', 'miml data.mat')
            full_dataset = SceneDataset(image_dir, label_path)
            NUM_CLASSES = full_dataset.num_classes
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
        # 使用 generator 确保随机分割可复现
            generator = torch.Generator().manual_seed(args.seed)
            train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)
            val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
            client_sets, _ = split_dataset(train_set, cfg.NUM_CLIENTS, partition_mode=args.partition_mode, dirichlet_alpha=args.dirichlet_alpha, labels_per_client=args.labels_per_client, seed=args.seed)
        elif ds_name == 'msrc':
            print("Loading MSRC dataset...")
            base_path = os.path.join('data', 'MSRCv2', 'msrc2_seg')
            with open(os.path.join(base_path, 'color_to_class.json')) as f:
                num_classes = len(json.load(f))
            cfg.NUM_CLASSES = num_classes
            train_set = MSRCDataset(base_path, 'Train_labels.txt', num_classes)
            val_set = MSRCDataset(base_path, 'Validation_labels.txt', num_classes)
            test_set = MSRCDataset(base_path, 'Test_labels.txt', num_classes)
            trainval_set = ConcatDataset([train_set, val_set])
            val_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            client_sets, _ = split_dataset(trainval_set, cfg.NUM_CLIENTS, partition_mode=args.partition_mode, dirichlet_alpha=args.dirichlet_alpha, labels_per_client=args.labels_per_client, seed=args.seed)
        train_dataset = VOCMultiLabelDataset(root=DATA_ROOT2, year='2012', image_set='train', transform=train_transform)
        val_dataset = VOCMultiLabelDataset(root=DATA_ROOT2, year='2012', image_set='val', transform=val_transform)
    except:
        print("⚠️ 没找到数据，使用 Mock")
        class MockDS(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx): return torch.randn(3, 448, 448), torch.zeros(20)
        train_dataset = MockDS(); val_dataset = MockDS()

    client_datasets, client_allowed_labels = split_dataset(
        train_dataset,
        NUM_CLIENTS,
        partition_mode=args.partition_mode,
        dirichlet_alpha=args.dirichlet_alpha,
        labels_per_client=args.labels_per_client,
        seed=args.seed,
    )
    model_args = {'num_classes': 20, 'd_model': 256, 'glove_path': GLOVE_PATH, 'freeze_backbone': False}
    global_model = VOCMambaModel(**model_args)
    server = FedServer(global_model, val_dataset, DEVICE)
    clients = [FedClient(i, ds, VOCMambaModel, model_args, DEVICE, allowed_labels=client_allowed_labels[i]) for i, ds in enumerate(client_datasets)]
    
    for round_idx in range(ROUNDS):
        print(f"\n================ Round {round_idx+1}/{ROUNDS} ================")
        global_params = server.distribute_parameters()
        client_weights = [client.local_train(global_params, epochs=LOCAL_EPOCHS, current_round=round_idx) for client in clients]
        server.aggregate_parameters(client_weights)
        
    torch.save(server.global_model.state_dict(), "fed_align_bimamba_voc.pth")
    print("💾 训练完成。")

if __name__ == "__main__":
    run_federated_training()
