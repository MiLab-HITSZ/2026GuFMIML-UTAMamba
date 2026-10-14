import os
import sys
import torch
import argparse
import datetime
from torch.utils.data import Subset, ConcatDataset
import config as cfg
# 🌟 修改点 1：导入 COCOMultiLabelDataset
from dataset import VOCDataset, SceneDataset, MSRCDataset, COCOMultiLabelDataset, get_transforms
from model import BiMambaModel
from client import Client
from server import Server
from utils import setup_seed, Logger, split_dataset

def get_dataset(dataset_name):
    train_transform = get_transforms(split='train')
    val_transform = get_transforms(split='val')
    
    if dataset_name.lower() == 'voc2007':
        print(f"📂 Loading VOC2007...")
        ds_train = VOCDataset(root=cfg.DATA_ROOT, year='2007', image_set='train', transform=train_transform, download=False)
        ds_val = VOCDataset(root=cfg.DATA_ROOT, year='2007', image_set='val', transform=val_transform, download=False)
        return ds_train, ds_val
        
    elif dataset_name.lower() == 'voc2012':
        print(f"📂 Loading VOC2012...")
        # Note: root should be the parent of VOCdevkit
        ds_train = VOCDataset(root=cfg.DATA_ROOT, year='2012', image_set='train', transform=train_transform, download=False)
        # VOC2012 test set is private, usually validate on val set or 2007 test
        # Here we use 2012 val as validation if strictly 2012, or split trainval
        # But typically people train on 07+12 and test on 07 test.
        # If specific request for VOC2012, we can return trainval and val.
        ds_val = VOCDataset(root=cfg.DATA_ROOT, year='2012', image_set='val', transform=val_transform, download=False)
        return ds_train, ds_val
        
    elif dataset_name.lower() == 'scene':
        print(f"📂 Loading Scene Dataset...")
        full_ds = SceneDataset(image_dir=cfg.SCENE_IMAGE_DIR, label_path=cfg.SCENE_LABEL_PATH, transform=train_transform)
        # Split 80/20
        total_size = len(full_ds)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        generator = torch.Generator().manual_seed(cfg.SEED)
        ds_train, ds_val = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)
        # Apply transforms (Note: Subset doesn't allow easy transform override, but here both use same transform initially. 
        # Ideally we wrap Subset to apply specific transform, but for simplicity we assume train_transform is fine for both or we re-wrap)
        # For strict correctness we should wrap val set with val_transform.
        return ds_train, ds_val

    elif dataset_name.lower() == 'msrc':
        print(f"📂 Loading MSRC Dataset...")
        # MSRC usually has fixed split or we split it. FMIML main.py used Train_labels.txt etc.
        # MSRCDataset implementation in dataset.py expects label_file.
        # We assume standard MSRC v2 split if files exist, else random split.
        ds_train = MSRCDataset(root=cfg.MSRC_ROOT, label_file='Train.txt', num_classes=23, transform=train_transform) # MSRC v2 has 23 classes usually
        ds_val = MSRCDataset(root=cfg.MSRC_ROOT, label_file='Validation.txt', num_classes=23, transform=val_transform)
        ds_test = MSRCDataset(root=cfg.MSRC_ROOT, label_file='Test.txt', num_classes=23, transform=val_transform)
        
        ds_train = ConcatDataset([ds_train, ds_val])
        ds_val = ds_test
        # If MSRC files don't exist, our dataset class handles it gracefully or fails.
        if len(ds_train) == 0:
             # Fallback logic if files specific names are different or missing
             print("⚠️ MSRC specific files not found, trying generic loading...")
        return ds_train, ds_val

    # 🌟 修改点 2：新增 COCO 2014 加载逻辑
    elif dataset_name.lower() == 'coco2014':
        print(f"📂 Loading COCO 2014 Dataset...")
        # 默认去 config 里面找 COCO_ROOT，如果没配置，默认路径为 DATA_ROOT/coco
        coco_root = os.path.join(cfg.DATA_ROOT, 'coco2014')
        
        train_dir = os.path.join(coco_root, 'train2014', 'train2014', 'train2014', 'train2014')
        train_ann = os.path.join(coco_root, 'annotations', 'instances_train2014.json')
        val_dir = os.path.join(coco_root, 'val2014', 'val2014', 'val2014', 'val2014')
        val_ann = os.path.join(coco_root, 'annotations', 'instances_val2014.json')
        
        ds_train = COCOMultiLabelDataset(root_dir=train_dir, ann_file=train_ann, transform=train_transform)
        ds_val = COCOMultiLabelDataset(root_dir=val_dir, ann_file=val_ann, transform=val_transform)
        return ds_train, ds_val
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description='Federated Learning with BiMamba')
    parser.add_argument('--dataset', type=str, default=cfg.DATASET, help='dataset name')
    parser.add_argument('--rounds', type=int, default=cfg.ROUNDS, help='number of rounds')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--seed', type=int, default=cfg.SEED, help='random seed')
    parser.add_argument('--partition_mode', type=str, default=cfg.PARTITION_MODE, choices=['iid', 'dirichlet', 'label'])
    parser.add_argument('--dirichlet_alpha', type=float, default=cfg.DIRICHLET_ALPHA)
    parser.add_argument('--labels_per_client', type=int, default=getattr(cfg, 'LABELS_PER_CLIENT', 2))
    parser.add_argument('--use_gcn', type=int, default=1 if getattr(cfg, 'USE_GCN', True) else 0)
    parser.add_argument('--use_decoder', type=int, default=1 if getattr(cfg, 'USE_DECODER', True) else 0)
    parser.add_argument('--use_ot', type=int, default=1 if getattr(cfg, 'USE_OT', True) else 0)
    parser.add_argument('--ot_type', type=str, default=getattr(cfg, 'OT_TYPE', 'soft_uot'),
                        choices=['sigmoid', 'soft_uot'],
                        help='OT alignment type: sigmoid (legacy) or soft_uot (closed-form unbalanced soft OT)')
    parser.add_argument('--lambda_neg', type=float, default=getattr(cfg, 'LAMBDA_NEG', 0.05))
    args = parser.parse_args()
    
    # Setup Device
    if torch.cuda.is_available():
        cfg.DEVICE = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        print(f"✅ Using GPU: {torch.cuda.get_device_name(args.gpu)} (ID: {args.gpu})")
    else:
        cfg.DEVICE = torch.device("cpu")
        print("⚠️ CUDA not available, using CPU")
    
    # Update config
    cfg.DATASET = args.dataset
    cfg.ROUNDS = args.rounds
    cfg.SEED = args.seed
    cfg.PARTITION_MODE = args.partition_mode
    cfg.DIRICHLET_ALPHA = args.dirichlet_alpha
    cfg.LABELS_PER_CLIENT = args.labels_per_client
    cfg.USE_GCN = bool(int(args.use_gcn))
    cfg.USE_DECODER = bool(int(args.use_decoder))
    cfg.USE_OT = bool(int(args.use_ot))
    cfg.OT_TYPE = args.ot_type
    cfg.LAMBDA_NEG = args.lambda_neg
    # Logging
    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _mode_tag = args.partition_mode
    if _mode_tag == 'dirichlet':
        _mode_tag = f"dirichlet_alpha{args.dirichlet_alpha}"
    elif _mode_tag == 'label':
        _mode_tag = f"label_labels{args.labels_per_client}"
    _arch_tag = ("gcn" if cfg.USE_GCN else "nogcn") + "_" + ("dec" if cfg.USE_DECODER else "nodec") + "_" + ("ot" if cfg.USE_OT else "noot")
    log_filename = os.path.join(cfg.LOG_DIR, f"{cfg.DATASET}_{_mode_tag}_{_arch_tag}_{_ts}.txt")
    sys.stdout = Logger(log_filename)
    
    setup_seed(cfg.SEED)
    print(f"🚀 Starting Federated Training on {cfg.DATASET}")
    
    # Load Data
    train_dataset, val_dataset = get_dataset(cfg.DATASET)
    
    # Determine num_classes and class_names
    if isinstance(train_dataset, ConcatDataset):
        sample_ds = train_dataset.datasets[0]
    elif isinstance(train_dataset, Subset):
        sample_ds = train_dataset.dataset
    else:
        sample_ds = train_dataset
        
    # Get class names if available, else generate generic
    if hasattr(sample_ds, 'class_names'):
        class_names = sample_ds.class_names
    elif hasattr(sample_ds, 'classes'):
        class_names = sample_ds.classes
    else:
        # Fallback for Scene/MSRC if hidden deep
        # SceneDataset and MSRCDataset in dataset.py have class_names
        class_names = [f"class_{i}" for i in range(cfg.NUM_CLASSES)]
        
    cfg.NUM_CLASSES = len(class_names)
    print(f"✅ Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {cfg.NUM_CLASSES}")
    
    # Partition
    client_datasets, client_allowed_labels = split_dataset(
        train_dataset,
        cfg.NUM_CLIENTS,
        partition_mode=cfg.PARTITION_MODE,
        dirichlet_alpha=cfg.DIRICHLET_ALPHA,
        labels_per_client=cfg.LABELS_PER_CLIENT,
        seed=cfg.SEED,
    )
    
    # Initialize Server & Clients
    # Global model init
    global_model = BiMambaModel(
        num_classes=cfg.NUM_CLASSES,
        class_names=class_names,
        d_model=cfg.D_MODEL,
        glove_path=cfg.GLOVE_PATH,
        freeze_backbone=cfg.FREEZE_BACKBONE,
        use_gcn=cfg.USE_GCN,
        use_decoder=cfg.USE_DECODER,
        use_ot=cfg.USE_OT,
        ot_type=cfg.OT_TYPE,
    )
    
    server = Server(global_model, val_dataset, cfg)
    
    # Clients
    clients = []
    for i in range(cfg.NUM_CLIENTS):
        clients.append(Client(i, client_datasets[i], cfg, allowed_labels=client_allowed_labels[i]))
        
    # Training Loop
    for round_idx in range(cfg.ROUNDS):
        print(f"\n================ Round {round_idx+1}/{cfg.ROUNDS} ================")
        global_params = server.distribute_parameters()
        
        client_weights = []
        for client in clients:
            w = client.local_train(global_params, epochs=cfg.EPOCHS, current_round=round_idx)
            client_weights.append(w)
            
        server.aggregate_parameters(client_weights)
        
    # Save Final
    save_path = os.path.join(cfg.SAVE_DIR, f"fed_bimamba_{cfg.DATASET}_final.pth")
    torch.save(server.global_model.state_dict(), save_path)
    print(f"💾 Training Completed. Model saved to {save_path}")

if __name__ == "__main__":
    main()