import os
import sys
import math
import argparse
import datetime
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import average_precision_score, coverage_error, label_ranking_loss

import config as cfg
from dataset import VOCDataset, SceneDataset, MSRCDataset, get_transforms
from model import BiMambaModel, AsymmetricLoss, compute_mass_bce, compute_ot_consistency_reg
from utils import setup_seed, Logger


def get_dataset(dataset_name: str):
    train_transform = get_transforms(split='train')
    val_transform = get_transforms(split='val')

    name = dataset_name.lower()
    if name == 'voc2007':
        print('📂 Loading VOC2007...')
        ds_train = VOCDataset(root=cfg.DATA_ROOT, year='2007', image_set='train', transform=train_transform, download=False)
        ds_val = VOCDataset(root=cfg.DATA_ROOT, year='2007', image_set='val', transform=val_transform, download=False)
        return ds_train, ds_val

    if name == 'voc2012':
        print('📂 Loading VOC2012...')
        ds_train = VOCDataset(root=cfg.DATA_ROOT, year='2012', image_set='train', transform=train_transform, download=False)
        ds_val = VOCDataset(root=cfg.DATA_ROOT, year='2012', image_set='val', transform=val_transform, download=False)
        return ds_train, ds_val

    if name == 'scene':
        print('📂 Loading Scene Dataset...')
        full_ds = SceneDataset(image_dir=cfg.SCENE_IMAGE_DIR, label_path=cfg.SCENE_LABEL_PATH, transform=train_transform)
        total_size = len(full_ds)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        generator = torch.Generator().manual_seed(cfg.SEED)
        ds_train, ds_val = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)
        return ds_train, ds_val

    if name == 'msrc':
        print('📂 Loading MSRC Dataset...')
        ds_train = MSRCDataset(root=cfg.MSRC_ROOT, label_file='Train.txt', num_classes=23, transform=train_transform)
        ds_val = MSRCDataset(root=cfg.MSRC_ROOT, label_file='Validation.txt', num_classes=23, transform=val_transform)
        ds_test = MSRCDataset(root=cfg.MSRC_ROOT, label_file='Test.txt', num_classes=23, transform=val_transform)
        ds_train = ConcatDataset([ds_train, ds_val])
        ds_val = ds_test
        if len(ds_train) == 0:
            print('⚠️ MSRC specific files not found, trying generic loading...')
        return ds_train, ds_val

    raise ValueError(f'Unknown dataset: {dataset_name}')


def unwrap_dataset(ds):
    if isinstance(ds, ConcatDataset):
        return unwrap_dataset(ds.datasets[0])
    if isinstance(ds, Subset):
        return unwrap_dataset(ds.dataset)
    return ds


def get_class_names(train_dataset) -> List[str]:
    sample_ds = unwrap_dataset(train_dataset)
    if hasattr(sample_ds, 'class_names'):
        return list(sample_ds.class_names)
    if hasattr(sample_ds, 'classes'):
        return list(sample_ds.classes)
    return [f'class_{i}' for i in range(cfg.NUM_CLASSES)]


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def compute_negative_assignment_loss(model, targets):
    if not hasattr(model, 'decoder') or model.decoder is None:
        return None
    transport_plan = getattr(model.decoder, 'last_transport_plan', None)
    if transport_plan is None:
        return None
    neg_mask = 1.0 - targets.float()
    neg_peak = transport_plan.max(dim=-1).values
    loss_neg = (neg_peak * neg_mask).sum() / (neg_mask.sum() + 1e-6)
    return loss_neg


def compute_multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    eps = 1e-12
    y_pred = (y_score >= threshold).astype(np.float32)

    ap_list = []
    for c in range(y_true.shape[1]):
        if np.sum(y_true[:, c]) > 0:
            ap_list.append(average_precision_score(y_true[:, c], y_score[:, c]))
    mAP = float(np.mean(ap_list)) if ap_list else 0.0

    sample_ap_list = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            try:
                sample_ap_list.append(average_precision_score(y_true[i], y_score[i]))
            except ValueError:
                pass
    mAP_per_sample = float(np.mean(sample_ap_list)) if sample_ap_list else 0.0

    hamming = float(np.mean(np.not_equal(y_pred, y_true).astype(np.float32)))

    top_idx = np.argmax(y_score, axis=1)
    one_error = float(np.mean(1.0 - y_true[np.arange(y_true.shape[0]), top_idx]))

    try:
        coverage = float(coverage_error(y_true, y_score) - 1.0)
    except ValueError:
        coverage = float('nan')
    try:
        ranking = float(label_ranking_loss(y_true, y_score))
    except ValueError:
        ranking = float('nan')

    tp = float((y_pred * y_true).sum())
    fp = float((y_pred * (1.0 - y_true)).sum())
    fn = float(((1.0 - y_pred) * y_true).sum())
    micro_precision = tp / (tp + fp + eps)
    micro_recall = tp / (tp + fn + eps)
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + eps)

    macro_prec_list = []
    macro_rec_list = []
    macro_f1_list = []
    for c in range(y_true.shape[1]):
        tp_c = float(np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 1)))
        fp_c = float(np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 0)))
        fn_c = float(np.sum((y_pred[:, c] == 0) & (y_true[:, c] == 1)))
        prec_c = tp_c / (tp_c + fp_c + eps)
        rec_c = tp_c / (tp_c + fn_c + eps)
        f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c + eps)
        macro_prec_list.append(prec_c)
        macro_rec_list.append(rec_c)
        macro_f1_list.append(f1_c)

    macro_precision = float(np.mean(macro_prec_list))
    macro_recall = float(np.mean(macro_rec_list))
    macro_f1 = float(np.mean(macro_f1_list))

    return {
        'mAP': mAP,
        'mAP_per_sample': mAP_per_sample,
        'HammingLoss': hamming,
        'OneError': one_error,
        'Coverage': coverage,
        'RankingLoss': ranking,
        'MicroPrecision': micro_precision,
        'MicroRecall': micro_recall,
        'MicroF1': micro_f1,
        'MacroPrecision': macro_precision,
        'MacroRecall': macro_recall,
        'MacroF1': macro_f1,
    }


@torch.no_grad()
def validate_model(model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_scores = []

    print('🔍 [Validation] Evaluating centralized model...')
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = model(imgs)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        all_targets.append(targets.cpu().numpy())
        all_scores.append(logits.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    metrics = compute_multilabel_metrics(all_targets, all_scores)
    metrics['Loss'] = total_loss / max(len(loader), 1)
    return metrics


def print_metrics_like_server(metrics: Dict[str, float]) -> None:
    print('📈 [Centralized Validation]')
    print(f"   Loss: {metrics['Loss']:.4f} | mAP: {metrics['mAP']:.4f}")
    print(f"   mAP_per_sample: {metrics['mAP_per_sample']:.4f}")
    print(f"   Hamming: {metrics['HammingLoss']:.4f} | OneErr: {metrics['OneError']:.4f} | Cov: {metrics['Coverage']:.4f}")
    print(f"   RankLoss: {metrics['RankingLoss']:.4f}")
    print(f"   Micro - Prec: {metrics['MicroPrecision']:.4f} | Rec: {metrics['MicroRecall']:.4f} | F1: {metrics['MicroF1']:.4f}")
    print(f"   Macro - Prec: {metrics['MacroPrecision']:.4f} | Rec: {metrics['MacroRecall']:.4f} | F1: {metrics['MacroF1']:.4f}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, use_decoder: bool, use_ot: bool, lambda_neg: float, lambda_mass: float, lambda_cons: float, print_freq: int = 20):
    model.train()

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_neg_loss = 0.0
    total_mass_loss = 0.0
    total_cons_loss = 0.0
    neg_steps = 0
    mass_steps = 0
    cons_steps = 0
    steps = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = model(images)
        loss_cls = criterion(logits, targets)
        loss = loss_cls
        loss_neg = None
        loss_mass = None
        loss_cons = None

        if use_decoder and use_ot:
            loss_neg = compute_negative_assignment_loss(model, targets)
            if loss_neg is not None:
                loss = loss + lambda_neg * loss_neg

            if lambda_mass > 0.0:
                loss_mass = compute_mass_bce(model, targets)
                if loss_mass is not None:
                    loss = loss + lambda_mass * loss_mass

            if lambda_cons > 0.0:
                loss_cons = compute_ot_consistency_reg(model, targets)
                if loss_cons is not None:
                    loss = loss + lambda_cons * loss_cons

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        if loss_neg is not None:
            total_neg_loss += loss_neg.item()
            neg_steps += 1
        if loss_mass is not None:
            total_mass_loss += loss_mass.item()
            mass_steps += 1
        if loss_cons is not None:
            total_cons_loss += loss_cons.item()
            cons_steps += 1
        steps += 1

        if step % print_freq == 0:
            parts = [
                f'Epoch {epoch:03d}',
                f'Step {step:04d}/{len(loader):04d}',
                f'Loss {loss.item():.4f}',
                f'Cls {loss_cls.item():.4f}',
            ]
            if loss_neg is not None:
                parts.append(f'Neg {loss_neg.item():.4f}')
            if loss_mass is not None:
                parts.append(f'Mass {loss_mass.item():.4f}')
            if loss_cons is not None:
                parts.append(f'Cons {loss_cons.item():.4f}')
            print(' | '.join(parts))

    avg = {'loss': total_loss / max(steps, 1), 'cls_loss': total_cls_loss / max(steps, 1)}
    if neg_steps > 0:
        avg['neg_loss'] = total_neg_loss / neg_steps
    if mass_steps > 0:
        avg['mass_loss'] = total_mass_loss / mass_steps
    if cons_steps > 0:
        avg['cons_loss'] = total_cons_loss / cons_steps
    return avg


def main():
    parser = argparse.ArgumentParser(description='Centralized Training with BiMamba')
    parser.add_argument('--dataset', type=str, default=cfg.DATASET, help='dataset name')
    parser.add_argument('--epochs', type=int, default=getattr(cfg, 'EPOCHS', 5), help='number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--seed', type=int, default=cfg.SEED, help='random seed')
    parser.add_argument('--batch_size', type=int, default=getattr(cfg, 'BATCH_SIZE', 16))
    parser.add_argument('--lr', type=float, default=getattr(cfg, 'LR', 1e-4))
    parser.add_argument('--weight_decay', type=float, default=getattr(cfg, 'WEIGHT_DECAY', 1e-4))
    parser.add_argument('--num_workers', type=int, default=getattr(cfg, 'NUM_WORKERS', 4))
    parser.add_argument('--use_gcn', type=int, default=1 if getattr(cfg, 'USE_GCN', True) else 0)
    parser.add_argument('--use_decoder', type=int, default=1 if getattr(cfg, 'USE_DECODER', True) else 0)
    parser.add_argument('--use_ot', type=int, default=1 if getattr(cfg, 'USE_OT', True) else 0)
    parser.add_argument('--lambda_neg', type=float, default=getattr(cfg, 'LAMBDA_NEG', 0.05))
    parser.add_argument('--lambda_mass', type=float, default=getattr(cfg, 'LAMBDA_MASS', 0.05),
                        help='Weight for direct BCE supervision on the OT mass head (0 disables).')
    parser.add_argument('--lambda_cons', type=float, default=getattr(cfg, 'LAMBDA_CONSISTENCY', 1e-3),
                        help='Weight for KL(T_f || T_b) consistency on present labels (0 disables).')
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'asl'])
    args = parser.parse_args()

    if torch.cuda.is_available():
        cfg.DEVICE = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"✅ Using GPU: {torch.cuda.get_device_name(args.gpu)} (ID: {args.gpu})")
    else:
        cfg.DEVICE = torch.device('cpu')
        print('⚠️ CUDA not available, using CPU')

    cfg.DATASET = args.dataset
    cfg.EPOCHS = args.epochs
    cfg.SEED = args.seed
    cfg.BATCH_SIZE = args.batch_size
    cfg.LR = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.NUM_WORKERS = args.num_workers
    cfg.USE_GCN = bool(int(args.use_gcn))
    cfg.USE_DECODER = bool(int(args.use_decoder))
    cfg.USE_OT = bool(int(args.use_ot))
    cfg.LAMBDA_NEG = args.lambda_neg
    cfg.LAMBDA_MASS = args.lambda_mass
    cfg.LAMBDA_CONSISTENCY = args.lambda_cons

    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _arch_tag = ('gcn' if cfg.USE_GCN else 'nogcn') + '_' + ('dec' if cfg.USE_DECODER else 'nodec') + '_' + ('ot' if cfg.USE_OT else 'noot')
    log_filename = os.path.join(cfg.LOG_DIR, f'centralized_{cfg.DATASET}_{_arch_tag}_{_ts}.txt')
    sys.stdout = Logger(log_filename)

    setup_seed(cfg.SEED)
    print(f'🚀 Starting Centralized Training on {cfg.DATASET}')

    train_dataset, val_dataset = get_dataset(cfg.DATASET)
    class_names = get_class_names(train_dataset)
    cfg.NUM_CLASSES = len(class_names)
    print(f'✅ Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Classes: {cfg.NUM_CLASSES}')

    train_loader = make_loader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = make_loader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = BiMambaModel(
        num_classes=cfg.NUM_CLASSES,
        class_names=class_names,
        d_model=cfg.D_MODEL,
        glove_path=cfg.GLOVE_PATH,
        freeze_backbone=cfg.FREEZE_BACKBONE,
        use_gcn=cfg.USE_GCN,
        use_decoder=cfg.USE_DECODER,
        use_ot=cfg.USE_OT,
    ).to(cfg.DEVICE)

    if args.loss_type == 'asl':
        criterion = AsymmetricLoss(
            gamma_neg=getattr(cfg, 'ASL_GAMMA_NEG', 4),
            gamma_pos=getattr(cfg, 'ASL_GAMMA_POS', 1),
            clip=getattr(cfg, 'ASL_CLIP', 0.05),
        ).to(cfg.DEVICE)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_map = -math.inf
    best_metrics: Dict[str, float] = {}
    save_dir = cfg.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f'centralized_bimamba_{cfg.DATASET}_{_arch_tag}_best.pth')
    last_path = os.path.join(save_dir, f'centralized_bimamba_{cfg.DATASET}_{_arch_tag}_last.pth')

    for epoch_idx in range(cfg.EPOCHS):
        print(f'\n================ Epoch {epoch_idx+1}/{cfg.EPOCHS} ================')
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            cfg.DEVICE,
            epoch_idx + 1,
            use_decoder=cfg.USE_DECODER,
            use_ot=cfg.USE_OT,
            lambda_neg=cfg.LAMBDA_NEG,
            lambda_mass=cfg.LAMBDA_MASS,
            lambda_cons=cfg.LAMBDA_CONSISTENCY,
        )

        print(f"🟢 [Centralized] Train Done. Avg Loss: {train_stats['loss']:.4f} | Avg Cls Loss: {train_stats['cls_loss']:.4f}" +
              (f" | Avg Neg Loss: {train_stats['neg_loss']:.4f}" if 'neg_loss' in train_stats else '') +
              (f" | Avg Mass Loss: {train_stats['mass_loss']:.4f}" if 'mass_loss' in train_stats else '') +
              (f" | Avg Cons Loss: {train_stats['cons_loss']:.4f}" if 'cons_loss' in train_stats else ''))

        val_metrics = validate_model(model, val_loader, criterion, cfg.DEVICE)
        print_metrics_like_server(val_metrics)

        if val_metrics['mAP'] > best_map:
            best_map = val_metrics['mAP']
            best_metrics = val_metrics.copy()
            torch.save(model.state_dict(), best_path)
            print(f'💾 New best model saved to {best_path}')

    torch.save(model.state_dict(), last_path)
    print(f'💾 Training Completed. Model saved to {last_path}')
    if best_metrics:
        print('🏆 Best Metrics')
        print_metrics_like_server(best_metrics)


if __name__ == '__main__':
    main()
