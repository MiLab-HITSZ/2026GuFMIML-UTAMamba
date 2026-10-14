
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from utils import evaluate_metrics
from model import AsymmetricLoss

class Server:
    def __init__(self, global_model, test_dataset, config):
        self.global_model = global_model.to(config.DEVICE)
        self.device = config.DEVICE
        self.cfg = config
        self.test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4) if test_dataset else None
        
    def aggregate_parameters(self, client_weights):
        print("\n🔵 [Server] Aggregating parameters...")
        global_dict = self.global_model.state_dict()
        
        # Zero out
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key]).float()
            
        # Sum
        for w in client_weights:
            for key in global_dict.keys():
                global_dict[key] += w[key]
                
        # Average
        for key in global_dict.keys():
            global_dict[key] = global_dict[key] / len(client_weights)
            
        self.global_model.load_state_dict(global_dict)
        self.validate_global_model()
        return self.global_model.state_dict()

    def distribute_parameters(self):
        return copy.deepcopy(self.global_model.state_dict())
    
    def validate_global_model(self):
        if not self.test_loader:
            return
            
        self.global_model.eval()
        # Use BCEWithLogitsLoss for consistency
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(self.device)
        total_loss = 0.0
        all_targets, all_scores = [], []
        
        print("🔍 [Validation] Evaluating global model...")
        with torch.no_grad():
            for imgs, targets in self.test_loader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                logits = self.global_model(imgs)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                all_targets.append(targets.cpu().numpy())
                all_scores.append(logits.detach().cpu().numpy())
                
        all_targets = np.concatenate(all_targets, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        
        metrics = evaluate_metrics(all_targets, all_scores)
        metrics["Loss"] = total_loss / len(self.test_loader)
        
        print(f"📈 [Server Validation]")
        print(f"   Loss: {metrics['Loss']:.4f} | mAP: {metrics['mAP']:.4f}")
        print(f"   mAP_per_sample: {metrics['mAP_per_sample']:.4f}")
        print(f"   Hamming: {metrics['HammingLoss']:.4f} | OneErr: {metrics['OneError']:.4f} | Cov: {metrics['Coverage']:.4f}")
        print(f"   RankLoss: {metrics['RankingLoss']:.4f}")
        print(f"   Micro - Prec: {metrics['MicroPrecision']:.4f} | Rec: {metrics['MicroRecall']:.4f} | F1: {metrics['MicroF1']:.4f}")
        print(f"   Macro - Prec: {metrics['MacroPrecision']:.4f} | Rec: {metrics['MacroRecall']:.4f} | F1: {metrics['MacroF1']:.4f}")
