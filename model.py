
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import numpy as np


def _inv_softplus(y: float) -> float:
    """Return x such that softplus(x) == y. Used to init parameters whose
    effective value passes through softplus, while preserving the original
    init-value semantics."""
    if y <= 0:
        raise ValueError("inv_softplus requires y > 0")
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


class _Entmax15Function(torch.autograd.Function):
    """Exact entmax-1.5 (Peters, Niculae & Martins, 2019) with custom backward.

    Like sparsemax it produces a true sparse distribution (some entries are
    exactly 0), but the truncation boundary is smooth (gradient is continuous),
    avoiding the hard-cutoff jitter that sparsemax can cause near the support
    boundary.

    Closed form (per row, sorted descending z, with cumulative S1_k = sum z, S2_k = sum z^2):
        tau_k = (S1_k - sqrt(S1_k^2 - k*(S2_k - 4))) / k
        k* = largest k s.t. z_sorted_k > tau_k
        p_i = max(0.5*(z_i - tau_{k*}), 0)^2

    Backward (let s = sqrt(p), zero off-support):
        grad_z = s * (grad_p - sum(s * grad_p) / sum(s))
    """

    @staticmethod
    def forward(ctx, z, dim):
        if dim != -1:
            z = z.transpose(dim, -1).contiguous()
        K = z.size(-1)
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        range_k = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
        cumsum1 = z_sorted.cumsum(dim=-1)
        cumsum2 = (z_sorted * z_sorted).cumsum(dim=-1)
        discriminant = cumsum1 * cumsum1 - range_k * (cumsum2 - 4.0)
        discriminant = discriminant.clamp(min=0.0)
        tau_candidates = (cumsum1 - torch.sqrt(discriminant)) / range_k
        cond = z_sorted > tau_candidates
        k_max = cond.to(z.dtype).sum(dim=-1).clamp(min=1.0)
        idx = (k_max - 1).long().unsqueeze(-1)
        tau = tau_candidates.gather(-1, idx).squeeze(-1)
        p = (0.5 * (z - tau.unsqueeze(-1))).clamp(min=0.0) ** 2
        ctx.save_for_backward(p)
        ctx.dim = dim
        if dim != -1:
            p = p.transpose(dim, -1)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        if dim != -1:
            grad_output = grad_output.transpose(dim, -1).contiguous()
        s = torch.sqrt(p.clamp(min=0.0))            # off-support: s=0
        sum_s = s.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weighted = (s * grad_output).sum(dim=-1, keepdim=True) / sum_s
        grad_z = s * (grad_output - weighted)
        if dim != -1:
            grad_z = grad_z.transpose(dim, -1)
        return grad_z, None


def entmax15(z, dim=-1):
    return _Entmax15Function.apply(z, dim)


class _SparsemaxFunction(torch.autograd.Function):
    """Standard sparsemax (Martins & Astudillo, 2016) with custom backward.

    Forward:  argmin_p ||p - z||^2  s.t.  p in probability simplex along `dim`.
    Output is a true sparse distribution: many entries are exactly 0.
    Backward derivative (well-known closed form):
        grad_z = support * (grad_p - mean(grad_p over support))
    """

    @staticmethod
    def forward(ctx, z, dim):
        if dim != -1:
            z = z.transpose(dim, -1).contiguous()
        K = z.size(-1)
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        range_k = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
        cumsum = z_sorted.cumsum(dim=-1)
        cond = (1.0 + range_k * z_sorted) > cumsum
        k_max = cond.to(z.dtype).sum(dim=-1).clamp(min=1.0)
        idx = (k_max - 1).long().unsqueeze(-1)
        tau = (cumsum.gather(-1, idx).squeeze(-1) - 1.0) / k_max
        p = (z - tau.unsqueeze(-1)).clamp(min=0.0)
        ctx.save_for_backward(p)
        ctx.dim = dim
        if dim != -1:
            p = p.transpose(dim, -1)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        if dim != -1:
            grad_output = grad_output.transpose(dim, -1).contiguous()
        support = (p > 0).to(grad_output.dtype)
        sum_support = support.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_grad = (grad_output * support).sum(dim=-1, keepdim=True) / sum_support
        grad_z = support * (grad_output - mean_grad)
        if dim != -1:
            grad_z = grad_z.transpose(dim, -1)
        return grad_z, None


def sparsemax(z, dim=-1):
    return _SparsemaxFunction.apply(z, dim)

# Try importing Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("⚠️ mamba_ssm not found, using LSTM fallback for BiMamba")

# ==========================================
# 1. Basic Blocks
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
            
        return x_out

class LabelGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x, adj):
        support = self.proj(x)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        output = torch.bmm(adj, support)
        return self.act(output)

class OTAlignment(nn.Module):
    def __init__(self, d_model, num_classes=None):
        super().__init__()

        # Sigmoid assignment 不要用 0.1 这种太小的温度，容易饱和
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 负初始化，让 assignment 初始更保守，有助于降低 false positives / Hamming Loss
        self.assign_bias = nn.Parameter(torch.tensor(-1.5))

        self.last_sim_matrix = None
        self.last_assignment = None

    def forward(self, image_features, label_embeddings):
        """
        image_features: (B, L, D)
        label_embeddings: (B, C, D)
        """
        B, L, D = image_features.shape

        img_norm = F.normalize(image_features, p=2, dim=-1)
        lbl_norm = F.normalize(label_embeddings, p=2, dim=-1)

        sim_matrix = torch.bmm(lbl_norm, img_norm.transpose(1, 2))  # (B, C, L)

        temp = self.temperature.abs() + 1e-6

        # Unbalanced sigmoid assignment:
        # 每个 label-instance pair 独立决定是否有证据，不再强制 sum_l A_{c,l}=1
        assignment = torch.sigmoid(sim_matrix / temp + self.assign_bias)  # (B, C, L)

        # 不按 assignment sum 归一化，否则负标签的小 assignment 会被重新放大
        aligned_feat = torch.bmm(assignment, image_features) / float(L)   # (B, C, D)

        self.last_sim_matrix = sim_matrix
        self.last_assignment = assignment

        return aligned_feat, assignment


class UnbalancedSoftOT(nn.Module):
    """
    闭式 (无迭代) 不平衡软 OT, 适合联邦学习场景.

    设计动机:
      - 多标签场景里多数标签缺席, 不能强制 marginal sum 守恒 (PatchCT 双 softmax 的问题)
      - 也不能像 sigmoid 那样完全无约束 (退化成独立门控, 没有 OT 含义)
      - Sinkhorn 迭代 K 次, 在 FL 下 5 客户端每步 5 次迭代代价过大

    解法:
      1. cost matrix C = 1 - cos(label, patch); dummy 背景行的 cost 由 patch 特征
         自适应预测 (per-patch dummy_head), 真背景 patch 自己流向 dummy.
      2. 行 softmax T_f: 每个标签把质量分到 patch (label -> patch)
         再乘上 image-conditional 标签 mass alpha(b, c), 缺席标签 mass -> 0,
         T_f 整行接近 0
      3. 列 softmax T_b: 每个 patch 把质量分到标签 (patch -> label),
         softmax 内嵌 log(mass) 先验, 缺席标签在反向也被压制
      4. 融合 T = 0.5 (T_f + T_b), 切片去掉 dummy 行后聚合特征,
         output 端再用 mass 门控一次, 缺席标签 aligned_feat 量级和方向都接近 0

    复杂度: 2 次 BMM + 2 次 softmax, 与原 sigmoid 同量级, 远低于 K-step Sinkhorn.
    """

    def __init__(self, d_model, num_classes=None, init_temp=1.0, init_dummy_cost=0.5):
        super().__init__()
        self.d_model = d_model

        # 温度通过 softplus 重参数化为正数 (real_temp = softplus(raw) + eps).
        # 处处可导且单调, 没有 abs() 在 0 处的梯度奇点.
        # 存储的 Parameter 初始化为 inv_softplus(target) 以保留原 API 含义.
        self.temperature = nn.Parameter(torch.tensor(_inv_softplus(float(init_temp))))

        # 每标签 mass 预测器: 从 [label_emb, image_global] 派生.
        # 之前只看 label_emb, 由于 label_embeddings 是 batch 广播的, mass[b, c]
        # 跟 b 无关, 只能学到类别先验. 现在拼上图像全局特征, mass 才是真正的
        # P(label c | image b), 配合 BCE 监督和 output gating 才有意义.
        self.mass_predictor = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # 每 patch dummy cost: 让背景代价随图像内容变化, 而不是全局共享一个标量.
        # 真前景 patch 学到大 dummy_cost (不愿流向背景), 真背景 patch 学到小
        # dummy_cost (主动流向背景), 真标签部分自然更稀疏.
        self.dummy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # 用 inv_softplus(init_dummy_cost) 初始化 dummy_head 最后一层 bias,
        # 让训练初期 softplus(dummy_logits) 的均值接近原来的 init_dummy_cost,
        # 避免初始化偏差打乱已经调好的 cost 尺度.
        with torch.no_grad():
            self.dummy_head[-1].bias.fill_(_inv_softplus(float(init_dummy_cost)))

        # === OT 专用投影 (Q/K 投影; V 仍用原始 image_features) ===
        # 之前 image_features 既算 cost 又被聚合, 同一组特征要兼顾 matching 和
        # content 两个目标, 互相妥协. 这里给 cost 计算单独开一条通道, OT 几何
        # 与内容几何解耦, 各自被各自的梯度专门拉扯.
        self.img_proj_ot = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.lbl_proj_ot = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # 调试 / 辅助 loss 缓存
        self.last_sim_matrix = None
        self.last_assignment = None
        self.last_transport_plan = None
        self.last_mass = None
        self.last_entropy = None       # (B, C) 真标签部分按 patch 维归一化后的熵
        self.last_consistency = None   # (B, C) 真标签部分 KL(T_f_norm || T_b_norm)

    def forward(self, image_features, label_embeddings):
        """
        image_features:   (B, L, D)
        label_embeddings: (B, C, D)
        Returns:
            aligned_feat: (B, C, D)  per-label aligned visual feature (dummy 已切除)
            assignment:   (B, C, L)  真标签部分的 transport 计划 (dummy 已切除)
        """
        B, L, D = image_features.shape
        _, C, _ = label_embeddings.shape

        # === 1. Cost matrix: 1 - cosine similarity, 用 OT 专用投影 (Q/K 解耦) ===
        img_for_cost = self.img_proj_ot(image_features)                # (B, L, D)
        lbl_for_cost = self.lbl_proj_ot(label_embeddings)              # (B, C, D)
        img_norm = F.normalize(img_for_cost, p=2, dim=-1)
        lbl_norm = F.normalize(lbl_for_cost, p=2, dim=-1)
        sim = torch.bmm(lbl_norm, img_norm.transpose(1, 2))            # (B, C, L)
        cost = 1.0 - sim                                               # (B, C, L)

        # 拼接 dummy 背景行: per-patch 学习的 cost, 让真背景 patch 主动流向 dummy.
        dummy_logits = self.dummy_head(image_features).squeeze(-1)     # (B, L)
        dummy_row = F.softplus(dummy_logits).unsqueeze(1)              # (B, 1, L)
        cost_aug = torch.cat([cost, dummy_row], dim=1)                 # (B, C+1, L)

        # === 2. 可学习每标签 mass (image-conditional) ===
        # 缺席标签可以学到 mass -> 0, 让它的 transport 行整体衰减.
        # 拼上图像全局特征, mass 真正成为 P(label c | image b), 不再 batch-broadcast.
        img_global = image_features.mean(dim=1)                          # (B, D)
        img_global_exp = img_global.unsqueeze(1).expand(-1, C, -1)       # (B, C, D)
        mass_input = torch.cat([label_embeddings, img_global_exp], dim=-1)  # (B, C, 2D)
        mass_logits = self.mass_predictor(mass_input).squeeze(-1)        # (B, C)
        mass = torch.sigmoid(mass_logits)                                # (B, C) in [0, 1]

        # dummy mass 固定为 1: 背景永远存在, 不参与 "可消失" 机制
        dummy_mass = torch.ones(B, 1, device=mass.device, dtype=mass.dtype)
        mass_aug = torch.cat([mass, dummy_mass], dim=1)                  # (B, C+1)

        # === 3. 双向 softmax (T_f 用 entmax-1.5 实现真稀疏) ===
        temp = F.softplus(self.temperature) + 1e-6
        logits = -cost_aug / temp                                        # (B, C+1, L)

        # 正向 (label -> patch): 行归一化用 entmax-1.5, 输出真稀疏 (不相关 patch = 0).
        # 相比 sparsemax, entmax-1.5 在支撑边界处梯度连续, 不会因为硬截断让边界
        # 样本的 logit 抖动放大 (对 HL/二值化指标更友好).
        # 缺席标签 mass~0, 整行 T_f 接近 0, 既不抢质量也不输出特征.
        T_f_base = entmax15(logits, dim=-1)                              # (B, C+1, L)
        T_f = T_f_base * mass_aug.unsqueeze(-1)                          # (B, C+1, L)

        # 反向 (patch -> label): 列归一化用 softmax, 把 mass 作为 log-prior 一并参与.
        # 缺席标签 (mass~0 -> log(mass) 很小) 在反向也会被抑制,
        # patch 自然流向 dummy 或其他高 mass 真标签, 真标签部分的 T_b 同样稀疏.
        # dummy mass=1 -> log(1)=0, 不偏置 dummy.
        log_mass = torch.log(mass_aug.clamp(min=1e-6)).unsqueeze(-1)     # (B, C+1, 1)
        T_b = F.softmax(logits + log_mass, dim=1)                        # (B, C+1, L)

        # === 4. 融合 + 切除 dummy ===
        T = 0.5 * (T_f + T_b)                                            # (B, C+1, L)
        T_real = T[:, :C, :]                                             # (B, C, L) 丢掉 dummy 行

        # 聚合特征后再用 mass 门控一次: 缺席标签的 aligned_feat 在量级和方向上都接近 0.
        # 这是结构性保证, 防止"虽然 transport 量小但方向仍像该类"的污染进入后续层.
        # 注意: 聚合用原始 image_features (V), 不用投影后的 (Q/K 解耦).
        aligned_feat = torch.bmm(T_real, image_features)                 # (B, C, D)
        aligned_feat = mass.unsqueeze(-1) * aligned_feat                 # (B, C, D)

        # === 5. 辅助指标: 熵 + 双向一致性 ===
        # 熵: 每个 (b, c) transport 行的熵, 用于可选的稀疏化辅助 loss.
        # 先按 patch 维做 row-normalization (每行已经被 mass 缩放了量级,
        # 这里只评估 "形状" 是否锐利, 因此重新归一化), 再求负熵.
        row_sum = T_real.sum(dim=-1, keepdim=True).clamp(min=1e-6)       # (B, C, 1)
        T_real_norm = T_real / row_sum                                    # (B, C, L)
        entropy = -(T_real_norm * torch.log(T_real_norm.clamp(min=1e-12))).sum(dim=-1)  # (B, C)

        # 一致性: T_f 和 T_b 描述同一张 plan 的两个方向, 理论上应该自洽.
        # 计算真标签部分 KL(T_f_norm || T_b_norm), 用于可选的一致性正则.
        T_f_real = T_f[:, :C, :]                                          # (B, C, L)
        T_b_real = T_b[:, :C, :]                                          # (B, C, L)
        T_f_real_norm = T_f_real / T_f_real.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        T_b_real_norm = T_b_real / T_b_real.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        consistency = (T_f_real_norm * torch.log(
            (T_f_real_norm.clamp(min=1e-12) / T_b_real_norm.clamp(min=1e-12))
        )).sum(dim=-1)  # (B, C)

        # 缓存
        self.last_sim_matrix = sim
        self.last_assignment = T_real
        self.last_transport_plan = T_real
        self.last_mass = mass
        self.last_entropy = entropy
        self.last_consistency = consistency

        return aligned_feat, T_real

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

# ==========================================
# 2. Decoder
# ==========================================

class AlignBiMambaDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_classes,
        num_layers=2,
        dropout=0.1,
        use_gcn=True,
        use_ot=True,
        ot_type='soft_uot',  # 'sigmoid' (旧) | 'soft_uot' (L2 闭式不平衡 OT)
        label_noise_std=0.25,
        label_dropout_p=0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_gcn = bool(use_gcn)
        self.use_ot = bool(use_ot)
        self.ot_type = str(ot_type)
        self.label_noise_std = float(label_noise_std)
        self.label_dropout_p = float(label_dropout_p)

        self.label_dropout = (
            nn.Dropout(self.label_dropout_p)
            if self.label_dropout_p and self.label_dropout_p > 0.0
            else None
        )

        # 1. Optional GCN branch
        if self.use_gcn:
            self.gcn = LabelGCN(d_model, d_model)
            self.dynamic_adj = DynamicAdjLayer(d_model, num_classes)
        else:
            self.gcn = None
            self.dynamic_adj = None

        # 2. OT branch — 支持两种实现, 通过 ot_type 切换
        if self.use_ot:
            if self.ot_type == 'soft_uot':
                self.ot_align = UnbalancedSoftOT(d_model, num_classes)
            elif self.ot_type == 'sigmoid':
                self.ot_align = OTAlignment(d_model, num_classes)
            else:
                raise ValueError(f"Unknown ot_type: {self.ot_type}")
        else:
            self.ot_align = None

        # 3. Fusion Layer
        self.fusion = nn.Linear(d_model * 2, d_model)

        # 4. SE Attention
        self.se_block = SEBlock(d_model)

        # 5. BiMamba Layers
        self.mamba_layers = nn.ModuleList([
            BiMambaBlock(d_model) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Label-wise bias.
        # Negative initialization makes the classifier more conservative,
        # which is usually helpful for reducing false positives / Hamming Loss.
        self.label_bias = nn.Parameter(torch.ones(num_classes) * -1.0)

        # Cache for analysis / auxiliary losses if needed
        self.last_transport_plan = None
        self.last_assignment = None

    def forward(self, image_features, label_embeddings, _=None):
        """
        image_features: (B, L, D)
        label_embeddings: (B, C, D)
        """

        # --- Step 1: Label Enhancement ---
        if self.use_gcn:
            adj = self.dynamic_adj(label_embeddings)
            label_ctx = self.gcn(label_embeddings, adj)
            enhanced_labels = label_embeddings + label_ctx
        else:
            enhanced_labels = label_embeddings

            # Optional regularization for no-GCN setting
            if self.training and (self.label_dropout is not None or self.label_noise_std > 0.0):
                enhanced_labels = enhanced_labels.clone()

                if self.label_dropout is not None:
                    enhanced_labels = self.label_dropout(enhanced_labels)

                if self.label_noise_std > 0.0:
                    enhanced_labels = enhanced_labels + torch.randn_like(enhanced_labels) * self.label_noise_std

        # --- Step 2: Unbalanced Sigmoid OT Assignment ---
        if self.use_ot:
            aligned_img_feat, assignment = self.ot_align(image_features, enhanced_labels)

            # For compatibility with old client.py / analysis code
            self.last_transport_plan = assignment
            self.last_assignment = assignment
        else:
            # Strong no-OT baseline: shared global visual context
            img_ctx = image_features.mean(dim=1, keepdim=True)
            aligned_img_feat = img_ctx.expand(-1, enhanced_labels.size(1), -1)

            self.last_transport_plan = None
            self.last_assignment = None

        # --- Step 3: Fusion & Sequence Construction ---
        fused_labels = self.fusion(
            torch.cat([enhanced_labels, aligned_img_feat], dim=-1)
        )
        fused_labels = self.norm(fused_labels)

        full_sequence = torch.cat([fused_labels, image_features], dim=1)
        full_sequence = self.dropout(full_sequence)

        # --- Step 4: BiMamba Reasoning ---
        x = full_sequence
        for layer in self.mamba_layers:
            x = layer(x) + x

        x = self.norm(x)
        x = self.se_block(x)

        # --- Step 5: Classification ---
        num_classes = label_embeddings.size(1)
        final_label_features = x[:, :num_classes, :]

        logits = (final_label_features * enhanced_labels).sum(dim=-1)

        # Label-wise bias for multi-label calibration
        logits = logits + self.label_bias

        return logits
# ==========================================
# 3. Main Model
# ==========================================

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
    
    # Simple mapping for VOC (can be extended)
    voc_glove_map = {'aeroplane': 'airplane', 'pottedplant': 'plant', 'tvmonitor': 'monitor', 'diningtable': 'table'}
    
    matrix = []
    for cls in class_names:
        # Lowercase and map
        cls_lower = cls.lower()
        query_word = voc_glove_map.get(cls_lower, cls_lower)
        vec = embeddings_dict.get(query_word, np.random.normal(scale=0.6, size=(glove_dim,)))
        matrix.append(vec)
    matrix = torch.tensor(np.array(matrix)).float()
    if glove_dim != d_model:
        with torch.no_grad(): matrix = nn.Linear(glove_dim, d_model)(matrix)
    return nn.Parameter(matrix.unsqueeze(0))

class BiMambaModel(nn.Module):
    def __init__(self, num_classes, class_names, d_model=256, pretrained=True, glove_path=None,
                 freeze_backbone=False, use_gcn=True, use_decoder=True, use_ot=True,
                 ot_type='soft_uot'):
        super().__init__()
        self.num_classes = num_classes
        self.use_decoder = bool(use_decoder)
        self.use_ot = bool(use_ot)
        self.ot_type = str(ot_type)
        
        # Backbone: ResNet18 (remove last 2 layers)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Freezing Logic
        if freeze_backbone == 'all' or freeze_backbone is True:
            for param in resnet.parameters():
                param.requires_grad = False
            print("🔒 Backbone Frozen (All Layers)")
        elif freeze_backbone == 'layer4':
            for param in resnet.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            print("🔒 Backbone Frozen (Except Layer 4)")
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.backbone_dim = 512
        self.feature_proj = nn.Conv2d(self.backbone_dim, d_model, kernel_size=1)
        
        # Load Glove
        self.label_embeddings = load_glove_embeddings(class_names, d_model, glove_path)
        
        # Decoder
        if self.use_decoder:
            self.decoder = AlignBiMambaDecoder(d_model, num_classes,
                                               use_gcn=use_gcn,
                                               use_ot=self.use_ot,
                                               ot_type=self.ot_type)
            self.logit_scale = None
            self.no_decoder_dropout = None
            self.feature_noise_std = 0.0
        else:
            self.decoder = None
            self.logit_scale = nn.Parameter(torch.tensor(10.0))
            self.no_decoder_dropout = nn.Dropout(0.5)
            self.feature_noise_std = 0.05

    def forward(self, images, adj_matrix=None):
    # 1. Features: (B, 512, 7, 7)
        features = self.backbone(images)

    # 2. Project: (B, d_model, 7, 7)
        features = self.feature_proj(features)

    # 3. Flatten: (B, 49, d_model)
        features = features.flatten(2).transpose(1, 2)

        batch_labels = self.label_embeddings.expand(images.size(0), -1, -1)

        if not self.use_decoder:
            bag = features.mean(dim=1)

            if self.training:
                bag_reg = bag
                if self.no_decoder_dropout is not None:
                    bag_reg = self.no_decoder_dropout(bag_reg)
                if self.feature_noise_std and self.feature_noise_std > 0.0:
                    bag_reg = bag_reg + torch.randn_like(bag_reg) * float(self.feature_noise_std)
            else:
                bag_reg = bag

            bag_norm = F.normalize(bag_reg, dim=-1)
            label_norm = F.normalize(batch_labels, dim=-1)

            logits = torch.matmul(
            bag_norm.unsqueeze(1),
            label_norm.transpose(1, 2)
        ).squeeze(1)

            if self.logit_scale is not None:
                logits = logits * self.logit_scale

            return logits

        logits = self.decoder(features, batch_labels)
        return logits

# ==========================================
# 4. Loss
# ==========================================

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        xs_pos = x.sigmoid()
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y) 
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = -one_sided_w * (los_pos + los_neg)
        else:
            loss = - (los_pos + los_neg)
        return loss.sum()

def compute_negative_assignment_loss(model, targets):
    """
    Auxiliary loss to suppress overly confident instance assignment for absent labels.

    Args:
        model: a BiMambaModel instance.
        targets: multi-label targets of shape (B, C), where 1=positive and 0=negative.

    Returns:
        A scalar tensor loss if transport plans are available; otherwise None.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        return None

    transport_plan = getattr(model.decoder, "last_transport_plan", None)
    if transport_plan is None:
        return None

    neg_mask = 1.0 - targets.float()             # (B, C)
    neg_peak = transport_plan.max(dim=-1).values # (B, C)

    loss_neg = (neg_peak * neg_mask).sum() / (neg_mask.sum() + 1e-6)
    return loss_neg


def compute_ot_entropy_reg(model, targets):
    """
    Auxiliary regularizer that sharpens the transport plan for *present*
    labels by penalising the entropy of their per-patch distribution.

    Absent labels are skipped — their rows should already be near-zero through
    the mass / dummy mechanism, and entropy on a near-zero row is meaningless.

    Returns:
        Scalar mean entropy over present labels, or None if unavailable.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        return None
    ot_align = getattr(model.decoder, "ot_align", None)
    if ot_align is None:
        return None
    entropy = getattr(ot_align, "last_entropy", None)
    if entropy is None:
        return None

    pos_mask = targets.float()                       # (B, C)
    denom = pos_mask.sum() + 1e-6
    return (entropy * pos_mask).sum() / denom


def compute_mass_bce(model, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
    """
    Asymmetric (focal) supervision on the OT mass head, mirroring the
    AsymmetricLoss design (Ridnik et al., 2021).

    Why asymmetric / focal:
      Multi-label targets are extremely imbalanced (~85% absent labels in VOC).
      Plain BCE on `mass` saturates near the class prior — absent-label mass
      stalls at ~0.15 instead of ~0. With (mass)^gamma_neg down-weighting on
      easy negatives, the optimizer is forced to keep pushing on borderline
      / mid-confidence absent labels, driving mass toward true 0. The
      probability-shift `clip` zeros out the loss for already-easy absents
      (mass < clip), preventing them from dragging gradients.

      Together, this lets the mass head learn an honest existence posterior
      so the output-side gating `mass * aligned_feat` actually suppresses
      absent-label features instead of leaking ~15% of their magnitude.

    Args:
        gamma_neg, gamma_pos: focal exponents (asymmetric for negatives).
        clip: ASL probability shift, set to 0 to disable.

    Returns scalar loss (mean over B*C), or None if mass is unavailable.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        return None
    ot_align = getattr(model.decoder, "ot_align", None)
    if ot_align is None:
        return None
    mass = getattr(ot_align, "last_mass", None)
    if mass is None:
        return None

    targets = targets.float()
    xs_pos = mass
    xs_neg = 1.0 - mass

    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)

    los_pos = targets * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps))

    pt0 = xs_pos * targets
    pt1 = xs_neg * (1.0 - targets)
    pt = pt0 + pt1
    one_sided_gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

    loss = -one_sided_w * (los_pos + los_neg)
    return loss.mean()


def compute_ot_consistency_reg(model, targets):
    """
    Auxiliary regularizer that pushes the forward (T_f) and backward (T_b)
    transport plans to agree on PRESENT labels. Cached as last_consistency on
    the OT module: KL(T_f_norm || T_b_norm) per (B, C). We only average over
    present labels — for absent labels both rows are near zero and KL is noisy.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        return None
    ot_align = getattr(model.decoder, "ot_align", None)
    if ot_align is None:
        return None
    consistency = getattr(ot_align, "last_consistency", None)
    if consistency is None:
        return None

    pos_mask = targets.float()                       # (B, C)
    denom = pos_mask.sum() + 1e-6
    return (consistency * pos_mask).sum() / denom

