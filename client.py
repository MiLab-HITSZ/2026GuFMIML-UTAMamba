import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import (
    BiMambaModel,
    AsymmetricLoss,
    compute_mass_bce,
    compute_ot_consistency_reg,
)


class Client:
    def __init__(self, client_id, dataset, config, allowed_labels=None):
        self.client_id = client_id
        self.dataset = dataset
        self.cfg = config
        self.device = config.DEVICE
        self.allowed_labels = allowed_labels

        self.loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        def get_class_names(ds):
            if hasattr(ds, "class_names"):
                return ds.class_names
            elif hasattr(ds, "classes"):
                return ds.classes
            elif hasattr(ds, "dataset"):
                return get_class_names(ds.dataset)
            elif hasattr(ds, "datasets"):
                return get_class_names(ds.datasets[0])
            else:
                return [f"class_{i}" for i in range(config.NUM_CLASSES)]

        class_names = get_class_names(dataset)

        self.model = BiMambaModel(
            num_classes=config.NUM_CLASSES,
            class_names=class_names,
            d_model=config.D_MODEL,
            glove_path=config.GLOVE_PATH,
            freeze_backbone=config.FREEZE_BACKBONE,
            use_gcn=getattr(config, "USE_GCN", True),
            use_decoder=getattr(config, "USE_DECODER", True),
            use_ot=getattr(config, "USE_OT", True),
            ot_type=getattr(config, "OT_TYPE", "soft_uot"),
        ).to(self.device)

        # 当前默认使用 BCEWithLogitsLoss，默认 reduction="mean"
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        # 如果后面要切 ASL，再改成这一行：
        # self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05).to(self.device)

    def _compute_loss(self, logits, targets):
        return self.criterion(logits, targets.float())

    def _safe_minmax(self, x):
        if x is None:
            return "None"

        if not torch.is_tensor(x):
            return str(x)

        if x.numel() == 0:
            return "empty"

        x_detach = x.detach()
        x_safe = torch.nan_to_num(x_detach, nan=0.0, posinf=1e6, neginf=-1e6)

        return (
            f"shape={tuple(x.shape)}, "
            f"finite={torch.isfinite(x_detach).all().item()}, "
            f"min={x_safe.min().item():.6f}, "
            f"max={x_safe.max().item():.6f}, "
            f"mean={x_safe.float().mean().item():.6f}"
        )

    def _print_target_stats(self, targets, prefix=""):
        with torch.no_grad():
            targets_safe = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)

            if targets_safe.dim() == 2:
                pos_per_sample = targets_safe.sum(dim=1)
                pos_per_class = targets_safe.sum(dim=0)

                print(
                    f"{prefix}target stats: "
                    f"shape={tuple(targets.shape)}, "
                    f"finite={torch.isfinite(targets).all().item()}, "
                    f"min={targets_safe.min().item():.4f}, "
                    f"max={targets_safe.max().item():.4f}, "
                    f"total_pos={targets_safe.sum().item():.1f}, "
                    f"pos/sample_mean={pos_per_sample.float().mean().item():.4f}, "
                    f"pos/sample_min={pos_per_sample.min().item():.1f}, "
                    f"pos/sample_max={pos_per_sample.max().item():.1f}, "
                    f"active_classes={(pos_per_class > 0).sum().item()}/{targets_safe.size(1)}"
                )
            else:
                print(
                    f"{prefix}target stats: "
                    f"shape={tuple(targets.shape)}, "
                    f"finite={torch.isfinite(targets).all().item()}, "
                    f"min={targets_safe.min().item():.4f}, "
                    f"max={targets_safe.max().item():.4f}, "
                    f"sum={targets_safe.sum().item():.1f}"
                )

    def _check_model_params(self, stage):
        bad_params = []

        for name, p in self.model.named_parameters():
            if p is None:
                continue

            if not torch.isfinite(p).all():
                bad_params.append(name)

                if len(bad_params) <= 5:
                    print(
                        f"❌ [Client {self.client_id}] Non-finite parameter at {stage}: {name}, "
                        f"{self._safe_minmax(p)}"
                    )

        if len(bad_params) > 0:
            print(
                f"❌ [Client {self.client_id}] Found {len(bad_params)} non-finite parameters at {stage}."
            )
            return False

        return True

    def _check_gradients(self):
        bad_grads = []

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            if not torch.isfinite(p.grad).all():
                bad_grads.append(name)

                if len(bad_grads) <= 5:
                    print(
                        f"❌ [Client {self.client_id}] Non-finite gradient: {name}, "
                        f"{self._safe_minmax(p.grad)}"
                    )

        if len(bad_grads) > 0:
            print(
                f"❌ [Client {self.client_id}] Found {len(bad_grads)} non-finite gradients."
            )
            return False

        return True

    def _print_ot_debug(self):
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return

        ot_align = getattr(decoder, "ot_align", None)
        if ot_align is None:
            return

        mass = getattr(ot_align, "last_mass", None)
        assignment = getattr(ot_align, "last_assignment", None)
        entropy = getattr(ot_align, "last_entropy", None)
        consistency = getattr(ot_align, "last_consistency", None)

        print(f"      OT mass: {self._safe_minmax(mass)}")
        print(f"      OT assignment: {self._safe_minmax(assignment)}")
        print(f"      OT entropy: {self._safe_minmax(entropy)}")
        print(f"      OT consistency: {self._safe_minmax(consistency)}")

    def local_train(self, global_params, epochs, current_round):
        print(f"🟢 [Client {self.client_id}] Starting training...")

        dataset_size = len(self.dataset)
        loader_steps = len(self.loader)

        print(
            f"   [Client {self.client_id}] "
            f"dataset_size={dataset_size}, "
            f"loader_steps={loader_steps}, "
            f"batch_size={self.cfg.BATCH_SIZE}, "
            f"epochs={epochs}, "
            f"round={current_round}"
        )

        if dataset_size == 0 or loader_steps == 0:
            print(f"⚠️ [Client {self.client_id}] Empty dataset or empty dataloader. Skip training.")
            return self.model.state_dict()

        self.model.load_state_dict(global_params)
        self.model.train()

        if not self._check_model_params(stage="after loading global params"):
            raise RuntimeError(f"[Client {self.client_id}] Global params already contain NaN/Inf.")

        # Freeze BN statistics during local training
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.LR,
            weight_decay=self.cfg.WEIGHT_DECAY
        )

        total_loss = 0.0
        total_base_loss = 0.0
        total_mass_loss = 0.0
        total_cons_loss = 0.0

        steps = 0
        nan_happened = False

        debug_every = int(getattr(self.cfg, "DEBUG_EVERY", 50))
        stop_on_nan = bool(getattr(self.cfg, "STOP_ON_NAN", True))

        lambda_mass = float(getattr(self.cfg, "LAMBDA_MASS", 0.0))
        lambda_cons = float(getattr(self.cfg, "LAMBDA_CONSISTENCY", 0.0))

        for epoch in range(epochs):
            print(f"   [Client {self.client_id}] Epoch {epoch + 1}/{epochs}")

            for batch_idx, (imgs, targets) in enumerate(self.loader):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()

                should_print = (
                    batch_idx == 0
                    or ((batch_idx + 1) % debug_every == 0)
                )

                if should_print:
                    print(
                        f"      [Client {self.client_id}] "
                        f"epoch={epoch + 1}, batch={batch_idx + 1}/{loader_steps}"
                    )
                    print(f"      imgs: {self._safe_minmax(imgs)}")
                    self._print_target_stats(targets, prefix="      ")

                if not torch.isfinite(imgs).all():
                    print(f"❌ [Client {self.client_id}] NaN/Inf detected in input images.")
                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite images.")
                    break

                if not torch.isfinite(targets).all():
                    print(f"❌ [Client {self.client_id}] NaN/Inf detected in targets.")
                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite targets.")
                    break

                if targets.min().item() < 0 or targets.max().item() > 1:
                    print(f"❌ [Client {self.client_id}] Targets are not valid 0/1 labels.")
                    print(f"      targets: {self._safe_minmax(targets)}")
                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Invalid target values.")
                    break

                logits = self.model(imgs)

                if not torch.isfinite(logits).all():
                    print(f"❌ [Client {self.client_id}] NaN/Inf detected in logits.")
                    print(f"      logits: {self._safe_minmax(logits)}")
                    self._print_target_stats(targets, prefix="      ")
                    self._print_ot_debug()

                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite logits.")
                    break

                loss_base = self._compute_loss(logits, targets)

                if not torch.isfinite(loss_base):
                    print(f"❌ [Client {self.client_id}] NaN/Inf detected in base BCE loss.")
                    print(f"      base_loss={loss_base.item()}")
                    print(f"      logits: {self._safe_minmax(logits)}")
                    self._print_target_stats(targets, prefix="      ")
                    self._print_ot_debug()

                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite base loss.")
                    break

                loss = loss_base
                loss_mass_value = 0.0
                loss_cons_value = 0.0

                if lambda_mass > 0.0:
                    loss_mass = compute_mass_bce(self.model, targets)

                    if loss_mass is not None:
                        if not torch.isfinite(loss_mass):
                            print(f"❌ [Client {self.client_id}] NaN/Inf detected in mass loss.")
                            print(f"      mass_loss={loss_mass.item()}")
                            self._print_target_stats(targets, prefix="      ")
                            self._print_ot_debug()

                            nan_happened = True
                            if stop_on_nan:
                                raise RuntimeError(f"[Client {self.client_id}] Non-finite mass loss.")
                            break

                        loss = loss + lambda_mass * loss_mass
                        loss_mass_value = loss_mass.item()

                if lambda_cons > 0.0:
                    loss_cons = compute_ot_consistency_reg(self.model, targets)

                    if loss_cons is not None:
                        if not torch.isfinite(loss_cons):
                            print(f"❌ [Client {self.client_id}] NaN/Inf detected in consistency loss.")
                            print(f"      consistency_loss={loss_cons.item()}")
                            self._print_target_stats(targets, prefix="      ")
                            self._print_ot_debug()

                            nan_happened = True
                            if stop_on_nan:
                                raise RuntimeError(f"[Client {self.client_id}] Non-finite consistency loss.")
                            break

                        loss = loss + lambda_cons * loss_cons
                        loss_cons_value = loss_cons.item()

                if not torch.isfinite(loss):
                    print(f"❌ [Client {self.client_id}] NaN/Inf detected in total loss.")
                    print(f"      base_loss={loss_base.item():.6f}")
                    print(f"      mass_loss={loss_mass_value:.6f}, lambda_mass={lambda_mass}")
                    print(f"      cons_loss={loss_cons_value:.6f}, lambda_cons={lambda_cons}")
                    self._print_ot_debug()

                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite total loss.")
                    break

                optimizer.zero_grad()
                loss.backward()

                if not self._check_gradients():
                    self._print_ot_debug()

                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite gradients.")
                    break

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                if not torch.isfinite(grad_norm):
                    print(f"❌ [Client {self.client_id}] NaN/Inf detected in grad_norm.")
                    print(f"      grad_norm={grad_norm}")
                    self._print_ot_debug()

                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Non-finite grad norm.")
                    break

                optimizer.step()

                if not self._check_model_params(stage=f"after optimizer step, epoch={epoch + 1}, batch={batch_idx + 1}"):
                    self._print_ot_debug()

                    nan_happened = True
                    if stop_on_nan:
                        raise RuntimeError(f"[Client {self.client_id}] Parameters became non-finite after optimizer step.")
                    break

                total_loss += loss.item()
                total_base_loss += loss_base.item()
                total_mass_loss += loss_mass_value
                total_cons_loss += loss_cons_value
                steps += 1

                if should_print:
                    print(
                        f"      losses: "
                        f"base={loss_base.item():.6f}, "
                        f"mass={loss_mass_value:.6f}, "
                        f"cons={loss_cons_value:.6f}, "
                        f"total={loss.item():.6f}, "
                        f"grad_norm={float(grad_norm):.6f}"
                    )
                    print(
                        f"      logits: "
                        f"min={logits.min().item():.6f}, "
                        f"max={logits.max().item():.6f}, "
                        f"mean={logits.mean().item():.6f}"
                    )

            if nan_happened:
                break

        avg_loss = total_loss / max(1, steps)
        avg_base_loss = total_base_loss / max(1, steps)
        avg_mass_loss = total_mass_loss / max(1, steps)
        avg_cons_loss = total_cons_loss / max(1, steps)

        if nan_happened:
            print(
                f"   ❌ [Client {self.client_id}] Stopped because NaN/Inf was detected. "
                f"finished_steps={steps}"
            )
        else:
            print(
                f"   [Client {self.client_id}] Done. "
                f"Avg Loss: {avg_loss:.6f} "
                f"(base={avg_base_loss:.6f}, "
                f"mass={avg_mass_loss:.6f}, "
                f"cons={avg_cons_loss:.6f}, "
                f"steps={steps})"
            )

        return self.model.state_dict()