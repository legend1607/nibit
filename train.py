# train.py
import os
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ArmPointCloudDataset, collate_fn_with_meta_and_pathlabel
from model.encoders.joint_pointlite_encoder import JointPointNetEncoder


# -----------------------------
# Logger
# -----------------------------
def setup_logger(log_dir: str, filename: str, name: str = "train"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log_path = os.path.join(log_dir, filename)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logger initialized. Writing to: {log_path}")
    return logger


# -----------------------------
# Loss utils
# -----------------------------
def bce_logits_pos_weight(logits: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
    """BCE with dynamic pos_weight to fight imbalance."""
    pos = target01.sum()
    neg = target01.numel() - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, target01, pos_weight=pos_weight)


def pairwise_ranking_loss(p: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, pairs_per_sample=1024):
    """
    Ranking loss: if t_i > t_j then want p_i > p_j.
    Helps "distance layers" generalize when sampling distribution shifts.
    """
    B, N = p.shape
    loss_sum = p.new_zeros(())
    count = 0

    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue

        m = idx.numel()
        num_pairs = min(pairs_per_sample, m * (m - 1) // 2)

        i = idx[torch.randint(0, m, (num_pairs,), device=p.device)]
        j = idx[torch.randint(0, m, (num_pairs,), device=p.device)]

        ti = t[b, i]
        tj = t[b, j]
        s = torch.sign(ti - tj)
        valid = s != 0
        if valid.sum() == 0:
            continue

        s = s[valid]
        pi = p[b, i][valid]
        pj = p[b, j][valid]

        loss_b = F.softplus(-s * (pi - pj)).mean()
        loss_sum = loss_sum + loss_b
        count += 1

    return loss_sum / max(count, 1)


def nonzero_uniformity_regularizer(p: torch.Tensor, mask: torch.Tensor, bins: torch.Tensor):
    """Soft hist spread regularizer (optional)."""
    pm = p[mask]
    if pm.numel() < 20:
        return p.new_zeros(())

    centers = (bins[:-1] + bins[1:]) / 2.0
    widths = (bins[1:] - bins[:-1]).clamp_min(1e-6)

    dist = (pm[:, None] - centers[None, :]).abs()
    w = (1.0 - dist / (widths[None, :] + 1e-6)).clamp_min(0.0)
    hist = w.sum(dim=0)
    hist = hist / (hist.sum() + 1e-6)

    K = hist.numel()
    target = hist.new_full((K,), 1.0 / K)
    return F.mse_loss(hist, target)


def hard_hist_counts(x_1d: torch.Tensor, bins: torch.Tensor):
    ids = torch.bucketize(x_1d, bins) - 1
    ids = ids.clamp(0, len(bins) - 2)
    K = len(bins) - 1
    out = torch.zeros(K, device=x_1d.device)
    for k in range(K):
        out[k] = (ids == k).float().sum()
    return out


# -----------------------------
# Configs
# -----------------------------
@dataclass
class LossWeights:
    lambda_path_max: float = 1.0
    lambda_cons: float = 0.1
    w_rank: float = 0.3
    w_udist: float = 1.0
    w_uniform: float = 0.0
    w_coll0: float = 0.2  # collision path->0


@dataclass
class TailCfg:
    # Tail-focused: make high values closer to GT
    alpha_tail: float = 4.0     # weight scale for BCE at high t
    gamma_tail: float = 2.0     # weight shape for BCE
    t_high: float = 0.8         # define "tail region" by GT t>=t_high
    w_under_high: float = 0.5   # punish under-pred in high t
    w_tail_mean: float = 0.2    # mean calibration in high t
    log_tail_thr: float = 0.9   # log tail mass at >=0.9


# -----------------------------
# Logging: zero / nonzero / hist (+tail)
# -----------------------------
@torch.no_grad()
def log_zero_and_hist(
    model, loader, device, logger,
    gt_zero_eps=1e-6,
    pred_zero_th=0.01,
    hist_eps=0.05,
    tail_thr=0.9,
):
    model.eval()

    bins = torch.tensor([0.05, 0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001], device=device)

    gt0_all = pr0_all = 0
    gt0_free = pr0_free = 0
    gt0_coll = pr0_coll = 0
    n_all = n_free = n_coll = 0

    tp = fp = fn = 0

    gt_hist = torch.zeros(len(bins) - 1, device=device)
    pr_hist = torch.zeros(len(bins) - 1, device=device)
    hist_n = 0

    # tail stats on (free & GT>=hist_eps)
    tail_total = 0
    gt_tail_cnt = 0
    pr_tail_cnt = 0
    gt_tail_mean = None
    pr_tail_mean = None

    for env_voxels, joint_states, labels, pathlabels, _ in loader:
        env_voxels = env_voxels.to(device)
        joint_states = joint_states.to(device)
        labels = labels.to(device)
        t = pathlabels.to(device).float().clamp(0.0, 1.0)

        logits, pathlogits, _ = model(env_voxels, joint_states)
        p = torch.sigmoid(pathlogits).clamp(0.0, 1.0)

        mask_free = (labels == 1)
        mask_coll = (labels == 0)

        gt0 = (t <= gt_zero_eps)
        pr0 = (p < pred_zero_th)

        n_all += t.numel()
        gt0_all += gt0.sum().item()
        pr0_all += pr0.sum().item()

        n_free += mask_free.sum().item()
        gt0_free += (gt0 & mask_free).sum().item()
        pr0_free += (pr0 & mask_free).sum().item()

        n_coll += mask_coll.sum().item()
        gt0_coll += (gt0 & mask_coll).sum().item()
        pr0_coll += (pr0 & mask_coll).sum().item()

        gt_nz = mask_free & (t >= hist_eps)
        pr_nz = mask_free & (p >= hist_eps)

        tp += (gt_nz & pr_nz).sum().item()
        fp += ((~gt_nz) & pr_nz & mask_free).sum().item()
        fn += (gt_nz & (~pr_nz)).sum().item()

        mask_hist = mask_free & (t >= hist_eps)
        if mask_hist.any():
            gt = t[mask_hist]
            pr = p[mask_hist]
            gt_hist += hard_hist_counts(gt, bins)
            pr_hist += hard_hist_counts(pr, bins)
            hist_n += gt.numel()

            # tail mass
            tail_total += gt.numel()
            gt_tail_cnt += (gt >= tail_thr).sum().item()
            pr_tail_cnt += (pr >= tail_thr).sum().item()

            # tail means (on GT tail region)
            gt_tail_mask = gt >= tail_thr
            if gt_tail_mask.any():
                m_gt = gt[gt_tail_mask].mean()
                m_pr = pr[gt_tail_mask].mean()
                gt_tail_mean = m_gt if gt_tail_mean is None else 0.5 * gt_tail_mean + 0.5 * m_gt
                pr_tail_mean = m_pr if pr_tail_mean is None else 0.5 * pr_tail_mean + 0.5 * m_pr

    if n_all > 0:
        logger.info(f"[zero] all : GT==0={gt0_all/n_all:.3f} | Pred<{pred_zero_th}={pr0_all/n_all:.3f}")
    if n_free > 0:
        logger.info(f"[zero] free: GT==0={gt0_free/n_free:.3f} | Pred<{pred_zero_th}={pr0_free/n_free:.3f}")
    if n_coll > 0:
        logger.info(f"[zero] coll: GT==0={gt0_coll/n_coll:.3f} | Pred<{pred_zero_th}={pr0_coll/n_coll:.3f}")

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    logger.info(f"[nz-det] free, thr={hist_eps:.3f}: precision={prec:.3f} recall={rec:.3f} (tp={tp}, fp={fp}, fn={fn})")

    if hist_n == 0:
        logger.info("[hist] no points in (free & GT>=hist_eps) region.")
        return

    gt_ratio = (gt_hist / gt_hist.sum()).tolist()
    pr_ratio = (pr_hist / pr_hist.sum()).tolist()

    logger.info(f"[hist] bins: {[(float(bins[i]), float(bins[i+1])) for i in range(len(bins)-1)]}")
    logger.info(f"[hist] GT  : {['{:.3f}'.format(r) for r in gt_ratio]}")
    logger.info(f"[hist] Pred: {['{:.3f}'.format(r) for r in pr_ratio]}")

    if tail_total > 0:
        logger.info(f"[tail@{tail_thr:.2f}] mass: GT={(gt_tail_cnt/tail_total):.3f} | Pred={(pr_tail_cnt/tail_total):.3f}")
        if gt_tail_mean is not None and pr_tail_mean is not None:
            logger.info(f"[tail@{tail_thr:.2f}] mean_on_GTtail: GT={float(gt_tail_mean):.3f} | Pred={float(pr_tail_mean):.3f}")


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(
    model, loader, device, epoch: int, logger,
    optimizer=None, scaler=None, amp=False,
    warmup_epochs=10, eps_nonzero=0.05,
    weights: LossWeights = LossWeights(),
    tail: TailCfg = TailCfg(),
    fixed_lambda_path_for_eval: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {k: 0.0 for k in [
        "loss", "loss_cls", "loss_path_bce", "loss_udist", "loss_rank", "loss_uniform",
        "loss_cons", "loss_coll0", "loss_under_high", "loss_tail_mean", "acc_cls"
    ]}
    n_samples = 0

    bins = torch.tensor([0.05, 0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001], device=device)

    pbar = tqdm(
        loader,
        desc=f"{'train' if is_train else 'val'} e{epoch:03d}",
        dynamic_ncols=True,
        leave=False
    )

    use_amp = amp and (device.type == "cuda")
    ac_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for step, (env_voxels, joint_states, labels, pathlabels, _metas) in enumerate(pbar):
        env_voxels = env_voxels.to(device, non_blocking=True)
        joint_states = joint_states.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pathlabels = pathlabels.to(device, non_blocking=True).float()

        y_cls = labels.float()  # 1=free, 0=collision
        free_mask = (labels == 1)
        coll_mask = (labels == 0)
        nz_mask = (pathlabels >= eps_nonzero)
        mask_nz_free = free_mask & nz_mask

        if fixed_lambda_path_for_eval:
            lambda_path = weights.lambda_path_max
        else:
            lambda_path = weights.lambda_path_max * min(1.0, (epoch + 1) / max(warmup_epochs, 1))

        with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=use_amp):
            logits, pathlogits, _ = model(env_voxels, joint_states)

            # 1) classification
            loss_cls = bce_logits_pos_weight(logits, y_cls)

            # predictions + targets
            t = pathlabels.clamp(0.0, 1.0)
            p = torch.sigmoid(pathlogits).clamp(1e-6, 1 - 1e-6)

            # 2) tail-weighted path BCE on free
            loss_path_point = F.binary_cross_entropy_with_logits(pathlogits, t, reduction="none")
            # weight = 1 + alpha * t^gamma (only meaningful on free points)
            w_tail = 1.0 + tail.alpha_tail * (t ** tail.gamma_tail)
            w_free = free_mask.float() * w_tail
            loss_path_bce = (loss_path_point * w_free).sum() / (w_free.sum() + 1e-6)

            # 3) normalized distance regression on nonzero region: u = 1 - t
            u_t = (1.0 - t)
            u_p = (1.0 - p)
            if mask_nz_free.any():
                loss_udist = F.smooth_l1_loss(u_p[mask_nz_free], u_t[mask_nz_free], beta=0.02)
            else:
                loss_udist = p.new_zeros(())

            # 4) ranking on nonzero region
            loss_rank = pairwise_ranking_loss(p, t, mask_nz_free, pairs_per_sample=1024)

            # 5) optional: spread regularizer
            if weights.w_uniform > 0:
                loss_uniform = nonzero_uniformity_regularizer(p, mask_nz_free, bins)
            else:
                loss_uniform = p.new_zeros(())

            # 6) consistency: p_path <= p_free
            p_free = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            loss_cons = F.relu(p - p_free).mean()

            # 7) collision path -> 0 (helps clean 0 region; keep small if you mainly care tail)
            if weights.w_coll0 > 0 and coll_mask.any():
                loss_coll0_point = F.binary_cross_entropy_with_logits(
                    pathlogits, torch.zeros_like(t), reduction="none"
                )
                loss_coll0 = (loss_coll0_point * coll_mask.float()).sum() / (coll_mask.float().sum() + 1e-6)
            else:
                loss_coll0 = p.new_zeros(())

            # 8) Tail boosting: only in high GT region (free & t>=t_high)
            high_mask = free_mask & (t >= tail.t_high)
            if high_mask.any():
                # punish under-pred only (push tail upward)
                loss_under_high = (F.relu(t - p) ** 2)[high_mask].mean()
                # calibrate mean in tail region
                loss_tail_mean = (p[high_mask].mean() - t[high_mask].mean()) ** 2
            else:
                loss_under_high = p.new_zeros(())
                loss_tail_mean = p.new_zeros(())

            loss_path_total = (
                loss_path_bce
                + weights.w_udist * loss_udist
                + weights.w_rank * loss_rank
                + weights.w_uniform * loss_uniform
                + tail.w_under_high * loss_under_high
                + tail.w_tail_mean * loss_tail_mean
            )

            loss = (
                loss_cls
                + lambda_path * loss_path_total
                + weights.lambda_cons * loss_cons
                + weights.w_coll0 * loss_coll0
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            pred_cls = (torch.sigmoid(logits) > 0.5).float()
            acc = (pred_cls == y_cls).float().mean()

        if step % 20 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "cls": f"{loss_cls.item():.3f}",
                "path": f"{loss_path_bce.item():.3f}",
                "acc": f"{acc.item():.3f}",
            })

        bs = env_voxels.size(0)
        n_samples += bs
        totals["loss"] += loss.item() * bs
        totals["loss_cls"] += loss_cls.item() * bs
        totals["loss_path_bce"] += float(loss_path_bce.item()) * bs
        totals["loss_udist"] += float(loss_udist.item()) * bs
        totals["loss_rank"] += float(loss_rank.item()) * bs
        totals["loss_uniform"] += float(loss_uniform.item()) * bs
        totals["loss_cons"] += float(loss_cons.item()) * bs
        totals["loss_coll0"] += float(loss_coll0.item()) * bs
        totals["loss_under_high"] += float(loss_under_high.item()) * bs
        totals["loss_tail_mean"] += float(loss_tail_mean.item()) * bs
        totals["acc_cls"] += float(acc.item()) * bs

    for k in totals:
        totals[k] /= max(n_samples, 1)

    totals["lambda_path"] = (weights.lambda_path_max if fixed_lambda_path_for_eval else
                             (weights.lambda_path_max * min(1.0, (epoch + 1) / max(warmup_epochs, 1))))
    return totals


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", type=str, default="data/liche/train/train.npz")
    parser.add_argument("--val_npz", type=str, default="data/liche/val/val.npz")

    parser.add_argument("--save_dir", type=str, default="ckpts")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--eps_nonzero", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")

    # loss weights
    parser.add_argument("--w_rank", type=float, default=0.3)
    parser.add_argument("--w_udist", type=float, default=1.0)
    parser.add_argument("--w_uniform", type=float, default=0.0)
    parser.add_argument("--w_coll0", type=float, default=0.2)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--lambda_path_max", type=float, default=1.0)

    # tail focus
    parser.add_argument("--alpha_tail", type=float, default=6.0)
    parser.add_argument("--gamma_tail", type=float, default=2.0)
    parser.add_argument("--t_high", type=float, default=0.8)
    parser.add_argument("--w_under_high", type=float, default=0.8)
    parser.add_argument("--w_tail_mean", type=float, default=0.2)
    parser.add_argument("--log_tail_thr", type=float, default=0.8)

    # logging freq
    parser.add_argument("--log_every_epochs", type=int, default=1)
    parser.add_argument("--analyze_every_epochs", type=int, default=5)

    # zero/hist thresholds
    parser.add_argument("--pred_zero_th", type=float, default=0.01)
    parser.add_argument("--gt_zero_eps", type=float, default=1e-6)

    args = parser.parse_args()

    # unique run dir
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = args.run_name.strip() or "train"
    run_dir = os.path.join(args.save_dir, f"{prefix}_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(run_dir, filename=f"{prefix}_{run_ts}.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Train NPZ: {args.train_npz}")
    logger.info(f"Val   NPZ: {args.val_npz}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # dataset / loader
    train_ds = ArmPointCloudDataset(args.train_npz, max_points=args.max_points, shuffle_points=True)
    val_ds = ArmPointCloudDataset(args.val_npz, max_points=args.max_points, shuffle_points=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn_with_meta_and_pathlabel
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn_with_meta_and_pathlabel
    )

    # model
    dof = train_ds.dof
    model = JointPointNetEncoder(joint_in_dim=dof).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    weights = LossWeights(
        lambda_path_max=args.lambda_path_max,
        lambda_cons=args.lambda_cons,
        w_rank=args.w_rank,
        w_udist=args.w_udist,
        w_uniform=args.w_uniform,
        w_coll0=args.w_coll0,
    )
    tail = TailCfg(
        alpha_tail=args.alpha_tail,
        gamma_tail=args.gamma_tail,
        t_high=args.t_high,
        w_under_high=args.w_under_high,
        w_tail_mean=args.w_tail_mean,
        log_tail_thr=args.log_tail_thr,
    )

    logger.info(f"Weights: {weights}")
    logger.info(f"TailCfg : {tail}")
    logger.info("Best model is selected by val_score with FIXED lambda_path=1.0 (not affected by warmup).")

    best_val_score = float("inf")
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    for epoch in range(args.epochs):
        tr = run_epoch(
            model, train_loader, device, epoch, logger,
            optimizer=optimizer, scaler=scaler, amp=args.amp,
            warmup_epochs=args.warmup_epochs, eps_nonzero=args.eps_nonzero,
            weights=weights, tail=tail,
            fixed_lambda_path_for_eval=False,
        )

        va = run_epoch(
            model, val_loader, device, epoch, logger,
            optimizer=None, scaler=None, amp=args.amp,
            warmup_epochs=args.warmup_epochs, eps_nonzero=args.eps_nonzero,
            weights=weights, tail=tail,
            fixed_lambda_path_for_eval=True,   # comparable across epochs
        )

        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, last_path)

        if (epoch % args.log_every_epochs) == 0:
            logger.info(
                f"[{epoch:03d}] "
                f"train(score={tr['loss']:.4f}, lambda_path={tr['lambda_path']:.2f}) "
                f"cls={tr['loss_cls']:.4f} path_bce={tr['loss_path_bce']:.4f} "
                f"udist={tr['loss_udist']:.4f} rank={tr['loss_rank']:.4f} "
                f"u={tr['loss_uniform']:.4f} cons={tr['loss_cons']:.4f} coll0={tr['loss_coll0']:.4f} "
                f"under={tr['loss_under_high']:.4f} tailm={tr['loss_tail_mean']:.4f} acc={tr['acc_cls']:.3f} | "
                f"val(score={va['loss']:.4f}, lambda_path={va['lambda_path']:.2f}) "
                f"acc={va['acc_cls']:.3f}"
            )

        if (epoch % args.analyze_every_epochs) == 0:
            log_zero_and_hist(
                model, val_loader, device, logger,
                gt_zero_eps=args.gt_zero_eps,
                pred_zero_th=args.pred_zero_th,
                hist_eps=args.eps_nonzero,
                tail_thr=args.log_tail_thr,
            )

        val_score = va["loss"]
        if val_score < best_val_score:
            best_val_score = val_score
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, best_path)
            logger.info(f"  -> saved {best_path} (best_val_score={best_val_score:.6f})")

    logger.info("done.")


if __name__ == "__main__":
    main()
