import os
import json
import numpy as np
from environment.liche_env import LicheEnv  # 使用 LicheEnv 做采样和碰撞检测  :contentReference[oaicite:2]{index=2}

# ============================
# 配置
# ============================
ENV_TYPE = "liche"
base_dir = "data"
splits = ["train", "val", "test"]

npoints = 2000

# 路径邻近判定阈值（归一化后空间）
PATH_DIST_THRESH = 0.2

# start/goal 邻近阈值（归一化后空间）
START_DIST_THRESH = 0.1
GOAL_DIST_THRESH = 0.1

# 过采样倍数：先生成 M 个候选点（允许碰撞），再过滤出 free 点，然后 FPS 取 npoints
OVERSAMPLE_FACTOR = 6  # 建议比原来的 3 稍大一些，过滤碰撞后更稳

# 如果过滤后 free 点不足，会额外补采样
EXTRA_BATCH_SIZE = 2048
MAX_EXTRA_TRIES = 80


# ============================
# FPS (Farthest Point Sampling)
# ============================
def farthest_point_sampling(points, n_samples, start_idx=None):
    """
    points: (N, D) float32
    return: idx (n_samples,) int64
    """
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if n_samples >= N:
        return np.arange(N, dtype=np.int64)

    if start_idx is None:
        start_idx = np.random.randint(0, N)

    idx = np.empty((n_samples,), dtype=np.int64)
    idx[0] = start_idx

    # squared distance to nearest selected point
    dist = np.sum((points - points[start_idx]) ** 2, axis=1)

    for i in range(1, n_samples):
        farthest = int(np.argmax(dist))
        idx[i] = farthest
        d = np.sum((points - points[farthest]) ** 2, axis=1)
        dist = np.minimum(dist, d)

    return idx


# ============================
# 点到路径（polyline）邻近判定：归一化后计算距离
# ============================
def points_to_polyline_is_near(points, path, pose_range, thresh=PATH_DIST_THRESH):
    points = np.asarray(points, dtype=np.float32)
    path = np.asarray(path, dtype=np.float32)
    pose_range = np.asarray(pose_range, dtype=np.float32)

    low = pose_range[:, 0]
    high = pose_range[:, 1]
    span = high - low
    span[span == 0] = 1e-6

    P = (points - low) / span  # (N, D)
    X = (path - low) / span    # (T, D)

    if X.shape[0] < 2:
        d = np.linalg.norm(P - X[None, 0, :], axis=-1)
        return (d <= thresh).astype(np.int8), d

    A = X[:-1]                     # (S, D)
    B = X[1:]                      # (S, D)
    AB = B - A                     # (S, D)
    denom = np.sum(AB * AB, axis=-1) + 1e-12  # (S,)

    PA = P[:, None, :] - A[None, :, :]        # (N, S, D)
    t = np.sum(PA * AB[None, :, :], axis=-1) / denom[None, :]  # (N, S)
    t = np.clip(t, 0.0, 1.0)

    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]      # (N, S, D)
    dists = np.linalg.norm(P[:, None, :] - proj, axis=-1)       # (N, S)

    min_d = dists.min(axis=1)  # (N,)
    is_near = (min_d <= thresh).astype(np.int8)
    return is_near, min_d


# ============================
# 点到单点（start/goal）邻近掩码：归一化后计算距离
# ============================
def points_near_anchor_mask(points, anchor, pose_range, thresh):
    """
    points: (N, D) raw
    anchor: (D,) raw
    return: (N,) int8
    """
    points = np.asarray(points, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    pose_range = np.asarray(pose_range, dtype=np.float32)

    low = pose_range[:, 0]
    high = pose_range[:, 1]
    span = high - low
    span[span == 0] = 1e-6

    P = (points - low) / span
    a = (anchor - low) / span
    d = np.linalg.norm(P - a[None, :], axis=-1)
    return (d <= thresh).astype(np.int8)


# ============================
# 单个 split 处理函数
# ============================
def process_split(split_name):
    json_path = os.path.join(base_dir, ENV_TYPE, split_name, "envs.json")
    out_path = os.path.join(base_dir, ENV_TYPE, f"{split_name}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"\n=== 处理 {split_name.upper()} ===")
    print(f"读取 {json_path} ...")
    with open(json_path, "r") as f:
        env_list = json.load(f)

    # 最终只保留：token, pc, free, start, goal, astar
    token_list = []
    pc_list = []
    free_list = []
    start_list = []
    goal_list = []
    astar_list = []

    sample_count = 0
    skipped_samples = 0

    for env_idx, env_data in enumerate(env_list):
        pose_range = np.array(env_data["pose_range"], dtype=np.float32)
        obstacles = env_data["obstacles"]

        starts = env_data["start"]
        goals = env_data["goal"]
        paths = env_data["paths"]

        low = pose_range[:, 0]
        high = pose_range[:, 1]
        span = high - low
        span[span == 0] = 1e-6

        env_sim = LicheEnv(GUI=False)
        env_sim.clear_obstacles()
        for typ, size, pos in obstacles:
            if typ == "box":
                env_sim.add_box_obstacle(size, pos)
            elif typ == "sphere":
                r = float(size[0])
                env_sim.add_sphere_obstacle(r, pos)

        num_paths = len(paths)
        for i in range(num_paths):
            start_raw = np.array(starts[i], dtype=np.float32)
            goal_raw = np.array(goals[i], dtype=np.float32)
            path_raw = np.array(paths[i], dtype=np.float32)

            D = start_raw.shape[0]
            M = npoints * OVERSAMPLE_FACTOR

            # -------- 生成 M 个候选点（允许碰撞）--------
            states_raw_big = []

            # 用归一化 path 估计路径长度（保持你原先逻辑）
            path_norm = (path_raw - low) / span
            T = len(path_norm)

            if T > 1:
                seg_vecs = path_norm[1:] - path_norm[:-1]
                seg_lens = np.linalg.norm(seg_vecs, axis=1)
                total_len = float(np.sum(seg_lens))
            else:
                seg_lens = None
                total_len = 0.0

            SAMPLES_PER_UNIT_LEN = 800
            SAMPLES_PER_WAYPOINT = 5
            M_path_raw = SAMPLES_PER_UNIT_LEN * total_len + SAMPLES_PER_WAYPOINT * T
            M_path_eff = int(np.clip(M_path_raw, 0, M))
            M_uniform = M - M_path_eff

            # 1) 全局均匀采样
            for _ in range(M_uniform):
                states_raw_big.append(env_sim.uniform_sample())

            # 2) 路径附近采样
            if M_path_eff > 0:
                if T < 2:
                    center = path_norm[0]
                    for _ in range(M_path_eff):
                        dir_noise = np.random.normal(size=D).astype(np.float32)
                        norm = np.linalg.norm(dir_noise) + 1e-9
                        dir_noise = dir_noise / norm
                        r = PATH_DIST_THRESH * np.random.rand()
                        q_norm = np.clip(center + dir_noise * r, 0.0, 1.0)
                        q = np.clip(low + q_norm * span, low, high)
                        states_raw_big.append(q)
                else:
                    seg_lens_safe = seg_lens + 1e-9
                    total_len_safe = float(np.sum(seg_lens_safe))
                    raw_counts = M_path_eff * seg_lens_safe / total_len_safe
                    samples_per_seg = np.floor(raw_counts).astype(int)
                    assigned = int(np.sum(samples_per_seg))
                    remaining = M_path_eff - assigned
                    if remaining > 0:
                        frac = raw_counts - samples_per_seg
                        order = np.argsort(-frac)
                        for k in range(remaining):
                            samples_per_seg[order[k]] += 1

                    num_segs = T - 1
                    for seg_id in range(num_segs):
                        nseg = samples_per_seg[seg_id]
                        if nseg <= 0:
                            continue
                        p0 = path_norm[seg_id]
                        p1 = path_norm[seg_id + 1]
                        for _ in range(nseg):
                            t = np.random.rand()
                            base_norm = (1.0 - t) * p0 + t * p1
                            dir_noise = np.random.normal(size=D).astype(np.float32)
                            norm = np.linalg.norm(dir_noise) + 1e-9
                            dir_noise = dir_noise / norm
                            r = PATH_DIST_THRESH * np.random.rand()
                            q_norm = np.clip(base_norm + dir_noise * r, 0.0, 1.0)
                            q = np.clip(low + q_norm * span, low, high)
                            states_raw_big.append(q)

            states_raw_big = np.array(states_raw_big, dtype=np.float32)

            # 补齐到 M
            if len(states_raw_big) != M:
                if len(states_raw_big) > M:
                    idx = np.random.choice(len(states_raw_big), size=M, replace=False)
                    states_raw_big = states_raw_big[idx]
                else:
                    extra = M - len(states_raw_big)
                    extra_samples = np.array([env_sim.uniform_sample() for _ in range(extra)], dtype=np.float32)
                    states_raw_big = np.concatenate([states_raw_big, extra_samples], axis=0)

            # -------- 过滤碰撞点：只保留 free 点 --------
            free_big = np.array([env_sim._state_fp(q) for q in states_raw_big], dtype=bool)
            free_idx = np.where(free_big)[0]

            # 如果 free 点不足，继续补采样直到够 npoints 或达到上限
            tries = 0
            while free_idx.shape[0] < npoints and tries < MAX_EXTRA_TRIES:
                tries += 1
                extra = np.array([env_sim.uniform_sample() for _ in range(EXTRA_BATCH_SIZE)], dtype=np.float32)
                extra_free = np.array([env_sim._state_fp(q) for q in extra], dtype=bool)
                extra = extra[extra_free]
                if extra.shape[0] > 0:
                    states_raw_big = np.concatenate([states_raw_big, extra], axis=0)
                    free_big = np.concatenate([free_big, np.ones((extra.shape[0],), dtype=bool)], axis=0)
                    free_idx = np.where(free_big)[0]

            if free_idx.shape[0] < npoints:
                skipped_samples += 1
                continue

            states_free = states_raw_big[free_idx]               # (M_free, D)
            pc_free = ((states_free - low) / span).astype(np.float32)  # (M_free, D) normalized

            # -------- FPS：在 free 点集合上均匀取 npoints --------
            fps_local_idx = farthest_point_sampling(pc_free, npoints)
            states_raw = states_free[fps_local_idx]     # raw (npoints, D)
            pc_norm = pc_free[fps_local_idx]           # normalized (npoints, D)

            # -------- 生成四种掩码 --------
            start_mask = points_near_anchor_mask(states_raw, start_raw, pose_range, START_DIST_THRESH).astype(np.float32)
            goal_mask = points_near_anchor_mask(states_raw, goal_raw, pose_range, GOAL_DIST_THRESH).astype(np.float32)

            # free：按你 3D 数据集的定义（非 start/goal 邻域）
            free_mask = ((1.0 - start_mask) * (1.0 - goal_mask)).astype(np.float32)

            astar_mask, _ = points_to_polyline_is_near(states_raw, path_raw, pose_range, PATH_DIST_THRESH)
            astar_mask = astar_mask.astype(np.float32)
            # astar_mask: (npoints,), 0/1
            path_ratio = astar_mask.mean()

            print(
                f"[{split_name}] env {env_idx} sample {i} | "
                f"path points: {int(astar_mask.sum())}/{npoints} "
                f"({astar_mask.mean()*100:.2f}%)"
            )


            # -------- 保存 sample --------
            token = f"{split_name}-{env_idx}_{i}"
            token_list.append(token)
            pc_list.append(pc_norm)
            free_list.append(free_mask)
            start_list.append(start_mask)
            goal_list.append(goal_mask)
            astar_list.append(astar_mask)

            sample_count += 1

        env_sim.close()
        print(f"Env {env_idx}: {num_paths} 条路径 → {sample_count} 个样本累计 (skipped={skipped_samples})")

    # ============================
    # 保存 npz（只保留你要的字段）
    # ============================
    np.savez_compressed(
        out_path,
        token=np.array(token_list),
        pc=np.stack(pc_list, axis=0).astype(np.float32),       # (B, npoints, D)
        free=np.stack(free_list, axis=0).astype(np.float32),   # (B, npoints)
        start=np.stack(start_list, axis=0).astype(np.float32), # (B, npoints)
        goal=np.stack(goal_list, axis=0).astype(np.float32),   # (B, npoints)
        astar=np.stack(astar_list, axis=0).astype(np.float32), # (B, npoints)
    )

    print(f"✅ {split_name}.npz 完成：{sample_count} samples, skipped={skipped_samples}")
    print("字段: token, pc, free, start, goal, astar")
    if sample_count > 0:
        print(f"pc: {np.stack(pc_list, axis=0).shape}, masks: {np.stack(free_list, axis=0).shape}")


# ============================
# 主入口
# ============================
if __name__ == "__main__":
    for split in splits:
        process_split(split)
    print("\n🎉 全部分割(train/val/test)处理完成！")
