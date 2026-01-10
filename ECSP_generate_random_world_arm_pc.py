import os
import json
import numpy as np
from environment.liche_env import LicheEnv   # ✅ 使用 LicheEnv 做采样和碰撞检测

# ============================
# 配置
# ============================
ENV_TYPE = "liche"

base_dir = "data"
splits = ["train","val","test"]

npoints = 2000                           # 每个样本最终采样的关节状态数量
voxel_resolution = np.array([50, 50, 50], dtype=int)  # 体素分辨率 (X, Y, Z)

# 路径邻近判定阈值（归一化后空间）
PATH_DIST_THRESH = 0.2

# 过采样倍数
OVERSAMPLE_FACTOR = 3   # 实际采 M = npoints * OVERSAMPLE_FACTOR

# 加权抽样权重
W_PATH = 1.0       
W_COLL = 1.0       
W_FREE = 1.0       

# ============================
# 体素化函数
# ============================
def voxelize_env(env_range, obstacles, resolution):
    env_range = np.asarray(env_range, dtype=np.float32)
    res = np.asarray(resolution, dtype=int)

    mins = env_range[:, 0]
    maxs = env_range[:, 1]
    scale = res / (maxs - mins)

    grid = np.zeros(res, dtype=np.uint8)

    for obs in obstacles:
        typ, size, pos = obs
        size = np.array(size, dtype=np.float32)
        pos  = np.array(pos, dtype=np.float32)

        if typ == "sphere":
            r = float(size[0])
            size = np.array([r, r, r], dtype=np.float32)

        lo = pos - size
        hi = pos + size

        idx_lo = ((lo - mins) * scale).astype(int)
        idx_hi = ((hi - mins) * scale).astype(int)
        idx_lo = np.clip(idx_lo, 0, res - 1)
        idx_hi = np.clip(idx_hi, 0, res - 1)

        grid[idx_lo[0]:idx_hi[0]+1,
             idx_lo[1]:idx_hi[1]+1,
             idx_lo[2]:idx_hi[2]+1] = 1

    return grid

# ============================
# 新：考虑关节范围归一化的路径邻近判定
# ============================
def points_to_polyline_is_near(points, path, pose_range, thresh=PATH_DIST_THRESH):
    points = np.asarray(points, dtype=np.float32)
    path   = np.asarray(path, dtype=np.float32)
    pose_range = np.asarray(pose_range, dtype=np.float32)

    low  = pose_range[:, 0]
    high = pose_range[:, 1]
    span = high - low
    span[span == 0] = 1e-6

    P = (points - low) / span                  # (N, D)
    X = (path   - low) / span                  # (T, D)

    if X.shape[0] < 2:
        # path 只有 1 个点时退化成 point-to-point
        d = np.linalg.norm(P - X[None, 0, :], axis=-1)
        return (d <= thresh).astype(np.int8), d

    A = X[:-1]                                 # (S, D)
    B = X[1:]                                  # (S, D)
    AB = B - A                                 # (S, D)
    denom = np.sum(AB * AB, axis=-1) + 1e-12   # (S,)

    PA = P[:, None, :] - A[None, :, :]         # (N, S, D)
    t = np.sum(PA * AB[None, :, :], axis=-1) / denom[None, :]  # (N, S)
    t = np.clip(t, 0.0, 1.0)

    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]      # (N, S, D)
    dists = np.linalg.norm(P[:, None, :] - proj, axis=-1)       # (N, S)

    min_d = dists.min(axis=1)                   # (N,)
    is_near = (min_d <= thresh).astype(np.int8)
    return is_near, min_d

# ============================
# 单个 split 处理函数
# ============================
def process_split(split_name):
    json_path = os.path.join(base_dir, ENV_TYPE, split_name, "envs.json")
    out_path = os.path.join(base_dir, ENV_TYPE, split_name, f"{split_name}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"\n=== 处理 {split_name.upper()} ===")
    print(f"读取 {json_path} ...")
    with open(json_path, "r") as f:
        env_list = json.load(f)

    voxel_list, pc_list, starts_list, goals_list, labels_list, pathlabels_list = [], [], [], [], [], []
    paths_list, env_ranges_list, pose_ranges_list, obstacles_list = [], [], [], []

    sample_count = 0
    total_is_path = 0
    total_paths = 0
    total_free = 0

    for env_idx, env_data in enumerate(env_list):
        env_range = np.array(env_data["env_range"], dtype=np.float32)
        pose_range = np.array(env_data["pose_range"], dtype=np.float32)
        obstacles = env_data["obstacles"]

        starts = env_data["start"]
        goals  = env_data["goal"]
        paths  = env_data["paths"]

        voxel_grid = voxelize_env(env_range, obstacles, voxel_resolution)

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
            goal_raw  = np.array(goals[i],  dtype=np.float32)
            path_raw  = np.array(paths[i],  dtype=np.float32)

            D = start_raw.shape[0]
            M = npoints * OVERSAMPLE_FACTOR

            # -------- 采样（路径附近 + 全局均匀，数量由路径长度+点数决定）--------
            states_raw_big = []

            # 归一化 path，用于计算长度
            path_norm = (path_raw - low) / span
            T = len(path_norm)

            # 计算整条路径在归一化空间里的总长度
            if T > 1:
                seg_vecs = path_norm[1:] - path_norm[:-1]       # (T-1, D)
                seg_lens = np.linalg.norm(seg_vecs, axis=1)     # (T-1,)
                total_len = float(np.sum(seg_lens))
            else:
                seg_vecs = None
                seg_lens = None
                total_len = 0.0

            # === 核心：根据路径长度 + 点数决定 near-path 采样数 ===
            SAMPLES_PER_UNIT_LEN   = 800
            SAMPLES_PER_WAYPOINT   = 5

            M_path_raw = SAMPLES_PER_UNIT_LEN * total_len + SAMPLES_PER_WAYPOINT * T
            M_path_eff = int(np.clip(M_path_raw, 0, M))
            M_uniform  = M - M_path_eff

            # 1) 全局均匀采样
            for _ in range(M_uniform):
                states_raw_big.append(env_sim.uniform_sample())

            # 2) 按“线段长度”采样路径附近（总共 M_path_eff 个点）
            if M_path_eff > 0:
                if T < 2:
                    center = path_norm[0]
                    for _ in range(M_path_eff):
                        dir_noise = np.random.normal(size=D).astype(np.float32)
                        norm = np.linalg.norm(dir_noise) + 1e-9
                        dir_noise = dir_noise / norm

                        r = PATH_DIST_THRESH * np.random.rand()
                        noise_norm = dir_noise * r

                        q_norm = center + noise_norm
                        q_norm = np.clip(q_norm, 0.0, 1.0)

                        q = low + q_norm * span
                        q = np.clip(q, low, high)
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
                        num_samples_seg = samples_per_seg[seg_id]
                        if num_samples_seg <= 0:
                            continue

                        p0 = path_norm[seg_id]
                        p1 = path_norm[seg_id + 1]

                        for _ in range(num_samples_seg):
                            t = np.random.rand()
                            base_norm = (1.0 - t) * p0 + t * p1

                            dir_noise = np.random.normal(size=D).astype(np.float32)
                            norm = np.linalg.norm(dir_noise) + 1e-9
                            dir_noise = dir_noise / norm

                            r = PATH_DIST_THRESH * np.random.rand()
                            noise_norm = dir_noise * r

                            q_norm = base_norm + noise_norm
                            q_norm = np.clip(q_norm, 0.0, 1.0)

                            q = low + q_norm * span
                            q = np.clip(q, low, high)
                            states_raw_big.append(q)

            states_raw_big = np.array(states_raw_big, dtype=np.float32)
            if len(states_raw_big) != M:
                if len(states_raw_big) > M:
                    idx = np.random.choice(len(states_raw_big), size=M, replace=False)
                    states_raw_big = states_raw_big[idx]
                else:
                    extra = M - len(states_raw_big)
                    extra_samples = np.array([env_sim.uniform_sample() for _ in range(extra)], dtype=np.float32)
                    states_raw_big = np.concatenate([states_raw_big, extra_samples], axis=0)

            free_mask_big = np.array([env_sim._state_fp(q) for q in states_raw_big], dtype=bool)
            is_collision_big = (~free_mask_big).astype(np.uint8)

            # -------- 使用归一化后的路径邻近判定 --------
            is_path_big, distances_big = points_to_polyline_is_near(
                states_raw_big, path_raw, pose_range, PATH_DIST_THRESH
            )

            # -------- 权重抽样 --------
            weights = np.ones(M, dtype=np.float32) * W_FREE
            weights[is_collision_big == 1] = W_COLL
            weights[is_path_big == 1] = W_PATH
            prob = weights / np.sum(weights)

            if M >= npoints:
                final_idx = np.random.choice(M, size=npoints, replace=False, p=prob)
            else:
                final_idx = np.random.choice(M, size=npoints, replace=True, p=prob)

            states_raw   = states_raw_big[final_idx]
            is_collision = is_collision_big[final_idx]
            is_path      = is_path_big[final_idx]
            distances    = distances_big[final_idx]

            total_is_path += int(np.sum(is_path))
            total_paths   += 1

            # =========================================================
            # ✅ labels 改为 free/collision 二分类
            #    0 = collision, 1 = free
            # =========================================================
            label = np.ones(npoints, dtype=np.int8)     # 默认 free=1
            label[is_collision == 1] = 0               # collision=0
            total_free += int(np.sum(label == 1))

            # =========================================================
            # ✅ 新增 pathlabel 软标签
            #    PATH_DIST_THRESH 内: distance 越小 soft 越大
            #    线性衰减: 1 - d/thresh, 之外为 0
            #    碰撞点强制为 0
            # =========================================================
            pathlabel = np.clip(
                1.0 - distances / PATH_DIST_THRESH, 0.0, 1.0
            ).astype(np.float32)
            pathlabel[is_collision == 1] = 0.0

            pc_norm    = (states_raw - low) / span
            start_norm = (start_raw  - low) / span
            goal_norm  = (goal_raw   - low) / span
            path_norm  = (path_raw   - low) / span

            voxel_list.append(voxel_grid)
            pc_list.append(pc_norm)
            starts_list.append(start_norm)
            goals_list.append(goal_norm)
            labels_list.append(label)
            pathlabels_list.append(pathlabel)   # ✅ 记得 append

            paths_list.append(path_norm)
            env_ranges_list.append(env_range)
            pose_ranges_list.append(pose_range)
            obstacles_list.append(obstacles)

            sample_count += 1

        env_sim.close()
        print(f"Env {env_idx}: {num_paths} 条路径 → {sample_count} 个样本累计")

    # ============================
    # 保存
    # ============================
    voxel_grids = np.stack(voxel_list, axis=0)
    pc          = np.stack(pc_list, axis=0)
    starts      = np.stack(starts_list, axis=0)
    goals       = np.stack(goals_list, axis=0)
    labels      = np.stack(labels_list, axis=0)
    pathlabels  = np.stack(pathlabels_list, axis=0)
    env_ranges  = np.stack(env_ranges_list, axis=0)
    pose_ranges = np.stack(pose_ranges_list, axis=0)
    paths_arr     = np.array(paths_list, dtype=object)
    obstacles_arr = np.array(obstacles_list, dtype=object)

    print(f"voxel_grids: {voxel_grids.shape}, pc: {pc.shape}, labels: {labels.shape}, pathlabels: {pathlabels.shape}")

    avg_is_path = total_is_path / total_paths if total_paths > 0 else 0
    ratio_ispath_free = total_is_path / total_free if total_free > 0 else 0.0
    print(f"📊 平均每条路径附近点数(hard): {avg_is_path:.2f} / {npoints}")
    print(f"📈 is_path(hard) 占 free-space 比例: {ratio_ispath_free:.4f}")

    np.savez_compressed(
        out_path,
        voxel_grids=voxel_grids,
        pc=pc,
        starts=starts,
        goals=goals,
        labels=labels,
        pathlabels=pathlabels,
        env_ranges=env_ranges,
        pose_ranges=pose_ranges,
        paths=paths_arr,
        obstacles=obstacles_arr,
    )
    print(f"✅ {split_name}.npz 完成，共 {sample_count} 个样本。")

# ============================
# 主入口
# ============================
if __name__ == "__main__":
    for split in splits:
        process_split(split)
    print("\n🎉 全部分割(train/val/test)处理完成！")
