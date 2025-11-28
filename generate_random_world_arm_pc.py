import os
import json
import numpy as np
from environment.liche_env import LicheEnv   # ✅ 使用 LicheEnv 做采样和碰撞检测

# ============================
# 配置
# ============================
ENV_TYPE = "liche"

base_dir = "data"
splits = ["train", "val", "test"]

npoints = 2000                           # 每个样本最终采样的关节状态数量
voxel_resolution = np.array([50, 50, 50], dtype=int)  # 体素分辨率 (X, Y, Z)

# 路径邻近判定阈值（归一化后空间）
PATH_DIST_THRESH = 0.05

# 过采样倍数
OVERSAMPLE_FACTOR = 3   # 实际采 M = npoints * OVERSAMPLE_FACTOR

# 加权抽样权重
W_PATH = 2.0       
W_COLL = 2.0       
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
def points_to_path_is_near(points, path, pose_range, thresh=PATH_DIST_THRESH):
    points = np.asarray(points, dtype=np.float32)
    path   = np.asarray(path, dtype=np.float32)
    pose_range = np.asarray(pose_range, dtype=np.float32)
    
    low  = pose_range[:, 0]
    high = pose_range[:, 1]
    span = high - low
    span[span == 0] = 1e-6  # 防止除零

    points_norm = (points - low) / span
    path_norm   = (path - low) / span

    N, D = points_norm.shape
    T, _ = path_norm.shape

    if T < 2:
        diff = points_norm - path_norm[0]
        distances = np.linalg.norm(diff, axis=1)
    else:
        p0 = path_norm[:-1]
        p1 = path_norm[1:]
        seg = p1 - p0

        v = points_norm[:, None, :] - p0[None, :, :]
        seg_len2 = np.sum(seg * seg, axis=1)
        seg_len2 = np.where(seg_len2 < 1e-9, 1e-9, seg_len2)
        t = np.sum(v * seg[None, :, :], axis=2) / seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = p0[None, :, :] + seg[None, :, :] * t[:, :, None]
        d2 = np.sum((proj - points_norm[:, None, :]) ** 2, axis=2)
        distances = np.sqrt(np.min(d2, axis=1))

    is_path = distances < thresh
    return is_path.astype(np.uint8), distances

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

    voxel_list, pc_list, starts_list, goals_list, labels_list = [], [], [], [], []
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

            # -------- 采样 --------
            states_raw_big = []
            M_path = int(M * 0.3)
            M_uniform = M - M_path

            for _ in range(M_uniform):
                states_raw_big.append(env_sim.uniform_sample())
            for _ in range(M_path):
                idx = np.random.randint(0, len(path_raw))
                base = path_raw[idx]
                noise = np.random.uniform(-PATH_DIST_THRESH*0.8,
                                           PATH_DIST_THRESH*0.8,
                                           size=base.shape)
                q = np.clip(base + noise, low, high)
                states_raw_big.append(q)

            states_raw_big = np.array(states_raw_big, dtype=np.float32)

            free_mask_big = np.array([env_sim._state_fp(q) for q in states_raw_big], dtype=bool)
            is_collision_big = (~free_mask_big).astype(np.uint8)

            # -------- 使用归一化后的路径邻近判定 --------
            is_path_big, distances_big = points_to_path_is_near(states_raw_big, path_raw, pose_range, PATH_DIST_THRESH)

            # -------- 权重抽样 --------
            weights = np.ones(M, dtype=np.float32) * W_FREE
            weights[is_collision_big == 1] = W_COLL
            weights[is_path_big == 1] = W_PATH
            prob = weights / np.sum(weights)

            if M >= npoints:
                final_idx = np.random.choice(M, size=npoints, replace=False, p=prob)
            else:
                final_idx = np.random.choice(M, size=npoints, replace=True, p=prob)

            states_raw  = states_raw_big[final_idx]
            is_collision = is_collision_big[final_idx]
            is_path      = is_path_big[final_idx]
            distances    = distances_big[final_idx]

            total_is_path += int(np.sum(is_path))
            total_paths   += 1

            label = np.ones(npoints, dtype=np.int8)
            label[is_path == 1] = 2
            label[is_collision == 1] = 0
            total_free += int(np.sum(label == 1))

            pc_norm    = (states_raw - low) / span
            start_norm = (start_raw  - low) / span
            goal_norm  = (goal_raw   - low) / span
            path_norm  = (path_raw   - low) / span

            voxel_list.append(voxel_grid)
            pc_list.append(pc_norm)
            starts_list.append(start_norm)
            goals_list.append(goal_norm)
            labels_list.append(label)

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
    env_ranges  = np.stack(env_ranges_list, axis=0)
    pose_ranges = np.stack(pose_ranges_list, axis=0)
    paths_arr     = np.array(paths_list, dtype=object)
    obstacles_arr = np.array(obstacles_list, dtype=object)

    print(f"voxel_grids: {voxel_grids.shape}, pc: {pc.shape}, labels: {labels.shape}")

    avg_is_path = total_is_path / total_paths if total_paths > 0 else 0
    ratio_ispath_free = total_is_path / total_free if total_free > 0 else 0.0
    print(f"📊 平均每条路径附近点数: {avg_is_path:.2f} / {npoints}")
    print(f"📈 is_path 占 free-space 比例: {ratio_ispath_free:.4f}")

    np.savez_compressed(
        out_path,
        voxel_grids=voxel_grids,
        pc=pc,
        starts=starts,
        goals=goals,
        labels=labels,
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
