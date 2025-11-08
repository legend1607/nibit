import json
import time
from os.path import join
import numpy as np
import open3d as o3d

# ===============================
# 安全数据清洗函数
# ===============================
def sanitize_label(x):
    """数值稳定清洗：去除NaN、极小值，归一化并强制float32"""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    if np.max(x) > 0:
        x /= np.max(x)
    x[x < 1e-3] = 0.0
    return x.astype(np.float32).copy()

# ===============================
# 保存 npz 数据
# ===============================
def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    """保存 npz 数据集（修正版）"""
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == "token":
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            arr = np.stack(raw_dataset[k], axis=0).astype(np.float32)
            raw_dataset_saved[k] = arr
    filename = "_tmp.npz" if tmp else mode + ".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)

# ===============================
# 路径 soft 标签生成
# ===============================
def get_path_label(pc, path, s_goal=None, sigma=None, sigma_ratio=0.05):
    pc = np.asarray(pc)
    path = np.asarray(path)
    if s_goal is None:
        s_goal = path[-1]

    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        sigma = sigma_ratio * range_vec
    sigma = np.asarray(sigma)
    sigma_eps = np.maximum(sigma, 1e-8)

    path_label = np.zeros(len(pc), dtype=np.float32)

    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        vec = pc - p0
        t = np.clip(np.sum(vec * seg_vec, axis=1) / (seg_len**2), 0, 1)
        proj = p0 + t[:, None] * seg_vec
        diff = (pc - proj) / sigma_eps
        dist2 = np.sum(diff**2, axis=1)
        label = np.exp(-0.5 * dist2)
        path_label = np.maximum(path_label, label)

    return sanitize_label(path_label)

# ===============================
# 关键点 soft 标签生成
# ===============================
def get_keypoint_label(pc, keypoints, sigma=None, sigma_ratio=0.05):
    if len(keypoints) == 0:
        return np.zeros(len(pc), dtype=np.float32)

    pc = np.asarray(pc)
    keypoints = np.asarray(keypoints)

    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        if range_vec.max() / max(range_vec.min(), 1e-8) > 3:
            sigma = sigma_ratio * range_vec
        else:
            sigma = sigma_ratio * np.mean(range_vec)

    sigma = np.asarray(sigma)
    sigma_eps = np.maximum(sigma, 1e-8)

    diff = (pc[:, None, :] - keypoints[None, :, :]) / sigma_eps
    dist2 = np.sum(diff**2, axis=2)
    label = np.exp(-0.5 * np.min(dist2, axis=1))

    return sanitize_label(label)

# ===============================
# 点云邻域掩码
# ===============================
def get_point_cloud_mask_around_points(point_cloud, points, neighbor_radius=3):
    point_cloud = np.asarray(point_cloud)
    points = np.asarray(points)
    assert point_cloud.shape[1] == points.shape[1], "point_cloud 和 points 维度不一致"

    diff = point_cloud[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    neighbor_mask = dist < neighbor_radius
    return np.any(neighbor_mask, axis=1)

# ===============================
# Farthest Point Sampling (FPS)
# ===============================
def sample_fps_point_cloud(env, n_points, oversample_scale=5):
    n_oversample = n_points * oversample_scale
    points = np.array([env.sample_empty_points() for _ in range(n_oversample)])
    points_3d = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd_fps = pcd.farthest_point_down_sample(num_samples=n_points)
    return np.asarray(pcd_fps.points)[:, :2]

# ===============================
# 主函数：生成 npz 数据集
# ===============================
def generate_npz_dataset(env_type="random_2d_bitstar"):
    from environment.random_2d_env import Random2DEnv as Env

    dataset_dir = join("data", f"{env_type}")
    n_points = 2048
    start_radius = 10
    goal_radius = 10

    for mode in ["train", "val"]:  # 可扩展 ["train", "val", "test"]
        env_json_path = join(dataset_dir, mode, "envs.json")
        with open(env_json_path, "r") as f:
            env_list = json.load(f)

        raw_dataset = {
            "token": [],
            "pc": [],
            "start": [],
            "goal": [],
            "free": [],
            "path": [],
            "keypoint": [],
        }

        start_time = time.time()

        for env_dict in env_list:
            env = Env(env_dict)
            env_idx = env_dict["env_idx"]

            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                s_start, s_goal = np.array(s_start), np.array(s_goal)
                start_point = s_start[np.newaxis, :]
                goal_point = s_goal[np.newaxis, :]
                sample_title = f"{env_idx}_{sample_idx}"
                path = np.array(env_dict["paths"][sample_idx])
                token = f"{mode}-{sample_title}"

                # 点云采样 + FPS
                pc = sample_fps_point_cloud(env, n_points, oversample_scale=5)

                # 掩码计算
                around_start_mask = get_point_cloud_mask_around_points(pc, start_point, neighbor_radius=start_radius)
                around_goal_mask = get_point_cloud_mask_around_points(pc, goal_point, neighbor_radius=goal_radius)
                freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

                # 标签生成
                path_label = get_path_label(pc, path)
                keypoints = path[1:]
                keypoint_label = get_keypoint_label(pc, keypoints)

                # 保存数据
                raw_dataset["token"].append(token)
                raw_dataset["pc"].append(pc.astype(np.float32))
                raw_dataset["start"].append(around_start_mask.astype(np.float32))
                raw_dataset["goal"].append(around_goal_mask.astype(np.float32))
                raw_dataset["free"].append(freespace_mask.astype(np.float32))
                raw_dataset["path"].append(path_label)
                raw_dataset["keypoint"].append(keypoint_label)

            # 临时保存
            if (env_idx + 1) % 25 == 0:
                save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=True)
                time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
                print(f"{mode} {env_idx + 1}/{len(env_list)}, remaining time: {int(time_left)} min")

        save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
        print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz")


if __name__ == "__main__":
    generate_npz_dataset("random_2d")
