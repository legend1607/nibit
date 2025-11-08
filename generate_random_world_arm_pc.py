import json
import time
from matplotlib import pyplot as plt
import yaml
from os.path import join

import cv2
import numpy as np



def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    """保存 npz 数据集"""
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == "token":
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0)  # (b, n_points, ...)
    filename = "_tmp.npz" if tmp else mode + ".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)

def get_path_label(pc, path, s_goal=None, sigma=None, sigma_ratio=0.08, diff_threshold=3.0):
    """
    根据路径生成 soft 标签，每个维度可有不同 sigma（智能自适应）。
    
    pc: (N, D) 点云
    path: (M, D) 路径点
    s_goal: (D,) 目标点，可选
    sigma: None 或数组/标量
    sigma_ratio: sigma 相对于点云范围的比例
    diff_threshold: 当 range_max/range_min > diff_threshold 时使用统一 sigma
    """
    pc = np.asarray(pc)
    path = np.asarray(path)
    if s_goal is None:
        s_goal = path[-1]

    # 自动计算 sigma
    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        # 判断维度差异是否过大
        if range_vec.max() / max(range_vec.min(), 1e-8) > diff_threshold:
            # 使用统一 sigma
            sigma = sigma_ratio * np.mean(range_vec)
        else:
            # 各维度独立 sigma
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

        # 各维度加权距离（支持向量 sigma 或统一 sigma）
        diff = (pc - proj) / sigma_eps
        dist2 = np.sum(diff**2, axis=1)

        label = np.exp(-0.5 * dist2)
        path_label = np.maximum(path_label, label)

    return path_label.astype(np.float32)

def get_keypoint_label(pc, keypoints, sigma=None, sigma_ratio=0.08):
    """
    根据关键点生成 soft 标签（高斯衰减，支持自动 sigma 自适应）
    ------------------------------------------------------------
    pc: (N, D)  点云
    keypoints: (K, D)  关键点
    sigma: float 或 None
        - 若为 None，则根据 pc 的范围自动计算
        - 若为 float，则使用固定标量 sigma
    sigma_ratio: float
        - sigma 相对于点云范围的比例（默认 0.03）
    ------------------------------------------------------------
    返回:
        label: (N,) 每个点的 soft 权重标签
    """
    if len(keypoints) == 0:
        return np.zeros(len(pc), dtype=np.float32)

    pc = np.asarray(pc)
    keypoints = np.asarray(keypoints)

    # ✅ 自动计算 sigma（自适应场景尺度）
    if sigma is None:
        range_vec = pc.max(axis=0) - pc.min(axis=0)
        # 判断是否存在维度差异很大的情况
        if range_vec.max() / max(range_vec.min(), 1e-8) > 3:
            # 各维度独立 sigma
            sigma = sigma_ratio * range_vec
        else:
            # 单一 sigma
            sigma = sigma_ratio * np.mean(range_vec)
    sigma = np.asarray(sigma)
    sigma_eps = np.maximum(sigma, 1e-8)

    # ✅ 计算距离（支持向量 sigma）
    diff = (pc[:, None, :] - keypoints[None, :, :]) / sigma_eps
    dist2 = np.sum(diff**2, axis=2)  # (N, K)
    label = np.exp(-0.5 * np.min(dist2, axis=1))

    return label.astype(np.float32)

def get_point_cloud_mask_around_points(point_cloud, points, neighbor_radius=3):
    """
    - 自动支持 point_cloud 和 points 的任意维度
    - point_cloud: (n, C)
    - points: (m, C)
    - neighbor_radius: 半径阈值
    """
    point_cloud = np.asarray(point_cloud)
    points = np.asarray(points)

    # 检查维度一致
    assert point_cloud.shape[1] == points.shape[1], "point_cloud 和 points 维度不一致"

    # 计算欧氏距离
    diff = point_cloud[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, m, C)
    dist = np.linalg.norm(diff, axis=2)  # (n, m)

    # 判断是否在邻域半径内
    neighbor_mask = dist < neighbor_radius  # (n, m)
    around_points_mask = np.any(neighbor_mask, axis=1)  # (n,)

    return around_points_mask

def generate_npz_dataset(env_type="liche"):
    """把 generate_dataset 生成的环境转换成 .npz 训练集"""
    if env_type =="liche":
        from environment.liche_env import LicheEnv
        env= LicheEnv(GUI=False)
    dataset_dir = join("data", f"{env_type}")
    n_points = 2048
    start_radius = 0.1
    goal_radius = 0.1
    path_radius = 0.1

    for mode in ["test"]:
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
            env_idx=env_dict["env_idx"]
            # --- 1️⃣ 初始化障碍 ---
            env.clear_obstacles()  # 重置环境
            for obs in env_dict["obstacles"]:
                if obs[0] == "box":
                    half_extents, pos = obs[1], obs[2]
                    env.add_box_obstacle(half_extents, pos)
                elif obs[0] == "ball":
                    radius, pos = obs[1], obs[2]
                    env.add_sphere_obstacle(radius, pos)


            skip_env = False
            samples_data = []  # 暂存该环境的有效样本

            for sample_idx, (s_start, s_goal) in enumerate(zip(env_dict["start"], env_dict["goal"])):
                s_start, s_goal = np.array(s_start), np.array(s_goal)
                start_point = s_start[np.newaxis, :]
                goal_point = s_goal[np.newaxis, :]

                sample_title = f"{env_idx}_{sample_idx}"
                path = np.loadtxt(join(dataset_dir, mode, "paths", sample_title + ".txt"), delimiter=",")

                if len(path) <= 2:
                    print(f"跳过样本 {sample_title}：路径点数 {len(path)} 过少")
                    continue
                # 提取关键点
                keypoints = path[1:-1]
                if len(keypoints) == 0:
                    print(f"跳过样本 {sample_title}：无关键点")
                    continue
                token = mode + "-" + sample_title

                # 生成点云
                pc = []
                for _ in range(n_points):
                    p = env.sample_empty_points()  # 返回 ndarray
                    pc.append(p)
                pc = np.array(pc)
                around_start_mask = get_point_cloud_mask_around_points(pc, start_point, neighbor_radius=start_radius)
                around_goal_mask = get_point_cloud_mask_around_points(pc, goal_point, neighbor_radius=goal_radius)
                
                path_label = get_path_label(pc, path)
                freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

                keypoint_label = get_keypoint_label(pc, keypoints)

                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')

                # 绘制点云
                ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=2, c='lightgray', label='Point cloud')

                # 绘制起点和终点
                ax.scatter(s_start[0], s_start[1], s_start[2], c='green', s=100, marker='*', edgecolors='k', label='Start')
                ax.scatter(s_goal[0], s_goal[1], s_goal[2], c='magenta', s=100, marker='*', edgecolors='k', label='Goal')

                # 根据路径标签绘制路径点
                ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=path_label, cmap='Reds', s=15, alpha=0.6)

                # 绘制关键点
                # if len(keypoints) > 0:
                #     ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=keypoint_label, cmap='Oranges', s=15, alpha=0.6)


                # 保存样本
                raw_dataset["token"].append(token)
                raw_dataset["pc"].append(pc.astype(np.float32))
                raw_dataset["start"].append(around_start_mask.astype(np.float32))
                raw_dataset["goal"].append(around_goal_mask.astype(np.float32))
                raw_dataset["free"].append(freespace_mask.astype(np.float32))
                raw_dataset["path"].append(path_label.astype(np.float32))
                raw_dataset["keypoint"].append(keypoint_label.astype(np.float32))

            if (env_idx + 1) % 25 == 0:
                save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=True)
                time_left = (time.time() - start_time) * (len(env_list) / (env_idx + 1) - 1) / 60
                print(f"{mode} {env_idx + 1}/{len(env_list)}, remaining time: {int(time_left)} min")

        save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
        print(f"[{mode}] saved {len(raw_dataset['token'])} samples to {mode}.npz")


if __name__ == "__main__":
    # 先运行 generate_dataset("random_2d", planner_type="bitstar") 生成 envs.json + PNG + BIT* 路径
    # 然后运行本脚本，生成 train/val/test 的 .npz 文件
    generate_npz_dataset("liche")
