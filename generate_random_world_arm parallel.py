import os
import json
import time
import random
import numpy as np
import pybullet as p
from os.path import join
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from environment.kuka_env import KukaEnv
from environment.ur5_env import UR5Env
from environment.liche_env import LicheEnv
from path_planning_classes_arm.bit_star import BITStar


# ---------------- 随机障碍物 ----------------
def add_random_obstacles(env, config):
    obstacles = []
    xyz_max = config.get("xyz_max", [5, 5, 5])
    x_range = [-xyz_max[0], xyz_max[0]]
    y_range = [-xyz_max[1], xyz_max[1]]
    z_max = xyz_max[2]

    # 随机方块
    for _ in range(random.randint(*config["num_boxes_range"])):
        size_x = random.uniform(*config["box_size_range"])
        size_y = random.uniform(*config["box_size_range"])
        size_z = random.uniform(*config["box_size_range"])
        half_extents = [size_x / 2, size_y / 2, size_z / 2]

        while True:
            pos = [
                random.uniform(*x_range),
                random.uniform(*y_range),
                random.uniform(half_extents[2], z_max)
            ]
            if abs(pos[0]) >= 0.1 and abs(pos[1]) >= 0.1:
                break
        env.add_box_obstacle(half_extents, pos)
        obstacles.append(("box", half_extents, pos))

    # 随机球体
    for _ in range(random.randint(*config["num_balls_range"])):
        radius = random.uniform(*config["ball_radius_range"])
        pos = [
            random.uniform(*x_range),
            random.uniform(*y_range),
            random.uniform(radius, z_max)
        ]
        env.add_sphere_obstacle(radius, pos)
        obstacles.append(("ball", radius, pos))

    return obstacles


# ---------------- 可视化 ----------------
def visualize_env_task(env, pause_time=1.0):
    if not env.GUI:
        return
    start_id = env.arm_id
    env.set_config(env.start, start_id)
    p.changeVisualShape(start_id, env.end_effector_index, rgbaColor=[0, 1, 0, 1])

    goal_id = p.loadURDF(env.arm_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
    env.set_config(env.goal, goal_id)
    p.changeVisualShape(goal_id, env.end_effector_index, rgbaColor=[1, 0, 0, 1])

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30,
                                 cameraPitch=-40, cameraTargetPosition=[0, 0, 0.5])
    t0 = time.time()
    while time.time() - t0 < pause_time:
        time.sleep(1 / 240.0)


# ---------------- 单环境生成 ----------------
def generate_single_env(args):
    env_idx, config = args

    while True:  # 无限循环直到生成至少一条路径
        path_list, start_list, goal_list = [], [], []
        try:
            # 初始化环境
            if config["env_type"] == "ur5":
                env = UR5Env(GUI=config.get("GUI", False))
            elif config["env_type"] == "kuka":
                env = KukaEnv(GUI=config.get("GUI", False))
            elif config["env_type"] == "liche":
                env = LicheEnv(GUI=config.get("GUI", False))
            else:
                raise ValueError(f"未知的环境类型: {config['env_type']}")

            # 添加随机障碍
            obstacles = add_random_obstacles(env, config)

            # 生成路径
            for sample in range(config["num_samples_per_env"]):
                problem = env.set_random_init_goal()
                if problem["start"] is None or problem["goal"] is None:
                    continue
                start, goal = problem["start"], problem["goal"]

                planner = BITStar(start=start, goal=goal, environment=env,
                                  iter_max=1000, batch_size=config.get("batch_size", 200), pc_n_points=config.get("pc_n_points", 2048),
                                  plot_flag=False)
                planner.planning(visualize=False)
                path = planner.get_best_path()

                if path is None or len(path) == 0:
                    continue

                path_list.append(path)
                start_list.append(start)
                goal_list.append(goal)

            if path_list:  # 至少生成一条路径
                # print(f"[Env {env_idx}] completed")
                env_dict = {
                    "env_idx": env_idx,
                    "config_dim": env.config_dim,
                    "bound": env.bound.tolist(),
                    "start": [s.tolist() for s in start_list],
                    "goal": [g.tolist() for g in goal_list],
                    "paths": [np.array(p).tolist() for p in path_list],
                    "obstacles": obstacles
                }
                env.close()
                return env_dict

            # 如果没有生成路径，重新循环
            env.close()

        except Exception as e:
            # print(f"[Env {env_idx}] 生成失败: {e}")
            try:
                env.close()
            except:
                pass
            # 继续循环，保持无限重试

# ---------------- 数据集生成（并行 + 自动保证数量） ----------------
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os, json, numpy as np
from os.path import join

def generate_env_dataset_parallel(config):
    env_type = config.get("env_type", "kuka")
    target_sizes = {
        "train": config["train_env_size"],
        "val": config["val_env_size"],
        "test": config["test_env_size"],
    }

    num_workers = max(1, min(cpu_count(), config.get("num_workers", cpu_count())))
    print(f"🧩 使用 {num_workers} 个并行进程")

    for mode in ["test"]:
        data_dir = join("data", env_type, mode)
        os.makedirs(data_dir, exist_ok=True)
        path_dir = join(data_dir, "paths")
        os.makedirs(path_dir, exist_ok=True)

        env_list = [None] * target_sizes[mode]  # 预分配列表，保持顺序
        target_num = target_sizes[mode]
        success_count = 0

        print(f"\n=== 开始生成 [{mode}] 数据集，目标数量：{target_num} ===")
        pbar = tqdm(total=target_num)

        # 构建任务列表
        tasks = [(idx, config) for idx in range(target_num)]

        with Pool(processes=num_workers) as pool:
            for env_dict in pool.imap_unordered(generate_single_env, tasks):
                env_idx = env_dict["env_idx"]
                env_list[env_idx] = env_dict
                success_count += 1
                pbar.update(1)

                # 保存路径
                for i, path in enumerate(env_dict["paths"]):
                    np.savetxt(join(path_dir, f"{env_idx}_{i}.txt"),
                               np.array(path), fmt="%.4f", delimiter=",")

        # 生成完成后统一保存 JSON
        with open(join(data_dir, "envs.json"), "w") as f:
            json.dump(env_list, f, indent=2)

        pbar.close()
        print(f"[{mode}] ✅ 生成完成，共 {success_count} 个有效环境")

# ---------------- 主函数 ----------------
if __name__ == "__main__":
    config = {
        "env_type": "liche",  # "kuka" 或 "ur5" 或 "liche"
        "train_env_size": 5,
        "val_env_size": 2,
        "test_env_size": 100,
        "num_samples_per_env": 5,
        "batch_size": 200,
        "GUI": False,
        "num_workers": 4,  # 可根据CPU调整
        # 随机障碍物配置
        "xyz_max": [4, 4, 2],
        "box_size_range": [0.1, 1],
        "ball_radius_range": [0.1, 0.2],
        "num_boxes_range": [1, 8],
        "num_balls_range": [0, 5],
    }

    generate_env_dataset_parallel(config)
