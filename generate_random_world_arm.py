import os
import json
import time
import numpy as np
import pybullet as p
from os.path import join
from environment.kuka_env import KukaEnv
from environment.ur5_env import UR5Env
from environment.liche_env import LicheEnv
from path_planning_classes_arm.bit_star import BITStar
import random

# ---------------- 随机障碍物 ----------------
def add_random_obstacles(env, config):
    obstacles = []
    xyz_max = config.get("xyz_max", [5,5,5])
    x_range = [-xyz_max[0], xyz_max[0]]
    y_range = [-xyz_max[1], xyz_max[1]]
    z_max = xyz_max[2]

    # 随机方块
    for _ in range(random.randint(config["num_boxes_range"][0], config["num_boxes_range"][1])):
        size_x = random.uniform(config["box_size_range"][0], config["box_size_range"][1])
        size_y = random.uniform(config["box_size_range"][0], config["box_size_range"][1])
        size_z = random.uniform(config["box_size_range"][0], config["box_size_range"][1])
        half_extents = [size_x / 2, size_y / 2, size_z / 2]
        while True:
            pos = [
                random.uniform(*x_range),
                random.uniform(*y_range),
                random.uniform(half_extents[2], z_max)
            ]
            if abs(pos[0]) >= 0.1 and abs(pos[1]) >= 0.1 and abs(pos[2]) >= 0.1:
                break
        env.add_box_obstacle(half_extents, pos)
        obstacles.append(("box", half_extents, pos))

    # 随机球体
    for _ in range(random.randint(config["num_balls_range"][0], config["num_balls_range"][1])):
        radius = random.uniform(config["ball_radius_range"][0], config["ball_radius_range"][1])
        pos = [
            random.uniform(*x_range),
            random.uniform(*y_range),
            random.uniform(radius, z_max)
        ]
        env.add_sphere_obstacle(radius, pos)
        obstacles.append(("ball", radius, pos))

    return obstacles

# ---------------- 可视化 ----------------
def visualize_env_task(env, pause_time=3.0):
    if not env.GUI:
        return
    start_id = env.arm_id
    env.set_config(env.start, start_id)
    p.changeVisualShape(start_id, env.end_effector_index, rgbaColor=[0,1,0,1])

    goal_id = p.loadURDF(env.arm_file, [0,0,0],[0,0,0,1],useFixedBase=True)
    env.set_config(env.goal, goal_id)
    p.changeVisualShape(goal_id, env.end_effector_index, rgbaColor=[1,0,0,1])

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30,
                                 cameraPitch=-40, cameraTargetPosition=[0,0,0.5])
    t0 = time.time()
    while time.time() - t0 < pause_time:
        # p.stepSimulation()
        time.sleep(1/240.0)

# ---------------- 单环境生成 ----------------
def generate_single_env(env_idx, config):
    if config["env_type"] == "ur5":
        env = UR5Env(GUI=config.get("GUI", True))
    elif config["env_type"] == "kuka":
        env = KukaEnv(GUI=config.get("GUI", True))
    elif config["env_type"] == "liche":
        env = LicheEnv(GUI=config.get("GUI", True))
    path_list, start_list, goal_list = [], [], []

    # 随机障碍
    obstacles = add_random_obstacles(env, config)

    for num_sample in range(config["num_samples_per_env"]):
        problem = env.set_random_init_goal()
        if problem["start"] is None or problem["goal"] is None:
            env.close()
            print(f"[Env {env_idx}] 无效的起点或目标，放弃")
            return None
        start, goal = problem["start"], problem["goal"]

        if config.get("GUI", True):
            visualize_env_task(env, pause_time=1.0)

        # 使用 BITStar 规划
        planner = BITStar(start=start, goal=goal, environment=env,
                          iter_max=1000, batch_size=config.get("batch_size", 200), pc_n_points=config.get("pc_n_points", 2048),
                          plot_flag=False)
        print(f"{num_sample}")
        planner.planning(visualize=False)
        path = planner.get_best_path()

        if path is None or len(path) == 0:
            env.close()
            print(f"[Env {env_idx}] 无可行路径，放弃")
            continue
        if config.get("GUI", True):
            time.sleep(5)
            print("路径可视化")
            env.render_path(path, color=(0, 1, 0), life_time=0)
        path_list.append(path)
        start_list.append(start)
        goal_list.append(goal)

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

# ---------------- 数据集生成 ----------------
def generate_env_dataset_single(config):
    env_list = []
    env_type= config.get("env_type", "kuka")
    target_sizes = {
        "train": config["train_env_size"],
        "val": config["val_env_size"],
        "test": config["test_env_size"],
    }
    for mode in ["test"]:  
        data_dir = join("data", f"{env_type}", f"{mode}")
        os.makedirs(data_dir, exist_ok=True)
        path_dir = join(data_dir, "paths")
        os.makedirs(path_dir, exist_ok=True)

        env_list = []
        env_idx = 0

        print(f"开始生成 [{mode}] 环境目标数: {target_sizes[mode]}")

        # ✅ 持续尝试直到达到目标数量
        while env_idx < target_sizes[mode]:
            env_dict = generate_single_env(env_idx, config)

            if env_dict is None:
                continue  # 无效环境，重试

            env_list.append(env_dict)
            env_idx += 1  

            # 保存每个环境的数据
            for i, path in enumerate(env_dict["paths"]):
                np.savetxt(join(path_dir, f"{env_idx}_{i}.txt"),
                           np.array(path), fmt="%.4f", delimiter=",")

            with open(join(data_dir, "envs.json"), "w") as f:
                json.dump(env_list, f, indent=2)

            print(f"[{mode}] ✅ 已生成 {env_idx+1}/{target_sizes[mode]} 个环境")
    print(f"[{mode}] ✅ 完成，共 {len(env_list)} 个有效环境")

# ---------------- 主函数 ----------------
if __name__ == "__main__":
    config = {
        "env_type": "liche",  # "kuka" 或 "ur5" 或 "liche"
        "train_env_size": 5,
        "val_env_size": 2,
        "test_env_size": 20,
        "num_samples_per_env": 5,
        "batch_size": 200,
        "GUI": False,
        # 随机障碍物配置
        "xyz_max": [3, 3, 2],
        "box_size_range": [0.1, 1],
        "ball_radius_range": [0.1, 0.2],
        "num_boxes_range": [1, 8],
        "num_balls_range": [0, 5]
    }
    generate_env_dataset_single(config)
