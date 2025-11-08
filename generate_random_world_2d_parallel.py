import os
import json
import time
import random
import numpy as np
from os.path import join
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from environment.random_2d_env import Random2DEnv
from path_planning_classes.bit_star import BITStar  # 使用 2D BIT* 或 NBIT* 规划器

# ---------------- 随机障碍物生成 ----------------
def add_random_obstacles_2d(env, config):
    """
    在 2D 环境中生成随机矩形和圆形障碍
    """
    obstacles = []

    # 矩形
    for _ in range(random.randint(*config["num_boxes_range"])):
        w, h = random.uniform(*config["box_size_range"]), random.uniform(*config["box_size_range"])
        x = random.uniform(0, env.bound[1][0] - w)
        y = random.uniform(0, env.bound[1][1] - h)
        env.rect_obstacles.append([x, y, w, h])
        obstacles.append(("rect", [x, y, w, h]))

    # 圆形
    for _ in range(random.randint(*config["num_balls_range"])):
        r = random.uniform(*config["ball_radius_range"])
        x = random.uniform(r, env.bound[1][0] - r)
        y = random.uniform(r, env.bound[1][1] - r)
        env.circle_obstacles.append([x, y, r])
        obstacles.append(("circle", [x, y, r]))

    return obstacles

# ---------------- 单环境生成 ----------------
def generate_single_env(args):
    env_idx, config = args

    while True:
        path_list, start_list, goal_list = [], [], []
        try:
            # 初始化环境
            env = Random2DEnv({
                "env_dims": config.get("env_dims", [10, 10]),
                "rectangle_obstacles": [],
                "circle_obstacles": []
            })

            # 添加随机障碍
            add_random_obstacles_2d(env, config)

            # 生成路径
            for _ in range(config["num_samples_per_env"]):
                problem = env.set_random_init_goal()
                start, goal = problem["start"], problem["goal"]

                if start is None or goal is None:
                    continue

                planner = BITStar(start=start, goal=goal, environment=env,
                                   iter_max=config.get("iter_max", 500),
                                   batch_size=config.get("batch_size", 200),
                                   pc_n_points=config.get("pc_n_points", 500))
                planner.planning(visualize=False)
                path = planner.get_best_path()

                if path is None or len(path) == 0:
                    continue

                path_list.append(path)
                start_list.append(start)
                goal_list.append(goal)

            if path_list:
                env_dict = {
                    "env_idx": env_idx,
                    "config_dim": env.config_dim,
                    "bound": env.bound,
                    "env_dims": [env.bound[1][0], env.bound[1][1]],   # ✅ 新增
                    "rectangle_obstacles": env.rect_obstacles,         # ✅ 新增
                    "circle_obstacles": env.circle_obstacles,          # ✅ 新增
                    "start": [s.tolist() for s in start_list],
                    "goal": [g.tolist() for g in goal_list],
                    "paths": path_list
                }
                return env_dict

        except Exception as e:
            continue

# ---------------- 数据集生成 ----------------
def generate_env_dataset_parallel(config):
    env_type = config.get("env_type", "random_2d")
    target_sizes = {
        "train": config.get("train_env_size", 10),
        "val": config.get("val_env_size", 5),
        "test": config.get("test_env_size", 10),
    }

    num_workers = max(1, min(cpu_count(), config.get("num_workers", cpu_count())))
    print(f"🧩 使用 {num_workers} 个并行进程")

    for mode in ["train", "val", "test"]:
        data_dir = join("data", env_type, mode)
        os.makedirs(data_dir, exist_ok=True)
        path_dir = join(data_dir, "paths")
        os.makedirs(path_dir, exist_ok=True)

        env_list = [None] * target_sizes[mode]
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

        # 保存 JSON 文件
        with open(join(data_dir, "envs.json"), "w") as f:
            json.dump(env_list, f, indent=2)

        pbar.close()
        print(f"[{mode}] ✅ 生成完成，共 {success_count} 个有效环境")

# ---------------- 主函数 ----------------
if __name__ == "__main__":
    config = {
        "env_type": "random_2d",
        "train_env_size": 200,
        "val_env_size": 25,
        "test_env_size": 25,
        "num_samples_per_env": 5,
        "batch_size": 200,
        "iter_max": 200,
        "env_dims": [224, 224],
        "num_workers": 4,
        "num_boxes_range": [5, 10],
        "num_balls_range": [0, 10],
        "box_size_range": [16, 24],
        "ball_radius_range": [16, 24],
    }

    generate_env_dataset_parallel(config)
