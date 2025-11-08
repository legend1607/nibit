import os
import json
from matplotlib import pyplot as plt
import numpy as np
from os.path import join
from copy import copy
from types import SimpleNamespace

# === 导入 UR5 环境 ===
from environment.ur5_env import UR5Env
from environment.liche_env import LicheEnv


# --------------------------------------------------
# Step 1. 加载环境配置
# --------------------------------------------------
def get_env_configs(args):
    """
    加载 UR5 数据集的环境配置。

    每个 env_dict 对应一个场景，其中有多个 start-goal 对。
    这里展开成多个具体环境实例。
    """
    env_type = args.env_type
    root_dir=join("data", env_type)
    env_json_path = join(root_dir, "test", "envs.json")
    if not os.path.exists(env_json_path):
        raise FileNotFoundError(f"Cannot find envs.json at {env_json_path}")

    with open(env_json_path, 'r') as f:
        env_list = json.load(f)

    env_config_list = []
    for env_idx, env_dict_per_scene in enumerate(env_list):
        for pair_idx in range(len(env_dict_per_scene['start'])):
            env_cfg = {
                "env_idx": env_idx,
                "pair_idx": pair_idx,
                "env_dict": copy(env_dict_per_scene)
            }
            # 仅保留该 start-goal 对
            env_cfg["env_dict"]["start"] = [env_dict_per_scene["start"][pair_idx]]
            env_cfg["env_dict"]["goal"] = [env_dict_per_scene["goal"][pair_idx]]
            env_config_list.append(env_cfg)
    return env_config_list


# --------------------------------------------------
# Step 2. 根据配置创建 UR5Env 环境对象
# --------------------------------------------------
def get_problem(args,env_config, GUI=True):
    """
    根据配置生成机械臂环境与规划问题输入。
    """
    env_type = args.env_type
    env_dict = env_config["env_dict"]
    if env_type == "ur5":
        env = UR5Env(GUI=GUI)  # 也可设 GUI=False 用于批量测试
    elif env_type == "liche":
        env = LicheEnv(GUI=GUI)

    # 1️⃣ 添加障碍物
    for obs in env_dict["obstacles"]:
        shape, size, pos = obs
        if shape == "box":
            env.add_box_obstacle(size, pos)
        elif shape == "sphere":
            env.add_sphere_obstacle(size[0], pos)

    # 2️⃣ 设置起点与终点
    pair_idx = np.random.randint(len(env_dict["start"]))
    start = np.array(env_dict["start"][pair_idx])
    goal = np.array(env_dict["goal"][pair_idx])

    problem = {}
    problem['start'] = start
    problem['goal'] = goal
    problem['env_dict'] = env_dict
    problem['env'] = env
    return problem


# --------------------------------------------------
# Step 3. 主程序：运行路径规划
# --------------------------------------------------
if __name__ == "__main__":
    # === 参数定义 ===
    args = SimpleNamespace(
        env_type="liche",  # "ur5" 或 "liche"
        iter_max=500,
        batch_size=200,# 每批采样数量
        pc_n_points=100,       
        path_planner="BITStar",  # 可扩展成 RRT*, PRM*, FMT* 等
        visualize=True,
        neural_wrapper="None",  # "None" 或 "PointNet"
        device="cuda",
    )

    # === 加载环境配置 ===
    env_cfgs = get_env_configs(args)
    env_idx = np.random.randint(len(env_cfgs))
    env_idx = 358  # 你可以随机选一个，例如 np.random.randint(len(env_cfgs))
    print(f"选中的环境编号: {env_idx}")

    # === 构造 problem ===
    problem = get_problem(args,env_cfgs[env_idx], GUI=args.visualize)


    # === 创建路径规划器 ===
    print("使用的路径规划器:", args.path_planner)
    if args.neural_wrapper == "None":
        neural_wrapper = None  # 若无神经网络辅助，可设为 None
    if args.neural_wrapper == "PointNet" or args.path_planner in ["NBITStar"]:
        from neural_wrappers.pointnet2tf_wrapper import PNGWrapper
        neural_wrapper = PNGWrapper(coord_dim=4,device=args.device)
    if args.path_planner == "BITStar":
        from path_planning_classes_arm.bit_star import get_bit_planner
        planner = get_bit_planner(args, problem, neural_wrapper=None)
    elif args.path_planner == "NBITStar":
        from path_planning_classes_arm.nbit_star import get_bit_planner
        planner = get_bit_planner(args, problem, neural_wrapper=neural_wrapper)

    # === 执行规划 ===
    print("开始路径规划...")
    path, samples, edges, n_checks, best_cost, total_samples, runtime,total_iter,cost_list = planner.planning(visualize=args.visualize)

    print(f"\n✅ 规划完成")
    print(f"- 碰撞检测次数: {n_checks}")
    print(f"- 最优路径代价: {best_cost:.3f}")
    print(f"- 迭代次数: {total_iter}")
    print(f"- 总采样点数: {total_samples}")
    print(f"- 运行时间: {runtime:.2f} 秒")

    # 假设 cost_list 对应每次迭代的代价
    plt.figure(figsize=(8, 5))
    plt.plot(cost_list, marker='o', linestyle='-', color='blue', label='Path Cost per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Path Cost')
    plt.title('Iteration vs Path Cost')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(len(cost_list)))
    plt.tight_layout()
    plt.savefig('iterationvscost.jpg', dpi=300)
    path_array = np.array(path)  # shape: (num_points, 4)
    num_points = path_array.shape[0]

    plt.figure(figsize=(8, 5))
    for joint_idx in range(4):
        plt.plot(range(num_points), path_array[:, joint_idx], marker='o', label=f'Joint {joint_idx+1}')

    plt.xlabel('Path Point Index')
    plt.ylabel('Joint Angle (rad)')
    plt.title('Joint Space Trajectory')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存为 jpg
    plt.savefig('path.jpg', dpi=300)
    # === 可视化路径 ===
    if path is not None and len(path) > 1:
        problem["env"].render_path(path, gradient=True)
        input("按回车键退出...")
    else:
        print("未找到可行路径。")

    problem["env"].close()
