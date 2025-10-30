import json
import os
from os.path import join
import cv2
import numpy as np
from copy import copy
import argparse
from types import SimpleNamespace

# === 导入必要模块 ===
from environment.random_2d_env import Random2DEnv
from environment.random_2d_dy_env import Random2DKinodynamicEnv


# --------------------------------------------------
# Step 1. 获取环境配置（这里你可以自己定义 get_env_configs）
# --------------------------------------------------
def get_env_configs(root_dir='data/random_2d'):
    """
    加载随机2D环境配置。

    每张地图有多个 start-goal 对，将其展开成多个独立环境配置。
    例如 map_0 有4个pair，则输出4个配置项。

    Args:
        root_dir (str): 数据集根路径，默认 'data/random_2d'

    Returns:
        list[dict]: 每个元素对应一个具体的环境配置。
    """
    env_json_path = join(root_dir, "test", "envs.json")
    if not os.path.exists(env_json_path):
        raise FileNotFoundError(f"Cannot find envs.json at {env_json_path}")

    with open(env_json_path, 'r') as f:
        random_2d_map_list = json.load(f)

    env_config_list = []
    for map_idx, env_dict_per_map in enumerate(random_2d_map_list):
        for start_goal_pair_idx in range(len(env_dict_per_map['start'])):
            env_config = {}
            env_config['img_idx'] = map_idx
            env_config['start_goal_idx'] = start_goal_pair_idx

            # 深拷贝该地图的字典，保留当前 pair
            env_config['env_dict'] = copy(env_dict_per_map)
            env_config['env_dict']['start'] = [env_dict_per_map['start'][start_goal_pair_idx]]
            env_config['env_dict']['goal'] = [env_dict_per_map['goal'][start_goal_pair_idx]]
            env_config_list.append(env_config)
    return env_config_list


def get_problem_input(path_planner,random_2d_env_config):
    """
    根据给定的环境对象和索引编号，生成并保存一个 problem 字典。
    problem 是路径规划算法（BIT*, RRT* 等）的标准输入格式。
    """
    env_img = cv2.imread(join("data", "random_2d", "test", "env_imgs", "{0}.png".format(random_2d_env_config['img_idx'])))
    env_dict = random_2d_env_config['env_dict']
    start = tuple(env_dict['start'][0])
    goal = tuple(env_dict['goal'][0])
    if path_planner in ["BITStar","RRTStar","IRRTStar"]:
        env=Random2DEnv(env_dict, mode="test")
    elif path_planner=="KinoBITStar":
        env=Random2DKinodynamicEnv(env_dict, mode="test")
    problem = {}
    problem['start'] = start
    problem['goal'] = goal
    problem['env_dict'] = env_dict
    problem['env'] = env
    return problem

# --------------------------------------------------
# Step 2. 主程序
# --------------------------------------------------
if __name__ == "__main__":
    # 创建参数对象（你可以用 argparse 或手动定义）
    args = SimpleNamespace(
        iter_max=1000,          # 最大迭代次数
        pc_n_points=100,       # 每批采样数量
        env_resolution=1.0,    # 环境分辨率
        path_planner="BITStar", # 路径规划器名称KinoBITStar BITStar IRRTStar
        step_len=10,
        clearance = 3
    )

    # === 加载环境配置 ===
    env_config_list = get_env_configs()

    # 随机选择一个环境
    env_config_index = np.random.randint(len(env_config_list))
    # 也可以手动指定
    env_config_index = 200
    print("选中的环境编号:", env_config_index)

    # === 构造 problem 输入 ===
    problem = get_problem_input(args.path_planner,env_config_list[env_config_index])

    # === 创建路径规划器 ===
    print("使用的路径规划器:", args.path_planner)
    neural_wrapper = None  # 若无神经网络辅助，可设为 None

    if args.path_planner == "BITStar":
        from path_planning_classes.bit_star import get_bit_planner
        path_planner = get_bit_planner(args, problem, neural_wrapper)
    elif args.path_planner == "KinoBITStar":
        from kino_planning_classes.bit_star import get_bit_planner
        path_planner = get_bit_planner(args, problem, neural_wrapper)
    elif args.path_planner == "Kinofmt":
        from kino_planning_classes.fmt import get_planner as get_planner
        path_planner = get_planner(args, problem, neural_wrapper)
    elif args.path_planner == "RRTStar":
        from path_planning_classes.rrt_star_2d import get_path_planner
        path_planner = get_path_planner(args, problem, neural_wrapper)
    elif args.path_planner == "IRRTStar":
        from path_planning_classes.irrt_star_2d import get_path_planner
        path_planner = get_path_planner(args, problem, neural_wrapper)

    # === 执行规划 ===
    print("开始路径规划...")
    if args.path_planner in ["BITStar"]:
        path,samples, edges, n_checks, best_cost, total_samples, runtime = path_planner.planning(visualize=True)

        print(f"规划完成 ✅")
        print(f"- 碰撞检测次数: {n_checks}")
        print(f"- 最优路径代价: {best_cost:.2f}")
        print(f"- 总采样点数: {total_samples}")
        print(f"- 运行时间: {runtime:.2f} 秒")
    elif args.path_planner in ["IRRTStar","RRTStar"]:
        path_len_list = path_planner.planning(visualize=True)
    else:
        path_planner.planning()
        path_planner.plot()
