import os
import json
import argparse
from os.path import join

import numpy as np
import torch
import matplotlib.pyplot as plt

# ----------------- 导入环境类 -----------------
from environment.liche_env import LicheEnv
import numpy as np
import torch

from model.encoders.joint_pointlite_encoder import JointPointNetEncoder

# ------------------------------
# 从 generate_random_world_arm_pc.py 复用体素生成函数
# ------------------------------
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

        grid[idx_lo[0]:idx_hi[0] + 1,
             idx_lo[1]:idx_hi[1] + 1,
             idx_lo[2]:idx_hi[2] + 1] = 1

    return grid

# ------------------------------
# NeuralWrapper 主体
# ------------------------------
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 1. 读取 envs.json（由 generate_random_world_arm parallel.py 生成）
# ============================================================
def get_env_configs(root_dir: str = "data/liche", split: str = "test"):
    """
    读取 data/liche/{split}/envs.json 并返回环境列表。

    每个环境是 generate_random_world_arm parallel.py 中 generate_single_env
    返回的 dict，形如：
    {
        "ok": True,
        "straight_flags": [...],
        "env_range": [[-7, 7], [-7, 7], [-2, 5]],
        "pose_range": [...],
        "start": [[...], [...], ...],
        "goal": [[...], [...], ...],
        "paths": [...],
        "obstacles": [
            ["box", half_extents, pos],
            ...
        ],
        "num_obstacles": int,
        "prob_accept": float
    }
    """
    env_json_path = join(root_dir, split, "envs.json")
    if not os.path.exists(env_json_path):
        raise FileNotFoundError(f"Cannot find envs.json at {env_json_path}")

    with open(env_json_path, "r") as f:
        env_list = json.load(f)

    # 保险起见，只保留 ok == True 的
    env_list = [e for e in env_list if e.get("ok", True)]

    if len(env_list) == 0:
        raise RuntimeError(f"envs.json 中没有可用环境: {env_json_path}")

    print(f"[INFO] 加载 {env_json_path} 成功，共 {len(env_list)} 个环境。")
    return env_list


# ============================================================
# 2. 从记录中重建 PyBullet 环境（LicheEnv）
# ============================================================
def rebuild_env_from_record(env_record: dict, gui: bool = False) -> LicheEnv:
    """
    根据 envs.json 里的单个环境记录，重建 LicheEnv（只重建障碍物）：
      - 重新创建一个 LicheEnv(GUI=gui)
      - 按 obstacles 列表把 box 障碍物加回去
    """
    env = LicheEnv(GUI=gui)

    obstacles = env_record.get("obstacles", [])
    for ob in obstacles:
        if not ob or len(ob) < 3:
            continue
        ob_type = ob[0]
        if ob_type == "box":
            half_extents = ob[1]
            pos = ob[2]
            env.add_box_obstacle(half_extents, pos)
        # 如果以后有球体等类型，可以在这里扩展

    return env


# ============================================================
# 3. 构造 BITStar / NIBITStar 所需的 problem
# ============================================================
def get_problem_input(
    path_planner: str,
    env_record: dict,
    traj_index: int = 0,
    gui: bool = False,
) -> dict:
    """
    将 envs.json 中的一个 env_record + 第 traj_index 条路径
    转换为 BITStar/NIBITStar 需要的 problem = {start, goal, env_dict, env}。

    - env_record["start"] / ["goal"] 是列表（多条路径）
    - traj_index 选第几条（默认 0，如果越界自动 clip 到 [0, n-1]）
    """
    starts = env_record["start"]
    goals = env_record["goal"]

    if not starts or not goals:
        raise ValueError("env_record 中 start/goal 为空，无法构造问题。")

    n_traj = min(len(starts), len(goals))
    if n_traj == 0:
        raise ValueError("env_record 中无可用 start/goal 对。")

    if traj_index < 0 or traj_index >= n_traj:
        traj_index = 0

    start = tuple(starts[traj_index])
    goal = tuple(goals[traj_index])

    env = rebuild_env_from_record(env_record, gui=gui)

    problem = {
        "start": start,
        "goal": goal,
        "env_dict": env_record,
        "env": env,
    }
    return problem


# ============================================================
# 4. 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="LICHE 机械臂路径规划 Demo（使用生成的 envs.json + BITStar/NIBITStar）"
    )

    # 数据集路径
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/liche",
        help="数据集根目录（包含 train/val/test 子目录），默认 data/liche",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="使用哪个数据子集（train/val/test），默认 test",
    )

    # 规划器类型
    parser.add_argument(
        "--planner",
        dest="path_planner",
        type=str,
        default="NIBITStar",
        choices=["BITStar", "NIBITStar"],
        help="路径规划器类型：BITStar 或 NIBITStar，默认 BITStar",
    )

    # BIT* 参数
    parser.add_argument(
        "--iter_max",
        type=int,
        default=1000,
        help="BIT* 最大迭代次数（iter_max），默认 1000",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="每轮新增采样数量（batch_size），默认 200",
    )

    # 选择哪个环境 / 哪条路径
    parser.add_argument(
        "--env_idx",
        type=int,
        default=5,
        help="使用第几个环境（0-based），默认随机选择",
    )
    parser.add_argument(
        "--traj_idx",
        type=int,
        default=-1,
        help="使用该环境里的第几条路径（0-based），默认随机选择",
    )

    # 是否开 GUI
    parser.add_argument(
        "--gui",
        action="store_true",
        help="是否启用 PyBullet GUI（默认关闭）",
    )

    args = parser.parse_args()
    return args


# ============================================================
# 5. 主程序
# ============================================================
def main():
    args = parse_args()

    # 1) 读取 envs.json
    env_list = get_env_configs(root_dir=args.data_root, split=args.split)

    # 2) 选一个环境
    if args.env_idx < 0 or args.env_idx >= len(env_list):
        env_idx = np.random.randint(0, len(env_list))
    else:
        env_idx = args.env_idx

    env_record = env_list[env_idx]

    # 3) 选该环境中的第几条路径
    n_traj = len(env_record.get("start", []))
    if n_traj == 0:
        raise RuntimeError(f"环境 {env_idx} 中没有任何 start/goal。")

    if args.traj_idx < 0 or args.traj_idx >= n_traj:
        traj_idx = np.random.randint(0, n_traj)
    else:
        traj_idx = args.traj_idx

    print("======================================")
    print(f" 数据子集: {args.split}")
    print(f" 环境编号: {env_idx} / {len(env_list)-1}")
    print(f" 轨迹编号: {traj_idx} / {n_traj-1}")
    print(f" 规划器:   {args.path_planner}")
    print("======================================")

    # 4) 构造 problem
    problem = get_problem_input(
        path_planner=args.path_planner,
        env_record=env_record,
        traj_index=traj_idx,
        gui=args.gui,
    )

    # 5) 创建规划器实例
    neural_wrapper = None  # 如果你有网络，可以在这里传进去

    print("[INFO] 创建路径规划器...")
    if args.path_planner == "BITStar":
        from path_planning_classes_arm.bit_star import get_bit_planner
        path_planner = get_bit_planner(args, problem, neural_wrapper)

    elif args.path_planner == "NIBITStar":
        from path_planning_classes_arm.nibit_star_fixed import get_bit_planner
        from neural_wrapper import NeuralWrapper
        # 创建 NeuralWrapper：自动构建体素、加载模型、算 env_feat
        neural_wrapper = NeuralWrapper(
            problem=problem,
            ckpt_path="results/model_training/train_20251215-144528/best.pt",
            voxel_resolution=(50,50,50),
            device="cuda"
        )

        path_planner = get_bit_planner(args, problem, neural_wrapper)

    else:
        raise ValueError(f"未知的路径规划器类型: {args.path_planner}")

    # 6) 开始规划
    print("[INFO] 开始路径规划...")
    (
        path,
        samples,
        edges,
        n_checks,
        best_cost,
        total_samples,
        runtime,
        final_iter,
        iteration_costs,
    ) = path_planner.planning(visualize=True)

    # 7) 打印结果
    print("\n========== 规划结果 ==========")
    has_path = path is not None and len(path) > 0
    print(f"- 路径是否找到: {'是' if path is not None and len(path) > 0 else '否'}")
    print(f"- 路径点数:     {len(path) if path is not None else 0}")
    print(f"- 碰撞检测次数: {n_checks}")
    print(f"- 最优路径代价: {best_cost:.6f}")
    print(f"- 总采样点数 T: {total_samples}")
    print(f"- 迭代次数:     {final_iter + 1}")
    print(f"- 运行时间:     {runtime:.2f} 秒")
    print("======================================")
    # 7.5) 绘制迭代-代价值曲线
    if iteration_costs is not None and len(iteration_costs) > 0:
        iters = list(range(1, len(iteration_costs) + 1))

        plt.figure()
        plt.plot(iters, iteration_costs, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Best cost so far")
        plt.title(
            f"{args.path_planner} iteration cost (env {env_idx}, traj {traj_idx})"
        )
        plt.grid(True)

        # 保存到文件
        out_dir = os.path.join("results", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"iter_cost_env{env_idx}_traj{traj_idx}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] 迭代曲线已保存到: {out_path}")

        # 如果你是在本地跑，有图形界面，也可以直接弹窗显示
        # plt.show()
    else:
        print("[WARN] iteration_costs 为空，无法绘制迭代曲线。")

    # 8) 用 env.render_path 可视化路径
    if has_path:
        print("[INFO] 使用 env.render_path 可视化路径（需要 --gui 才能看到窗口）...")
        render_env = rebuild_env_from_record(env_record, gui=False)
        # 你可以根据需要调参数，比如渐变颜色 / 显示机器人 / 插值步长等
        render_env.render_path(
            path,
            gradient=True,       # 颜色渐变
            show_robots=True,    # 沿路径显示机械臂姿态
            pose_interval=5,     # 每隔多少个插值点放一个机械臂
            sleep_interval=0.03, # 动画速度
            interp_step=0.05,    # 插值步长，越小越平滑
            save_curve=True,
            save_file="traj_001.csv"
        )
    else:
        print("[WARN] 未找到路径，无法调用 env.render_path。")
    # 如需更详细的分析，比如 iteration_costs 曲线，可以自己在这里画图


if __name__ == "__main__":
    main()
