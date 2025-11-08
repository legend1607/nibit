import os
import json
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from types import SimpleNamespace

# === 导入环境类 ===
from environment.ur5_env import UR5Env
from environment.liche_env import LicheEnv
from neural_wrappers.pointnet2_wrapperce import PNGWrapper


# --------------------------------------------------
# Step 1. 加载环境配置
# --------------------------------------------------
def get_env_configs(args):
    env_type = args.env_type
    root_dir = os.path.join("data", env_type)
    env_json_path = os.path.join(root_dir, "test", "envs.json")

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
            env_cfg["env_dict"]["start"] = [env_dict_per_scene["start"][pair_idx]]
            env_cfg["env_dict"]["goal"] = [env_dict_per_scene["goal"][pair_idx]]
            env_config_list.append(env_cfg)
    return env_config_list


# --------------------------------------------------
# Step 2. 创建环境
# --------------------------------------------------
def get_problem(args, env_config, GUI=False):
    env_type = args.env_type
    env_dict = env_config["env_dict"]

    if env_type == "ur5":
        env = UR5Env(GUI=GUI)
    else:
        env = LicheEnv(GUI=GUI)

    for obs in env_dict["obstacles"]:
        shape, size, pos = obs
        if shape == "box":
            env.add_box_obstacle(size, pos)
        elif shape == "sphere":
            env.add_sphere_obstacle(size[0], pos)

    start = np.array(env_dict["start"][0])
    goal = np.array(env_dict["goal"][0])

    return {"env": env, "start": start, "goal": goal, "env_dict": env_dict}


# --------------------------------------------------
# Step 3. 批量运行两算法
# --------------------------------------------------
def batch_compare_algorithms(args, n_trials=30):
    env_cfgs = get_env_configs(args)
    algos = ["BITStar", "NBITStar"]

    algo_data = {
        algo: {"cost_lists": [], "ifs_list": [], "iter_counts": []}
        for algo in algos
    }

    for trial in range(n_trials):
        env_cfg = env_cfgs[np.random.randint(len(env_cfgs))]
        print(f"\n=== Trial {trial+1}/{n_trials}: 环境 {env_cfg['env_idx']} pair {env_cfg['pair_idx']} ===")

        for algo in algos:
            print(f"→ {algo} running ...")
            problem = get_problem(args, env_cfg, GUI=False)

            if algo == "BITStar":
                from path_planning_classes_arm.bit_star import get_bit_planner
                neural_wrapper = None
            else:
                from path_planning_classes_arm.nbit_star import get_bit_planner
                neural_wrapper = PNGWrapper(coord_dim=4, device=args.device)

            planner = get_bit_planner(args, problem, neural_wrapper=neural_wrapper)

            try:
                path, samples, edges, n_checks, best_cost, total_samples, runtime, total_iter, cost_list = \
                    planner.planning(visualize=False)

                # pad cost_list
                cost_arr = np.array(cost_list)
                if len(cost_arr) < args.iter_max:
                    pad = np.full(args.iter_max - len(cost_arr), np.inf)
                    cost_arr = np.concatenate([cost_arr, pad])

                # Iteration to first success (IFS)
                if np.any(np.isfinite(cost_arr)):
                    ifs = np.where(np.isfinite(cost_arr))[0][0]
                else:
                    ifs = np.nan

                algo_data[algo]["cost_lists"].append(cost_arr)
                algo_data[algo]["ifs_list"].append(ifs)
                algo_data[algo]["iter_counts"].append(total_iter)

                print(f"   ✓ {algo} done | cost={best_cost:.3f}, IFS={ifs}, iters={total_iter}")

            except Exception as e:
                print(f"   ✗ {algo} failed: {e}")
                algo_data[algo]["cost_lists"].append(np.full(args.iter_max, np.inf))
                algo_data[algo]["ifs_list"].append(np.nan)
                algo_data[algo]["iter_counts"].append(np.nan)

            problem["env"].close()

    return algo_data


# --------------------------------------------------
# Step 4. 绘图函数
# --------------------------------------------------
def plot_statistics(algo_data, args):
    algos = list(algo_data.keys())
    colors = ["tab:blue", "tab:orange"]

    # 1️⃣ Success Rate vs Iterations
    plt.figure(figsize=(8, 5))
    for algo, color in zip(algos, colors):
        cost_matrix = np.array(algo_data[algo]["cost_lists"])
        success_rate = np.mean(np.isfinite(cost_matrix), axis=0)
        plt.plot(success_rate, label=algo, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("success_rate_vs_iterations_compare.jpg", dpi=300)

    # 2️⃣ Path Cost vs Iterations
    plt.figure(figsize=(8, 5))
    for algo, color in zip(algos, colors):
        cost_matrix = np.array(algo_data[algo]["cost_lists"])
        finite_mask = np.isfinite(cost_matrix)
        avg_cost = np.divide(
            np.sum(cost_matrix * finite_mask, axis=0),
            np.sum(finite_mask, axis=0),
            out=np.full(cost_matrix.shape[1], np.nan),
            where=np.sum(finite_mask, axis=0) > 0
        )
        plt.plot(avg_cost, label=algo, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Average Path Cost")
    plt.title("Average Path Cost vs Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("path_cost_vs_iterations_compare.jpg", dpi=300)

    # 3️⃣ IFS Distribution
    plt.figure(figsize=(8, 5))
    for algo, color in zip(algos, colors):
        ifs_values = [x for x in algo_data[algo]["ifs_list"] if not np.isnan(x)]
        plt.hist(ifs_values, bins=20, alpha=0.6, label=algo, color=color)
    plt.xlabel("Iteration to First Success (IFS)")
    plt.ylabel("Frequency")
    plt.title("IFS Distribution Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ifs_distribution_compare.jpg", dpi=300)

    # 4️⃣ Planning Iteration Boxplot
    plt.figure(figsize=(7, 5))
    iter_data = [algo_data[a]["iter_counts"] for a in algos]
    plt.boxplot(iter_data, labels=algos)
    plt.ylabel("Planning Iterations")
    plt.title("Planning Iteration Boxplot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("planning_iteration_boxplot_compare.jpg", dpi=300)


# --------------------------------------------------
# 主函数
# --------------------------------------------------
if __name__ == "__main__":
    args = SimpleNamespace(
        env_type="liche",
        iter_max=100,
        batch_size=200,
        pc_n_points=1000,
        device="cuda",
    )

    n_trials = 100
    algo_data = batch_compare_algorithms(args, n_trials=n_trials)
    plot_statistics(algo_data, args)
    print("✅ 两算法对比完成，包含 IFS 与箱线图。")
