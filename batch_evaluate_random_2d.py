import os
import json
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from types import SimpleNamespace
from neural_wrappers.pointnet2_wrapperce import PNGWrapper


# --------------------------------------------------
# Step 1. 加载随机2D环境配置
# --------------------------------------------------
def get_env_configs(root_dir="data/random_2d"):
    env_json_path = os.path.join(root_dir, "test", "envs.json")
    if not os.path.exists(env_json_path):
        raise FileNotFoundError(f"Missing {env_json_path}")

    with open(env_json_path, "r") as f:
        env_list = json.load(f)

    env_cfgs = []
    for map_idx, env_dict_per_map in enumerate(env_list):
        for pair_idx in range(len(env_dict_per_map["start"])):
            cfg = {
                "map_idx": map_idx,
                "pair_idx": pair_idx,
                "env_dict": copy(env_dict_per_map)
            }
            cfg["env_dict"]["start"] = [env_dict_per_map["start"][pair_idx]]
            cfg["env_dict"]["goal"] = [env_dict_per_map["goal"][pair_idx]]
            env_cfgs.append(cfg)
    return env_cfgs


# --------------------------------------------------
# Step 2. 创建问题对象
# --------------------------------------------------
def get_problem(args, env_cfg):
    env_dict = env_cfg["env_dict"]
    start = tuple(env_dict["start"][0])
    goal = tuple(env_dict["goal"][0])

    if args.path_planner in ["BITStar", "NBITStar", "RRTStar", "IRRTStar"]:
        from environment.random_2d_env import Random2DEnv
    elif args.path_planner == "KinoBITStar":
        from kino_environment.random_2d_env import Random2DEnv
    else:
        raise ValueError(f"Unknown planner type: {args.path_planner}")

    env = Random2DEnv(env_dict, mode="test")
    return {"start": start, "goal": goal, "env_dict": env_dict, "env": env}


# --------------------------------------------------
# Step 3. 批量运行两算法（BITStar vs NBITStar）
# --------------------------------------------------
def batch_compare_algorithms(args, n_trials=30):
    env_cfgs = get_env_configs()
    algos = ["BITStar", "NBITStar"]

    algo_data = {
        algo: {"cost_lists": [], "ifs_list": [], "iter_counts": []}
        for algo in algos
    }

    for trial in range(n_trials):
        env_cfg = env_cfgs[np.random.randint(len(env_cfgs))]
        print(f"\n=== Trial {trial+1}/{n_trials}: map {env_cfg['map_idx']} pair {env_cfg['pair_idx']} ===")

        for algo in algos:
            args.path_planner = algo
            print(f"→ {algo} running ...")
            problem = get_problem(args, env_cfg)

            if algo == "BITStar":
                from path_planning_classes.bit_star import get_bit_planner
                neural_wrapper = None
            else:
                from path_planning_classes.nbit_star import get_bit_planner
                neural_wrapper = PNGWrapper(coord_dim=2, device=args.device)

            planner = get_bit_planner(args, problem, neural_wrapper=neural_wrapper)

            try:
                path, samples, edges, n_checks, best_cost, total_samples, runtime, total_iter, cost_list = \
                    planner.planning(visualize=False)
                print(f"{algo} | cost_list len = {len(cost_list)}, best_cost = {best_cost}")

                cost_arr = np.array(cost_list)
                if len(cost_arr) < args.iter_max:
                    # 用最后一个有限代价填充，而不是 inf
                    last_val = cost_arr[-1] if len(cost_arr) > 0 else np.inf
                    pad = np.full(args.iter_max - len(cost_arr), last_val)
                    cost_arr = np.concatenate([cost_arr, pad])
                # Iteration to first success (IFS)
                if np.any(np.isfinite(cost_arr)):
                    ifs = np.where(np.isfinite(cost_arr))[0][0]
                else:
                    ifs = np.nan

                algo_data[algo]["cost_lists"].append(cost_arr)
                algo_data[algo]["ifs_list"].append(ifs)
                algo_data[algo]["iter_counts"].append(total_iter)

                print(f"   ✓ {algo} done | cost={best_cost:.3f}, IFS={ifs}, iters={total_iter}, time={runtime:.2f}s")

            except Exception as e:
                print(f"   ✗ {algo} failed: {e}")
                algo_data[algo]["cost_lists"].append(np.full(args.iter_max, np.inf))
                algo_data[algo]["ifs_list"].append(np.nan)
                algo_data[algo]["iter_counts"].append(np.nan)

            # problem["env"].close()

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
    plt.savefig("success_rate_vs_iterations_random2d.jpg", dpi=300)

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
    plt.savefig("path_cost_vs_iterations_random2d.jpg", dpi=300)

    # 3️⃣ IFS Distribution
    plt.figure(figsize=(8, 5))
    for algo, color in zip(algos, colors):
        ifs_values = [x for x in algo_data[algo]["ifs_list"] if not np.isnan(x)]
        plt.hist(ifs_values, bins=20, alpha=0.6, label=algo, color=color)
    plt.xlabel("Iteration to First Success (IFS)")
    plt.ylabel("Frequency")
    plt.title("IFS Distribution (Random2D)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ifs_distribution_random2d.jpg", dpi=300)

    # 4️⃣ Planning Iteration Boxplot
    plt.figure(figsize=(7, 5))
    iter_data = [algo_data[a]["iter_counts"] for a in algos]
    plt.boxplot(iter_data, labels=algos)
    plt.ylabel("Planning Iterations")
    plt.title("Planning Iteration Boxplot (Random2D)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("planning_iteration_boxplot_random2d.jpg", dpi=300)


# --------------------------------------------------
# 主函数
# --------------------------------------------------
if __name__ == "__main__":
    args = SimpleNamespace(
        iter_max=100,
        batch_size=200,
        pc_n_points=500,
        device="cuda",
        path_planner="NBITStar",
        step_len=10,
        clearance=3
    )

    n_trials = 100
    algo_data = batch_compare_algorithms(args, n_trials=n_trials)
    plot_statistics(algo_data, args)
    print("✅ Random2D 两算法对比完成，结果已保存为图片。")
