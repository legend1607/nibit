#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# 复用你原来的工具函数 / 类
from demo_planning_arm import (
    get_env_configs,
    get_problem_input,
    NeuralWrapper,
)

from path_planning_classes_arm.bit_star import get_bit_planner as get_bit_planner_bit
from path_planning_classes_arm.nibit_star import get_bit_planner as get_bit_planner_nibit
from path_planning_classes_arm.irrt_star import get_irrt_planner as get_irrt_planner_irrt


# ---------------------------
# 构造单个 planner 实例
# ---------------------------
def build_planner(planner_name, args, problem, nw_cache):
    """
    根据 planner_name 创建对应的 planner。
    nw_cache 用于缓存 NeuralWrapper（避免同一个 env 反复创建）。
    """
    name = planner_name.upper()

    if name == "BITSTAR":
        return get_bit_planner_bit(args, problem, neural_wrapper=None)

    elif name == "NIBITSTAR":
        if "NIBITSTAR" not in nw_cache:
            nw_cache["NIBITSTAR"] = NeuralWrapper(
                problem=problem,
                ckpt_path=args.ckpt,
                voxel_resolution=tuple(args.voxel_resolution),
                device="cuda",
            )
        return get_bit_planner_nibit(args, problem, neural_wrapper=nw_cache["NIBITSTAR"])
    elif name == "IRRTSTAR":
        # IRRT* 不需要 NeuralWrapper，直接构造
        return get_irrt_planner_irrt(args, problem, neural_wrapper=None)

    else:
        raise ValueError(f"未知的 planner: {planner_name}")



# ---------------------------
# 跑一个任务上的一个 planner
# ---------------------------
def run_one_task_one_planner(planner_name, args, env_record, env_idx, traj_idx, gui=False):
    # 构造 problem
    problem = get_problem_input(
        path_planner=planner_name,
        env_record=env_record,
        traj_index=traj_idx,
        gui=gui,
    )

    # 为兼容 get_bit_planner
    args.path_planner = planner_name
    args.planner = planner_name

    # NeuralWrapper cache：同一 env / problem 内跨多个 planner 共用字典
    nw_cache = {}
    planner = build_planner(planner_name, args, problem, nw_cache)

    # 兼容：BITStar/NIBITStar 现在会多返回一些指标；其他 planner 仍然可以只返回 9 个值
    result = planner.planning(visualize=False)

    if len(result) < 9:
        raise RuntimeError(f"planner.planning 返回值数量异常: 期望至少9个, 实际 {len(result)}")

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
        *extras,
    ) = result

    has_path = path is not None and len(path) > 0

    # ---- 从 extras 中读取：iteration_times + 初始解信息 ----
    iteration_time_costs = []
    first_solution_iter = None
    first_solution_cost = None
    first_solution_nodes = None
    final_solution_nodes = None

    # 如果 extras[0] 是 list/ndarray，认为是 iteration_times
    if len(extras) >= 1 and isinstance(extras[0], (list, tuple, np.ndarray)):
        iteration_time_costs = list(extras[0])
        extras = extras[1:]
    else:
        iteration_time_costs = []

    if len(extras) >= 1:
        first_solution_iter = extras[0]
    if len(extras) >= 2:
        first_solution_cost = extras[1]
    if len(extras) >= 3:
        first_solution_nodes = extras[2]
    if len(extras) >= 4:
        final_solution_nodes = extras[3]

    # ---- 兼容性兜底：根据 iteration_costs 推断初始解迭代 / cost ----
    if iteration_costs is None:
        iteration_costs = []
    if len(iteration_costs) > 0:
        if first_solution_iter is None or not (first_solution_iter > 0):
            first_iter_idx = -1
            for i, c in enumerate(iteration_costs):
                if np.isfinite(c):
                    first_iter_idx = i
                    break
            if has_path and first_iter_idx == -1:
                # 有路径却没有有限 cost，退化：用最后一次迭代
                first_iter_idx = len(iteration_costs) - 1
            if first_iter_idx >= 0:
                first_solution_iter = first_iter_idx + 1
                if first_solution_cost is None:
                    first_solution_cost = float(iteration_costs[first_iter_idx])

    # 初始路径节点数如果 planner 没提供，用 -1 占位，后续统计时会过滤 <=0
    if first_solution_nodes is None:
        first_solution_nodes = -1

    # 最终路径节点数：从最终 path 直接得到（所有 planner 都适用）
    if final_solution_nodes is None:
        final_solution_nodes = len(path) if has_path else -1

    # 打印信息
    print(f"[{planner_name}] Path found      : {has_path}")
    print(f"[{planner_name}] Best cost       : {best_cost}")
    print(f"[{planner_name}] Runtime (s)     : {runtime:.3f}")
    print(f"[{planner_name}] Iterations      : {final_iter + 1}")
    print(f"[{planner_name}] Collision checks: {n_checks}")
    print(f"[{planner_name}] Total samples   : {total_samples}")
    if first_solution_iter is not None and first_solution_iter > 0:
        print(f"[{planner_name}] First solution iter : {first_solution_iter}")
        print(f"[{planner_name}] First solution cost : {first_solution_cost}")
        print(f"[{planner_name}] First solution nodes: {first_solution_nodes}")
    else:
        print(f"[{planner_name}] First solution iter : N/A")

    print(f"[{planner_name}] Final path nodes     : {final_solution_nodes}")

    return {
        "planner": planner_name,
        "env_idx": env_idx,
        "traj_idx": traj_idx,
        "success": bool(has_path),
        "best_cost": float(best_cost),
        "runtime": float(runtime),
        "n_checks": int(n_checks),
        "total_samples": int(total_samples),
        "final_iter": int(final_iter),
        "iteration_costs": list(iteration_costs) if iteration_costs is not None else [],
        "iteration_time_costs": list(iteration_time_costs) if iteration_time_costs is not None else [],
        # 初始解相关
        "first_solution_iter": int(first_solution_iter) if first_solution_iter is not None else -1,
        "first_solution_cost": float(first_solution_cost) if first_solution_cost is not None else float("inf"),
        "first_solution_nodes": int(first_solution_nodes) if first_solution_nodes is not None else -1,
        # 最优解相关
        "final_solution_nodes": int(final_solution_nodes),
    }


# ---------------------------
# 画平均迭代曲线 (iteration vs cost)
# ---------------------------
def plot_avg_iteration_curves(iter_curves_by_planner, out_path):
    """
    iter_curves_by_planner: dict[planner_name] = [list_of_iteration_costs_list]
    每个 list 是一条任务的 iteration_costs。
    """
    plt.figure()

    any_curve = False
    for name, seqs in iter_curves_by_planner.items():
        if not seqs:
            continue

        # 统一长度：用“保持当前 best cost”向后填充
        max_len = max(len(s) for s in seqs if len(s) > 0)
        if max_len == 0:
            continue

        arr = []
        for s in seqs:
            if len(s) == 0:
                continue
            tmp = np.empty(max_len, dtype=float)
            tmp[:len(s)] = s
            tmp[len(s):] = s[-1]   # 后面保持最后的 best cost
            arr.append(tmp)

        if len(arr) == 0:
            continue

        mat = np.vstack(arr)  # (num_tasks, max_len)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        iters = np.arange(1, max_len + 1)
        plt.plot(iters, mean, label=name)
        plt.fill_between(iters, mean - std, mean + std, alpha=0.2)

        any_curve = True

    if not any_curve:
        print("[WARN] 没有可用的 iteration 曲线来画平均图。")
        return

    plt.xlabel("Iteration")
    plt.ylabel("Best cost so far (mean ± std)")
    plt.title("Average iteration-cost curves over tasks")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 平均迭代曲线图已保存到: {out_path}")


# ---------------------------
# 画平均时间曲线 (time vs cost)
# ---------------------------
def plot_avg_time_curves(time_curves_by_planner, out_path):
    """
    time_curves_by_planner:
        dict[planner_name] = [ (time_list, cost_list), ... ]
    画：时间 t vs best_cost(t) 的平均曲线（带标准差）
    """
    plt.figure()
    any_curve = False

    for name, seqs in time_curves_by_planner.items():
        if not seqs:
            continue

        # 找到该 planner 下的最大时间
        max_t = 0.0
        for times, costs in seqs:
            if times and len(times) > 0:
                max_t = max(max_t, float(times[-1]))
        if max_t <= 0:
            continue

        # 建一个统一的时间网格
        time_grid = np.linspace(0.0, max_t, num=100)

        arr = []
        for times, costs in seqs:
            if not times or not costs or len(times) < 2 or len(costs) < 2:
                continue
            t_arr = np.array(times, dtype=float)
            c_arr = np.array(costs, dtype=float)

            # 确保时间严格递增（去掉重复）
            t_arr, unique_idx = np.unique(t_arr, return_index=True)
            c_arr = c_arr[unique_idx]

            # 插值到统一时间网格
            interp_cost = np.interp(time_grid, t_arr, c_arr)
            arr.append(interp_cost)

        if not arr:
            continue

        mat = np.vstack(arr)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        plt.plot(time_grid, mean, label=name)
        plt.fill_between(time_grid, mean - std, mean + std, alpha=0.2)

        any_curve = True

    if not any_curve:
        print("[WARN] 没有可用的 time 曲线来画平均图。")
        return

    plt.xlabel("Time (s)")
    plt.ylabel("Best cost so far (mean ± std)")
    plt.title("Average time-cost curves over tasks")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 平均时间-代价曲线图已保存到: {out_path}")


# ---------------------------
# 画 scatter：x vs best_cost（原始）
# ---------------------------
def plot_scatter_x_cost(metrics, x_key, out_path, x_label):
    """
    metrics: list[dict]，每个 dict 含有 planner, best_cost, x_key
    """
    plt.figure()
    planners = sorted({m["planner"] for m in metrics})

    any_point = False
    for p in planners:
        xs = []
        ys = []
        for m in metrics:
            if m["planner"] != p:
                continue
            if not m["success"]:
                continue
            val = m.get(x_key, None)
            if val is None:
                continue
            xs.append(val)
            ys.append(m["best_cost"])
        if not xs:
            continue
        plt.scatter(xs, ys, label=p, alpha=0.7)
        any_point = True

    if not any_point:
        print(f"[WARN] 无法绘制 {x_key} vs cost 散点图（没有成功样本）。")
        return

    plt.xlabel(x_label)
    plt.ylabel("Best cost")
    plt.title(f"{x_label} vs Best cost")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} vs cost 散点图已保存到: {out_path}")


# ---------------------------
# 通用 scatter：x vs y
# ---------------------------
def plot_scatter_xy(metrics, x_key, y_key, out_path, x_label, y_label, success_only=True):
    """
    通用散点图：按 planner 分组画 x vs y。
    """
    plt.figure()
    planners = sorted({m["planner"] for m in metrics})

    any_point = False
    for p in planners:
        xs, ys = [], []
        for m in metrics:
            if m["planner"] != p:
                continue
            if success_only and not m.get("success", False):
                continue

            xv = m.get(x_key, None)
            yv = m.get(y_key, None)
            if xv is None or yv is None:
                continue
            xs.append(xv)
            ys.append(yv)

        if not xs:
            continue

        plt.scatter(xs, ys, label=p, alpha=0.7)
        any_point = True

    if not any_point:
        print(f"[WARN] 无法绘制 {x_key} vs {y_key} 散点图（没有有效样本）。")
        return

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label}")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} vs {y_label} 散点图已保存到: {out_path}")


# ---------------------------
# 画某个指标的直方图 + 箱线图
# ---------------------------
def plot_metric_hist_and_box(metrics, value_key, out_prefix, x_label, success_only=True):
    """
    为某个指标绘制直方图和箱线图（按 planner 分组）。

    metrics: list[dict]
    value_key:
        - "first_solution_iter"
        - "first_solution_cost"
        - "first_solution_nodes"
        - "final_iter_plus1"  -> m["final_iter"] + 1
        - "best_cost"
        - "final_solution_nodes"
        - "runtime"
        - "avg_iter_time"
        - "time_to_threshold"
    out_prefix: 输出文件前缀，不含后缀，例如 ".../first_iter_stats_xxx"
    x_label: 坐标轴 / 标题中使用的名称
    success_only: True 时只统计成功的样本
    """
    by_planner = defaultdict(list)

    for m in metrics:
        if success_only and not m.get("success", False):
            continue

        if value_key == "final_iter_plus1":
            v = m.get("final_iter", -1)
            if v >= 0:
                v = v + 1  # 1-based
        elif value_key in (
            "first_solution_iter",
            "first_solution_cost",
            "first_solution_nodes",
            "best_cost",
            "final_solution_nodes",
            "runtime",
            "avg_iter_time",
            "time_to_threshold",
        ):
            v = m.get(value_key, None)
        else:
            # 允许扩展更多 key
            v = m.get(value_key, None)

        if v is None:
            continue

        # 过滤掉 inf / nan
        if isinstance(v, (int, float)) and not np.isfinite(v):
            continue

        # 对迭代数 / 节点数类指标，过滤掉 <=0 的值
        if value_key in ("first_solution_iter", "first_solution_nodes", "final_iter_plus1", "final_solution_nodes"):
            if v <= 0:
                continue

        by_planner[m["planner"]].append(v)

    if not by_planner:
        print(f"[WARN] 没有 {value_key} 的有效数据，无法绘制直方图/箱线图。")
        return

    # ---------- 直方图 ----------
    plt.figure()
    for p, vals in by_planner.items():
        if not vals:
            continue
        plt.hist(vals, bins="auto", alpha=0.6, label=p)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(f"Histogram of {x_label}")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    hist_path = out_prefix + "_hist.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} 直方图已保存到: {hist_path}")

    # ---------- 箱线图 ----------
    plt.figure()
    planners_sorted = sorted(by_planner.keys())
    data = [by_planner[p] for p in planners_sorted]
    plt.boxplot(data, labels=planners_sorted, showmeans=True)
    plt.xlabel("Planner")
    plt.ylabel(x_label)
    plt.title(f"Boxplot of {x_label}")
    plt.grid(True, axis="y")
    box_path = out_prefix + "_box.png"
    plt.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} 箱线图已保存到: {box_path}")


# ---------------------------
# 保存 CSV / NPY
# ---------------------------
def save_metrics(metrics, out_csv, out_npy):
    """
    metrics: list[dict]
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # CSV 字段顺序
    fieldnames = [
        "planner",
        "env_idx",
        "traj_idx",
        "success",
        # 初始解
        "first_solution_iter",
        "first_solution_cost",
        "first_solution_nodes",
        # 最优解
        "final_iter",
        "best_cost",
        "final_solution_nodes",
        # 运行过程
        "runtime",
        "n_checks",
        "total_samples",
        "num_iters",
        # 时间相关派生指标
        "avg_iter_time",
        "time_to_threshold",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            row = {
                "planner": m["planner"],
                "env_idx": m["env_idx"],
                "traj_idx": m["traj_idx"],
                "success": int(m["success"]),
                "first_solution_iter": m.get("first_solution_iter", -1),
                "first_solution_cost": m.get("first_solution_cost", float("inf")),
                "first_solution_nodes": m.get("first_solution_nodes", -1),
                "final_iter": m["final_iter"],
                "best_cost": m["best_cost"],
                "final_solution_nodes": m.get("final_solution_nodes", -1),
                "runtime": m["runtime"],
                "n_checks": m["n_checks"],
                "total_samples": m["total_samples"],
                "num_iters": len(m["iteration_costs"]),
                "avg_iter_time": m.get("avg_iter_time", float("nan")),
                "time_to_threshold": m.get("time_to_threshold", float("inf")),
            }
            writer.writerow(row)
    print(f"[INFO] 统计已保存到 CSV: {out_csv}")

    # NPY：存原始 dict 列表
    np.save(out_npy, metrics, allow_pickle=True)
    print(f"[INFO] 统计已保存到 NPY:  {out_npy}")


# ---------------------------
# 打印统计：均值 / 方差 / 成功率
# ---------------------------
def print_summary(metrics):
    by_planner = defaultdict(list)
    for m in metrics:
        by_planner[m["planner"]].append(m)

    print("\n========== 批量统计结果 ==========")
    for p, lst in by_planner.items():
        runtimes = np.array([m["runtime"] for m in lst], dtype=float)
        checks = np.array([m["n_checks"] for m in lst], dtype=float)
        iters = np.array([m["final_iter"] + 1 for m in lst], dtype=float)
        success_flags = np.array([m["success"] for m in lst], dtype=int)

        # 只统计成功样本的 cost / 节点数
        costs = np.array(
            [m["best_cost"] for m in lst if m["success"]],
            dtype=float,
        )
        first_iters = np.array(
            [
                m.get("first_solution_iter", -1)
                for m in lst
                if m["success"] and m.get("first_solution_iter", -1) > 0
            ],
            dtype=float,
        )
        first_costs = np.array(
            [
                m.get("first_solution_cost", float("inf"))
                for m in lst
                if m["success"] and np.isfinite(m.get("first_solution_cost", float("inf")))
            ],
            dtype=float,
        )
        first_nodes = np.array(
            [
                m.get("first_solution_nodes", -1)
                for m in lst
                if m["success"] and m.get("first_solution_nodes", -1) > 0
            ],
            dtype=float,
        )
        final_nodes = np.array(
            [
                m.get("final_solution_nodes", -1)
                for m in lst
                if m["success"] and m.get("final_solution_nodes", -1) > 0
            ],
            dtype=float,
        )

        # 平均每迭代耗时（过滤 nan / inf）
        avg_iter_times = np.array(
            [m.get("avg_iter_time", np.nan) for m in lst],
            dtype=float,
        )
        avg_iter_times = avg_iter_times[np.isfinite(avg_iter_times)]

        # time-to-threshold（过滤 inf / nan）
        time_to_threshold = np.array(
            [m.get("time_to_threshold", np.nan) for m in lst],
            dtype=float,
        )
        time_to_threshold = time_to_threshold[np.isfinite(time_to_threshold)]

        print(f"\n--- Planner: {p} ---")
        print(f"样本数: {len(lst)}")
        print(
            f"成功数: {success_flags.sum()} / {len(lst)} "
            f"(成功率 = {success_flags.mean() * 100:.1f}%)"
        )

        print(f"runtime 平均 / 方差: {runtimes.mean():.4f} / {runtimes.var():.4f}")
        print(f"collision_checks 平均 / 方差: {checks.mean():.1f} / {checks.var():.1f}")
        print(f"iterations (final, 1-based) 平均 / 方差: {iters.mean():.1f} / {iters.var():.1f}")

        if len(costs) > 0:
            print(
                f"best_cost (最终, 成功样本) 平均 / 方差: "
                f"{costs.mean():.4f} / {costs.var():.4f}"
            )

        if len(first_iters) > 0:
            print(
                f"first_solution_iter (成功样本, 1-based) 平均 / 方差: "
                f"{first_iters.mean():.1f} / {first_iters.var():.1f}"
            )
        else:
            print("first_solution_iter: 无成功样本或未记录。")

        if len(first_costs) > 0:
            print(
                f"first_solution_cost (成功样本) 平均 / 方差: "
                f"{first_costs.mean():.4f} / {first_costs.var():.4f}"
            )

        if len(first_nodes) > 0:
            print(
                f"first_solution_nodes (成功样本) 平均 / 方差: "
                f"{first_nodes.mean():.1f} / {first_nodes.var():.1f}"
            )

        if len(final_nodes) > 0:
            print(
                f"final_solution_nodes (成功样本) 平均 / 方差: "
                f"{final_nodes.mean():.1f} / {final_nodes.var():.1f}"
            )

        if avg_iter_times.size > 0:
            print(
                f"avg_iter_time (每迭代耗时) 平均 / 方差: "
                f"{avg_iter_times.mean():.6f} / {avg_iter_times.var():.6f}"
            )

        if time_to_threshold.size > 0:
            print(
                f"time_to_threshold (达阈值耗时) 平均 / 方差: "
                f"{time_to_threshold.mean():.4f} / {time_to_threshold.var():.4f}"
            )


# ---------------------------
# 命令行参数
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="批量对比多个 planner（BITStar/NIBITStar）并统计性能指标"
    )

    # 数据相关
    parser.add_argument("--data_root", type=str, default="data/liche")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    # 批量任务数：1 = 单任务模式（随机或指定）；>1 = 多任务统计模式
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=30,
        help="要随机抽取多少个 (env, traj) 任务；1 表示只跑一个任务（兼容原来行为）",
    )

    # 如果你想固定 env/traj，就指定这两个；否则随机
    parser.add_argument("--env_idx", type=int, default=-1)
    parser.add_argument("--traj_idx", type=int, default=-1)

    # 规划算法列表
    parser.add_argument(
        "--planners",
        nargs="+",
        type=str,
        default=["BITStar", "NIBITStar", "IRRTStar"],
        help="要对比的规划算法，例如: --planners BITStar NIBITStar",
    )

    # BIT*/NIBIT* 公共参数
    parser.add_argument("--iter_max", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--pc_n_points", type=int, default=2048)

    # NIBIT 相关
    parser.add_argument(
        "--ckpt",
        type=str,
        default="results/model_training/liche_pointnet/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--voxel_resolution",
        type=int,
        nargs=3,
        default=[50, 50, 50],
    )

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # 绝对阈值（可选）：所有任务共用
    parser.add_argument(
        "--cost_threshold",
        type=float,
        default=None,
        help="全局绝对路径代价阈值，用于 time-to-threshold；"
             "若同时设置了 --rel_cost_factor，则优先使用相对阈值",
    )

    # 相对阈值（推荐）：基于“每个任务所有 planner 的最终最优 best_cost”
    parser.add_argument(
        "--rel_cost_factor",
        type=float,
        default=None,
        help="若设置，例如 1.05，则对每个任务使用 rel_cost_factor * "
             "(该任务所有planner中最终最优best_cost) 作为 time-to-threshold 的阈值",
    )

    return parser.parse_args()


# ---------------------------
# 主函数
# ---------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载环境列表
    env_list = get_env_configs(root_dir=args.data_root, split=args.split)
    num_env = len(env_list)
    print(f"[INFO] 共 {num_env} 个 env。")

    metrics = []
    iter_curves_by_planner = defaultdict(list)
    time_curves_by_planner = defaultdict(list)

    # 决定任务数
    num_tasks = max(1, args.num_tasks)

    for task_id in range(num_tasks):
        # 选择 env_idx
        if args.env_idx < 0 or args.env_idx >= num_env:
            env_idx = random.randrange(num_env)
        else:
            env_idx = args.env_idx

        env_record = env_list[env_idx]

        # 该 env 内的路径数
        starts = env_record.get("start", [])
        n_traj = len(starts)
        if n_traj == 0:
            print(f"[WARN] 环境 {env_idx} 中没有轨迹，跳过。")
            continue

        # 选择 traj_idx
        if args.traj_idx < 0 or args.traj_idx >= n_traj:
            traj_idx = random.randrange(n_traj)
        else:
            traj_idx = args.traj_idx

        print("\n======================================")
        print(
            f"[TASK {task_id+1}/{num_tasks}] split={args.split}, env={env_idx}/{num_env-1}, "
            f"traj={traj_idx}/{n_traj-1}"
        )
        print(f"[INFO] Planners: {args.planners}")
        print("======================================")

        for planner_name in args.planners:
            print(f"\n[INFO] ===== Running planner: {planner_name} =====")
            res = run_one_task_one_planner(
                planner_name,
                args,
                env_record,
                env_idx,
                traj_idx,
                gui=args.gui,
            )
            metrics.append(res)

            if res["success"] and len(res["iteration_costs"]) > 0:
                iter_curves_by_planner[planner_name].append(res["iteration_costs"])

            if res["success"] and len(res.get("iteration_time_costs", [])) > 0:
                time_curves_by_planner[planner_name].append(
                    (res["iteration_time_costs"], res["iteration_costs"])
                )

    # 如果只跑 1 个任务，还画单任务的 iteration 对比图（带时间双轴）
    if num_tasks == 1 and metrics:
        fig, ax1 = plt.subplots()
        has_cost = False

        for m in metrics:
            costs = m["iteration_costs"]
            if not costs:
                continue
            iters = np.arange(1, len(costs) + 1)
            ax1.plot(iters, costs, marker="o", label=f"{m['planner']} cost")
            has_cost = True

        if has_cost:
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best cost so far")
            ax1.grid(True)

            # 右轴：累计时间
            ax2 = ax1.twinx()
            has_time = False
            for m in metrics:
                times = m.get("iteration_time_costs", [])
                if not times:
                    continue
                iters = np.arange(1, len(times) + 1)
                ax2.plot(iters, times, linestyle="--", alpha=0.7, label=f"{m['planner']} time (s)")
                has_time = True

            if has_time:
                ax2.set_ylabel("Time (s)")

            # 合并 legend
            lines, labels = [], []
            for ax in (ax1, ax2):
                h, l = ax.get_legend_handles_labels()
                lines += h
                labels += l
            if lines:
                ax1.legend(lines, labels, loc="best")

            out_dir = os.path.join("results", "plots")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f"compare_iter_cost_env{metrics[0]['env_idx']}_"
                f"traj{metrics[0]['traj_idx']}.png",
            )
            fig.suptitle(
                f"Iteration / time comparison (env {metrics[0]['env_idx']}, "
                f"traj {metrics[0]['traj_idx']})"
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] 单任务迭代/时间曲线对比图已保存到: {out_path}")

    # 批量统计 + 画图
    if metrics:
        # ---------- 先统计：每个 (env_idx, traj_idx) 的最优 best_cost ----------
        from collections import defaultdict as dd2

        best_cost_by_task = dd2(lambda: float("inf"))
        for m in metrics:
            key = (m["env_idx"], m["traj_idx"])
            if m.get("success", False) and np.isfinite(m["best_cost"]):
                if m["best_cost"] < best_cost_by_task[key]:
                    best_cost_by_task[key] = m["best_cost"]

        # ---------- 计算时间相关的派生指标 ----------
        # 平均每迭代耗时
        for m in metrics:
            num_iters = len(m.get("iteration_costs", []))
            if num_iters > 0 and m["runtime"] > 0:
                m["avg_iter_time"] = m["runtime"] / float(num_iters)
            else:
                m["avg_iter_time"] = float("nan")

        # 达到某个 cost 阈值所需时间
        # 优先使用每任务相对最优的阈值 (--rel_cost_factor)，
        # 若未设置，再退回全局绝对阈值 (--cost_threshold)
        for m in metrics:
            times = m.get("iteration_time_costs", [])
            costs = m.get("iteration_costs", [])
            if not times or not costs:
                m["time_to_threshold"] = float("inf")
                continue

            threshold = None

            # 优先：相对每个任务的最优 best_cost
            if args.rel_cost_factor is not None:
                alpha = float(args.rel_cost_factor)
                key = (m["env_idx"], m["traj_idx"])
                base = best_cost_by_task[key]
                if np.isfinite(base) and base > 0:
                    threshold = alpha * base

            # 其次：旧的全局绝对阈值
            if threshold is None and args.cost_threshold is not None:
                threshold = float(args.cost_threshold)

            # 两种都没设，就不统计这个指标
            if threshold is None:
                m["time_to_threshold"] = float("inf")
                continue

            # 找第一次 cost <= threshold 的时间
            tt = float("inf")
            for t, c in zip(times, costs):
                if c <= threshold:
                    tt = float(t)
                    break

            m["time_to_threshold"] = tt

        # 1) 打印统计
        print_summary(metrics)

        # 2) 保存 CSV / NPY
        stats_dir = os.path.join("results", "stats")
        csv_path = os.path.join(
            stats_dir,
            f"stats_{args.split}_tasks{num_tasks}_seed{args.seed}.csv",
        )
        npy_path = os.path.join(
            stats_dir,
            f"stats_{args.split}_tasks{num_tasks}_seed{args.seed}.npy",
        )
        save_metrics(metrics, csv_path, npy_path)

        # 3) 平均迭代曲线 (iteration vs cost)
        avg_plot_path = os.path.join(
            "results",
            "plots",
            f"avg_iter_cost_{args.split}_tasks{num_tasks}_seed{args.seed}.png",
        )
        plot_avg_iteration_curves(iter_curves_by_planner, avg_plot_path)

        # 3b) 平均时间-代价曲线 (time vs cost)
        avg_time_plot_path = os.path.join(
            "results",
            "plots",
            f"avg_time_cost_{args.split}_tasks{num_tasks}_seed{args.seed}.png",
        )
        plot_avg_time_curves(time_curves_by_planner, avg_time_plot_path)

        # 4) 若干散点图：runtime / checks / first_iter vs cost
        scatter_rt_path = os.path.join(
            "results",
            "plots",
            f"rt_vs_cost_{args.split}_tasks{num_tasks}_seed{args.seed}.png",
        )
        plot_scatter_x_cost(metrics, "runtime", scatter_rt_path, "Runtime (s)")

        scatter_ck_path = os.path.join(
            "results",
            "plots",
            f"checks_vs_cost_{args.split}_tasks{num_tasks}_seed{args.seed}.png",
        )
        plot_scatter_x_cost(metrics, "n_checks", scatter_ck_path, "Collision checks")

        scatter_first_iter_path = os.path.join(
            "results",
            "plots",
            f"first_iter_vs_cost_{args.split}_tasks{num_tasks}_seed{args.seed}.png",
        )
        plot_scatter_x_cost(
            metrics,
            "first_solution_iter",
            scatter_first_iter_path,
            "First solution iter (1-based)",
        )

        # Runtime vs Total samples 散点图
        rt_vs_samples_path = os.path.join(
            "results",
            "plots",
            f"rt_vs_samples_{args.split}_tasks{num_tasks}_seed{args.seed}.png",
        )
        plot_scatter_xy(
            metrics,
            x_key="total_samples",
            y_key="runtime",
            out_path=rt_vs_samples_path,
            x_label="Total samples",
            y_label="Runtime (s)",
            success_only=False,
        )

        # 5) (a)~(c)：初始路径相关直方图 / 箱线图
        first_iter_prefix = os.path.join(
            "results",
            "plots",
            f"first_iter_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="first_solution_iter",
            out_prefix=first_iter_prefix,
            x_label="Initial path iteration (1-based)",
            success_only=True,
        )

        first_cost_prefix = os.path.join(
            "results",
            "plots",
            f"first_cost_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="first_solution_cost",
            out_prefix=first_cost_prefix,
            x_label="Initial path cost",
            success_only=True,
        )

        first_nodes_prefix = os.path.join(
            "results",
            "plots",
            f"first_nodes_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="first_solution_nodes",
            out_prefix=first_nodes_prefix,
            x_label="Initial path nodes",
            success_only=True,
        )

        # 6) (d)~(f)：最优路径相关直方图 / 箱线图
        final_iter_prefix = os.path.join(
            "results",
            "plots",
            f"final_iter_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="final_iter_plus1",
            out_prefix=final_iter_prefix,
            x_label="Final iteration (1-based)",
            success_only=True,
        )

        final_cost_prefix = os.path.join(
            "results",
            "plots",
            f"final_cost_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="best_cost",
            out_prefix=final_cost_prefix,
            x_label="Final path cost",
            success_only=True,
        )

        final_nodes_prefix = os.path.join(
            "results",
            "plots",
            f"final_nodes_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="final_solution_nodes",
            out_prefix=final_nodes_prefix,
            x_label="Final path nodes",
            success_only=True,
        )

        # 7) runtime 分布（所有样本）
        runtime_prefix = os.path.join(
            "results",
            "plots",
            f"runtime_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="runtime",
            out_prefix=runtime_prefix,
            x_label="Runtime (s)",
            success_only=False,
        )

        # 8) time-to-threshold 分布（仅当设置了阈值时绘制）
        if args.rel_cost_factor is not None or args.cost_threshold is not None:
            tth_prefix = os.path.join(
                "results",
                "plots",
                f"time_to_threshold_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
            )
            label = "Time to threshold (s)"
            plot_metric_hist_and_box(
                metrics,
                value_key="time_to_threshold",
                out_prefix=tth_prefix,
                x_label=label,
                success_only=True,
            )

        # 9) 平均每迭代耗时分布
        avg_it_time_prefix = os.path.join(
            "results",
            "plots",
            f"avg_iter_time_stats_{args.split}_tasks{num_tasks}_seed{args.seed}",
        )
        plot_metric_hist_and_box(
            metrics,
            value_key="avg_iter_time",
            out_prefix=avg_it_time_prefix,
            x_label="Average iteration time (s)",
            success_only=False,
        )

    else:
        print("[WARN] 没有任何有效任务被跑到，无法统计。")


if __name__ == "__main__":
    main()
