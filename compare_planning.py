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
)
from neural_wrapper import ECSP_NeuralWrapper, PNGWrapper
from path_planning_classes_arm.bit_star import get_bit_planner as get_bit_planner_bit
from path_planning_classes_arm.nibit_star_fixed import get_bit_planner as get_bit_planner_nibit
from path_planning_classes_arm.irrtstar_updated import get_irrtstar_planner, get_nirrtstarECSP_planner, get_nirrtstar_png_planner
from path_planning_classes_arm.bifmtstar import get_bifmt_planner


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
            nw_cache["NIBITSTAR"] = ECSP_NeuralWrapper(
                problem=problem,
                ckpt_path=args.ckpt,
                voxel_resolution=tuple(args.voxel_resolution),
                device="cuda",
            )
        return get_bit_planner_nibit(args, problem, neural_wrapper=nw_cache["NIBITSTAR"])

    elif name == "IRRTSTAR":
        return get_irrtstar_planner(args, problem, neural_wrapper=None)

    elif name == "NIRRTSTARECSP":
        if "NIRRTSTARECSP" not in nw_cache:
            nw_cache["NIRRTSTARECSP"] = ECSP_NeuralWrapper(
                problem=problem,
                ckpt_path="results/model_training/train_20251215-144528/best.pt",
                voxel_resolution=tuple(args.voxel_resolution),
                device="cuda",
            )
        return get_nirrtstarECSP_planner(args, problem, neural_wrapper=nw_cache["NIRRTSTARECSP"])

    elif name == "NIRRTSTARPNG":
        if "NIRRTSTARPNG" not in nw_cache:
            nw_cache["NIRRTSTARPNG"] = PNGWrapper(
                device="cuda",
            )
        return get_nirrtstar_png_planner(args, problem, neural_wrapper=nw_cache["NIRRTSTARPNG"])

    elif name == "BIFMTSTAR":
        return get_bifmt_planner(args, problem, neural_wrapper=None)

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
                first_iter_idx = len(iteration_costs) - 1
            if first_iter_idx >= 0:
                first_solution_iter = first_iter_idx + 1
                if first_solution_cost is None:
                    first_solution_cost = float(iteration_costs[first_iter_idx])

    if first_solution_nodes is None:
        first_solution_nodes = -1

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
        # difficulty 字段稍后填
        "difficulty_bucket": -1,
        "difficulty_value": float("nan"),
    }


# ---------------------------
# Difficulty bucket（按 task 划分）
# ---------------------------
def compute_task_optimal_metric(metrics, metric_name):
    """
    对每个 task=(env_idx,traj_idx)，计算一个“最优(oracle)”值：
      - metric_name == "best_cost": 取成功样本里 best_cost 的最小值（越小越好）
      - metric_name == "runtime":   取成功样本里 runtime   的最小值（越小越好）

    返回:
      task2val: dict[(env,traj)] -> float (若该 task 无任何成功样本，则为 inf)
    """
    from collections import defaultdict as dd

    task2vals = dd(list)
    for m in metrics:
        if not m.get("success", False):
            continue
        key = (m["env_idx"], m["traj_idx"])
        v = m.get(metric_name, float("inf"))
        if v is None:
            continue
        if np.isfinite(v):
            task2vals[key].append(float(v))

    task2best = {}
    all_tasks = {(m["env_idx"], m["traj_idx"]) for m in metrics}
    for key in all_tasks:
        if key in task2vals and len(task2vals[key]) > 0:
            task2best[key] = float(np.min(task2vals[key]))
        else:
            task2best[key] = float("inf")
    return task2best


def assign_difficulty_buckets(metrics, metric_name="best_cost", num_buckets=3):
    """
    用分位数把 task 分成 num_buckets 个 bucket：
      bucket 0: 最容易（metric 最小的一段）
      bucket num_buckets-1: 最难（metric 最大的一段）

    注意：这里 metric_name 表示“以 task 的最优(best) cost/runtime”作为难度指标（二选一）。
    """
    num_buckets = int(max(2, num_buckets))

    task2best = compute_task_optimal_metric(metrics, metric_name)

    # 只用 finite 值算分位数；inf（无成功）默认归到最难 bucket
    finite_vals = np.array([v for v in task2best.values() if np.isfinite(v)], dtype=float)
    if finite_vals.size == 0:
        # 极端情况：所有 task 都失败
        for m in metrics:
            m["difficulty_bucket"] = num_buckets - 1
            m["difficulty_value"] = float("inf")
        edges = [float("inf")] * (num_buckets - 1)
        return edges, task2best

    # 分位数切分点：例如 3 buckets -> q=[1/3, 2/3]
    qs = [k / num_buckets for k in range(1, num_buckets)]
    edges = [float(np.quantile(finite_vals, q)) for q in qs]

    def bucket_of(v):
        if not np.isfinite(v):
            return num_buckets - 1
        b = 0
        while b < len(edges) and v > edges[b]:
            b += 1
        return b

    for m in metrics:
        key = (m["env_idx"], m["traj_idx"])
        v = task2best.get(key, float("inf"))
        m["difficulty_value"] = float(v)
        m["difficulty_bucket"] = int(bucket_of(v))

    return edges, task2best


def bucket_name(b, num_buckets):
    if num_buckets == 3:
        return ["easy", "medium", "hard"][b]
    return f"bucket{b}"


def filter_metrics(metrics, bucket=None):
    if bucket is None:
        return metrics
    return [m for m in metrics if int(m.get("difficulty_bucket", -1)) == int(bucket)]


# ---------------------------
# 画平均迭代曲线 (iteration vs cost)
# ---------------------------
def plot_avg_iteration_curves(metrics, out_path, bucket=None):
    """
    从 metrics 里按 planner 分组画平均 iteration-cost 曲线（mean ± std）。
    可选按 difficulty bucket 过滤。
    """
    ms = filter_metrics(metrics, bucket=bucket)
    plt.figure()

    planners = sorted({m["planner"] for m in ms})
    any_curve = False

    for name in planners:
        seqs = []
        for m in ms:
            if m["planner"] != name:
                continue
            if not m.get("success", False):
                continue
            s = m.get("iteration_costs", []) or []
            if len(s) > 0:
                seqs.append(s)

        if not seqs:
            continue

        max_len = max(len(s) for s in seqs if len(s) > 0)
        if max_len == 0:
            continue

        arr = []
        for s in seqs:
            tmp = np.empty(max_len, dtype=float)
            tmp[:len(s)] = s
            tmp[len(s):] = s[-1]
            arr.append(tmp)

        if len(arr) == 0:
            continue

        mat = np.vstack(arr)
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
    title = "Average iteration-cost curves over tasks"
    if bucket is not None:
        title += f" ({bucket})"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 平均迭代曲线图已保存到: {out_path}")


# ---------------------------
# 画成功率曲线 (Success Rate vs Iteration)
# ---------------------------
def plot_success_rate_over_iterations(metrics, out_path, bucket=None):
    """
    画：Success Rate vs Iteration（按 planner 分组）
    规则：若 iteration_costs 在某次迭代变为 finite，则从该迭代起视为“已成功”。
    可选按 difficulty bucket 过滤。
    """
    ms = filter_metrics(metrics, bucket=bucket)
    plt.figure()

    planners = sorted({m["planner"] for m in ms})
    any_curve = False

    for p in planners:
        seqs = []
        for m in ms:
            if m["planner"] != p:
                continue
            seqs.append(m.get("iteration_costs", []) or [])

        if not seqs:
            continue

        max_len = max((len(s) for s in seqs), default=0)
        if max_len == 0:
            continue

        success_mat = []
        for s in seqs:
            if len(s) == 0:
                success_mat.append(np.zeros(max_len, dtype=float))
                continue

            s_arr = np.array(s, dtype=float)
            if len(s_arr) < max_len:
                pad = np.full(max_len - len(s_arr), s_arr[-1], dtype=float)
                s_arr = np.concatenate([s_arr, pad], axis=0)

            success_vec = np.isfinite(s_arr).astype(float)
            success_mat.append(success_vec)

        mat = np.vstack(success_mat)
        success_rate = mat.mean(axis=0)

        iters = np.arange(1, max_len + 1)
        plt.plot(iters, success_rate, label=p)
        any_curve = True

    if not any_curve:
        print("[WARN] 没有可用的 success-rate 曲线来画图。")
        return

    plt.xlabel("Iteration")
    plt.ylabel("Success rate")
    title = "Success rate over iterations"
    if bucket is not None:
        title += f" ({bucket})"
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Success-rate 曲线图已保存到: {out_path}")


# ---------------------------
# 画平均时间曲线 (time vs cost)
# ---------------------------
def plot_avg_time_curves(metrics, out_path, bucket=None):
    """
    画：时间 t vs best_cost(t) 的平均曲线（带标准差）
    ✅ 改进：统一使用公共时间区间 [0, min(max_t_planner)] 做均值，避免不同 planner 覆盖时间段不同导致均值不可比
    """
    ms = filter_metrics(metrics, bucket=bucket)
    plt.figure()

    planners = sorted({m["planner"] for m in ms})

    # 1) 先收集每个 planner 的 (times,costs) 序列，并计算各自的 max_t
    seqs_by_planner = {}
    max_t_by_planner = {}
    for name in planners:
        seqs = []
        max_t = 0.0
        for m in ms:
            if m["planner"] != name:
                continue
            if not m.get("success", False):
                continue
            times = m.get("iteration_time_costs", []) or []
            costs = m.get("iteration_costs", []) or []
            if not times or not costs:
                continue
            if len(times) < 2 or len(costs) < 2:
                continue
            # 防御：长度对齐
            L = min(len(times), len(costs))
            times = times[:L]
            costs = costs[:L]
            # 防御：最后时间必须 > 0
            if times[-1] is None or float(times[-1]) <= 0:
                continue
            seqs.append((times, costs))
            max_t = max(max_t, float(times[-1]))

        if seqs:
            seqs_by_planner[name] = seqs
            max_t_by_planner[name] = max_t

    if not seqs_by_planner:
        print("[WARN] 没有可用的 time 曲线来画平均图。")
        return

    # 2) 公共时间上界：min(max_t_planner)
    T_common = min(max_t_by_planner.values())
    if T_common <= 0:
        print("[WARN] 公共时间上界无效，无法绘图。")
        return

    # 3) 统一时间网格（公共区间）
    time_grid = np.linspace(0.0, T_common, num=100)

    # 4) 对每个 planner：只在 [0, T_common] 上插值并平均
    any_curve = False
    for name, seqs in seqs_by_planner.items():
        arr = []
        for times, costs in seqs:
            t_arr = np.array(times, dtype=float)
            c_arr = np.array(costs, dtype=float)

            # 去重 + 保序（确保时间单调）
            t_arr, unique_idx = np.unique(t_arr, return_index=True)
            c_arr = c_arr[unique_idx]

            # 只保留 t <= T_common 的部分
            mask = t_arr <= T_common
            t_arr = t_arr[mask]
            c_arr = c_arr[mask]

            # 插值至少要2个点；否则跳过该任务曲线
            if t_arr.size < 2:
                continue

            # 插值到公共网格
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
        print("[WARN] 没有 planner 在公共区间内有足够数据来画 time 曲线。")
        return

    plt.xlabel("Time (s)")
    plt.ylabel("Best cost so far (mean ± std)")
    title = "Average time-cost curves over tasks"
    if bucket is not None:
        title += f" ({bucket})"
    title += f" | common T={T_common:.3f}s"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 平均时间-代价曲线图(公共区间)已保存到: {out_path} (T_common={T_common:.3f}s)")

# ---------------------------
# 画 scatter：x vs best_cost（原始）
# ---------------------------
def plot_scatter_x_cost(metrics, x_key, out_path, x_label, success_only=True, bucket=None):
    plt.figure()
    ms = filter_metrics(metrics, bucket=bucket)
    planners = sorted({m["planner"] for m in ms})

    any_point = False
    for p in planners:
        xs = []
        ys = []
        for m in ms:
            if m["planner"] != p:
                continue
            if success_only and not m.get("success", False):
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
        print(f"[WARN] 无法绘制 {x_key} vs cost 散点图（没有有效样本）。")
        return

    plt.xlabel(x_label)
    plt.ylabel("Best cost")
    title = f"{x_label} vs Best cost"
    if bucket is not None:
        title += f" ({bucket})"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} vs cost 散点图已保存到: {out_path}")


# ---------------------------
# 通用 scatter：x vs y
# ---------------------------
def plot_scatter_xy(metrics, x_key, y_key, out_path, x_label, y_label, success_only=True, bucket=None):
    plt.figure()
    ms = filter_metrics(metrics, bucket=bucket)
    planners = sorted({m["planner"] for m in ms})

    any_point = False
    for p in planners:
        xs, ys = [], []
        for m in ms:
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
    title = f"{x_label} vs {y_label}"
    if bucket is not None:
        title += f" ({bucket})"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} vs {y_label} 散点图已保存到: {out_path}")


# ---------------------------
# 画某个指标的直方图 + 箱线图
# ---------------------------
def plot_metric_hist_and_box(metrics, value_key, out_prefix, x_label, success_only=True, bucket=None):
    by_planner = defaultdict(list)
    ms = filter_metrics(metrics, bucket=bucket)

    for m in ms:
        if success_only and not m.get("success", False):
            continue

        if value_key == "final_iter_plus1":
            v = m.get("final_iter", -1)
            if v >= 0:
                v = v + 1
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
            v = m.get(value_key, None)

        if v is None:
            continue
        if isinstance(v, (int, float)) and not np.isfinite(v):
            continue

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
    title = f"Histogram of {x_label}"
    if bucket is not None:
        title += f" ({bucket})"
    plt.title(title)
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
    title = f"Boxplot of {x_label}"
    if bucket is not None:
        title += f" ({bucket})"
    plt.title(title)
    plt.grid(True, axis="y")
    box_path = out_prefix + "_box.png"
    plt.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] {x_label} 箱线图已保存到: {box_path}")


# ---------------------------
# 保存 CSV / NPY
# ---------------------------
def save_metrics(metrics, out_csv, out_npy):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "planner",
        "env_idx",
        "traj_idx",
        "success",
        # difficulty
        "difficulty_bucket",
        "difficulty_value",
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
                "difficulty_bucket": int(m.get("difficulty_bucket", -1)),
                "difficulty_value": float(m.get("difficulty_value", float("nan"))),
                "first_solution_iter": m.get("first_solution_iter", -1),
                "first_solution_cost": m.get("first_solution_cost", float("inf")),
                "first_solution_nodes": m.get("first_solution_nodes", -1),
                "final_iter": m["final_iter"],
                "best_cost": m["best_cost"],
                "final_solution_nodes": m.get("final_solution_nodes", -1),
                "runtime": m["runtime"],
                "n_checks": m["n_checks"],
                "total_samples": m["total_samples"],
                "num_iters": len(m.get("iteration_costs", [])),
                "avg_iter_time": m.get("avg_iter_time", float("nan")),
                "time_to_threshold": m.get("time_to_threshold", float("inf")),
            }
            writer.writerow(row)
    print(f"[INFO] 统计已保存到 CSV: {out_csv}")

    np.save(out_npy, metrics, allow_pickle=True)
    print(f"[INFO] 统计已保存到 NPY:  {out_npy}")


# ---------------------------
# 打印统计：均值 / 方差 / 成功率（支持按 bucket）
# ---------------------------
def print_summary(metrics, num_buckets=3):
    def _print_one(group_name, group_metrics):
        by_planner = defaultdict(list)
        for m in group_metrics:
            by_planner[m["planner"]].append(m)

        print(f"\n========== 统计结果: {group_name} ==========")
        for p, lst in by_planner.items():
            runtimes = np.array([m["runtime"] for m in lst], dtype=float)
            checks = np.array([m["n_checks"] for m in lst], dtype=float)
            iters = np.array([m["final_iter"] + 1 for m in lst], dtype=float)
            success_flags = np.array([m["success"] for m in lst], dtype=int)

            costs = np.array([m["best_cost"] for m in lst if m["success"]], dtype=float)
            first_iters = np.array(
                [m.get("first_solution_iter", -1) for m in lst
                 if m["success"] and m.get("first_solution_iter", -1) > 0],
                dtype=float,
            )
            first_costs = np.array(
                [m.get("first_solution_cost", float("inf")) for m in lst
                 if m["success"] and np.isfinite(m.get("first_solution_cost", float("inf")))],
                dtype=float,
            )
            first_nodes = np.array(
                [m.get("first_solution_nodes", -1) for m in lst
                 if m["success"] and m.get("first_solution_nodes", -1) > 0],
                dtype=float,
            )
            final_nodes = np.array(
                [m.get("final_solution_nodes", -1) for m in lst
                 if m["success"] and m.get("final_solution_nodes", -1) > 0],
                dtype=float,
            )

            avg_iter_times = np.array([m.get("avg_iter_time", np.nan) for m in lst], dtype=float)
            avg_iter_times = avg_iter_times[np.isfinite(avg_iter_times)]

            time_to_threshold = np.array([m.get("time_to_threshold", np.nan) for m in lst], dtype=float)
            time_to_threshold = time_to_threshold[np.isfinite(time_to_threshold)]

            print(f"\n--- Planner: {p} ---")
            print(f"样本数: {len(lst)}")
            print(f"成功数: {success_flags.sum()} / {len(lst)} (成功率 = {success_flags.mean() * 100:.1f}%)")

            print(f"runtime 平均 / 方差: {runtimes.mean():.4f} / {runtimes.var():.4f}")
            print(f"collision_checks 平均 / 方差: {checks.mean():.1f} / {checks.var():.1f}")
            print(f"iterations (final, 1-based) 平均 / 方差: {iters.mean():.1f} / {iters.var():.1f}")

            if len(costs) > 0:
                print(f"best_cost (最终, 成功样本) 平均 / 方差: {costs.mean():.4f} / {costs.var():.4f}")

            if len(first_iters) > 0:
                print(f"first_solution_iter (成功样本, 1-based) 平均 / 方差: {first_iters.mean():.1f} / {first_iters.var():.1f}")
            else:
                print("first_solution_iter: 无成功样本或未记录。")

            if len(first_costs) > 0:
                print(f"first_solution_cost (成功样本) 平均 / 方差: {first_costs.mean():.4f} / {first_costs.var():.4f}")

            if len(first_nodes) > 0:
                print(f"first_solution_nodes (成功样本) 平均 / 方差: {first_nodes.mean():.1f} / {first_nodes.var():.1f}")

            if len(final_nodes) > 0:
                print(f"final_solution_nodes (成功样本) 平均 / 方差: {final_nodes.mean():.1f} / {final_nodes.var():.1f}")

            if avg_iter_times.size > 0:
                print(f"avg_iter_time (每迭代耗时) 平均 / 方差: {avg_iter_times.mean():.6f} / {avg_iter_times.var():.6f}")

            if time_to_threshold.size > 0:
                print(f"time_to_threshold (达阈值耗时) 平均 / 方差: {time_to_threshold.mean():.4f} / {time_to_threshold.var():.4f}")

    # all
    _print_one("all", metrics)

    # by bucket
    for b in range(num_buckets):
        ms = filter_metrics(metrics, bucket=b)
        if ms:
            _print_one(bucket_name(b, num_buckets), ms)


# ---------------------------
# 命令行参数
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="批量对比多个 planner 并统计性能指标（支持按难度分位数 bucket）"
    )

    # 数据相关
    parser.add_argument("--data_root", type=str, default="data/liche")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    # 批量任务数
    parser.add_argument("--num_tasks", type=int, default=30,
                        help="要随机抽取多少个 (env, traj) 任务；1 表示只跑一个任务")

    # 如果你想固定 env/traj
    parser.add_argument("--env_idx", type=int, default=-1)
    parser.add_argument("--traj_idx", type=int, default=-1)

    # 规划算法列表
    parser.add_argument("--planners", nargs="+", type=str, default=["IRRTSTAR","NIRRTSTARPNG","NIRRTSTARECSP"],
                        help="要对比的规划算法，例如: --planners BITStar NIBITStar")

    # BIT*/NIBIT* 公共参数
    parser.add_argument("--iter_max", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--pc_n_points", type=int, default=2000)

    # NIBIT 相关
    parser.add_argument("--voxel_resolution", type=int, nargs=3, default=[50, 50, 50])

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # 绝对阈值（可选）：所有任务共用
    parser.add_argument("--cost_threshold", type=float, default=None,
                        help="全局绝对路径代价阈值，用于 time-to-threshold；"
                             "若同时设置了 --rel_cost_factor，则优先使用相对阈值")

    # 相对阈值（推荐）
    parser.add_argument("--rel_cost_factor", type=float, default=None,
                        help="若设置，例如 1.05，则对每个任务使用 rel_cost_factor * "
                             "(该任务所有planner中最终最优best_cost) 作为 time-to-threshold 的阈值")

    # difficulty bucket（新增）
    parser.add_argument("--difficulty_metric", type=str, default="runtime",
                        choices=["best_cost", "runtime"],
                        help="用 task 的最优 best_cost 或最优 runtime 来做难度分桶（分位数）")
    parser.add_argument("--difficulty_buckets", type=int, default=3,
                        help="分桶数量（>=2），例如 3 表示 easy/medium/hard")

    return parser.parse_args()


# ---------------------------
# 主函数
# ---------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env_list = get_env_configs(root_dir=args.data_root, split=args.split)
    num_env = len(env_list)
    print(f"[INFO] 共 {num_env} 个 env。")

    metrics = []

    num_tasks = max(1, args.num_tasks)

    for task_id in range(num_tasks):
        if args.env_idx < 0 or args.env_idx >= num_env:
            env_idx = random.randrange(num_env)
        else:
            env_idx = args.env_idx

        env_record = env_list[env_idx]

        starts = env_record.get("start", [])
        n_traj = len(starts)
        if n_traj == 0:
            print(f"[WARN] 环境 {env_idx} 中没有轨迹，跳过。")
            continue

        if args.traj_idx < 0 or args.traj_idx >= n_traj:
            traj_idx = random.randrange(n_traj)
        else:
            traj_idx = args.traj_idx

        print("\n======================================")
        print(f"[TASK {task_id+1}/{num_tasks}] split={args.split}, env={env_idx}/{num_env-1}, traj={traj_idx}/{n_traj-1}")
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

    # 单任务画对比图（保持原逻辑）
    if num_tasks == 1 and metrics:
        fig, ax1 = plt.subplots()
        has_cost = False

        for m in metrics:
            costs = m.get("iteration_costs", [])
            if not costs:
                continue
            iters = np.arange(1, len(costs) + 1)
            ax1.plot(iters, costs, marker="o", label=f"{m['planner']} cost")
            has_cost = True

        if has_cost:
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best cost so far")
            ax1.grid(True)

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

            lines, labels = [], []
            for ax in (ax1, ax2):
                h, l = ax.get_legend_handles_labels()
                lines += h
                labels += l
            if lines:
                ax1.legend(lines, labels, loc="best")

            out_dir = os.path.join("results", "plots")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"compare_iter_cost_env{metrics[0]['env_idx']}_traj{metrics[0]['traj_idx']}.png")
            fig.suptitle(f"Iteration / time comparison (env {metrics[0]['env_idx']}, traj {metrics[0]['traj_idx']})")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] 单任务迭代/时间曲线对比图已保存到: {out_path}")

    # 批量统计 + 画图
    if metrics:
        # ---------- difficulty 分桶（新增） ----------
        edges, task2best = assign_difficulty_buckets(
            metrics,
            metric_name=args.difficulty_metric,
            num_buckets=args.difficulty_buckets,
        )
        print("\n========== Difficulty bucket ==========")
        print(f"[INFO] difficulty_metric = {args.difficulty_metric}")
        print(f"[INFO] difficulty_buckets = {args.difficulty_buckets}")
        print(f"[INFO] quantile edges = {edges} (inf 或无成功任务自动归最难桶)")

        # ---------- 先统计：每个 (env_idx, traj_idx) 的最优 best_cost（供 rel_cost_factor 使用） ----------
        from collections import defaultdict as dd2

        best_cost_by_task = dd2(lambda: float("inf"))
        for m in metrics:
            key = (m["env_idx"], m["traj_idx"])
            if m.get("success", False) and np.isfinite(m["best_cost"]):
                if m["best_cost"] < best_cost_by_task[key]:
                    best_cost_by_task[key] = m["best_cost"]

        # ---------- 计算时间相关的派生指标 ----------
        for m in metrics:
            num_iters = len(m.get("iteration_costs", []))
            if num_iters > 0 and m["runtime"] > 0:
                m["avg_iter_time"] = m["runtime"] / float(num_iters)
            else:
                m["avg_iter_time"] = float("nan")

        for m in metrics:
            times = m.get("iteration_time_costs", [])
            costs = m.get("iteration_costs", [])
            if not times or not costs:
                m["time_to_threshold"] = float("inf")
                continue

            threshold = None

            if args.rel_cost_factor is not None:
                alpha = float(args.rel_cost_factor)
                key = (m["env_idx"], m["traj_idx"])
                base = best_cost_by_task[key]
                if np.isfinite(base) and base > 0:
                    threshold = alpha * base

            if threshold is None and args.cost_threshold is not None:
                threshold = float(args.cost_threshold)

            if threshold is None:
                m["time_to_threshold"] = float("inf")
                continue

            tt = float("inf")
            for t, c in zip(times, costs):
                if c <= threshold:
                    tt = float(t)
                    break
            m["time_to_threshold"] = tt

        # 1) 打印统计（all + 各 bucket）
        print_summary(metrics, num_buckets=args.difficulty_buckets)

        # 2) 保存 CSV / NPY
        stats_dir = os.path.join("results", "stats")
        os.makedirs(stats_dir, exist_ok=True)
        csv_path = os.path.join(
            stats_dir,
            f"stats_{args.split}_tasks{num_tasks}_seed{args.seed}_"
            f"diff{args.difficulty_metric}_B{args.difficulty_buckets}.csv",
        )
        npy_path = os.path.join(
            stats_dir,
            f"stats_{args.split}_tasks{num_tasks}_seed{args.seed}_"
            f"diff{args.difficulty_metric}_B{args.difficulty_buckets}.npy",
        )
        save_metrics(metrics, csv_path, npy_path)

        # 3) 平均迭代曲线 / 成功率曲线 / 平均时间曲线：all + 每个 bucket 各一份
        plots_dir = os.path.join("results", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # all
        plot_avg_iteration_curves(
            metrics,
            os.path.join(plots_dir, f"avg_iter_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            bucket=None,
        )
        plot_success_rate_over_iterations(
            metrics,
            os.path.join(plots_dir, f"success_rate_iter_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            bucket=None,
        )
        plot_avg_time_curves(
            metrics,
            os.path.join(plots_dir, f"avg_time_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            bucket=None,
        )

        # per bucket
        for b in range(args.difficulty_buckets):
            name = bucket_name(b, args.difficulty_buckets)
            plot_avg_iteration_curves(
                metrics,
                os.path.join(plots_dir, f"avg_iter_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_{name}.png"),
                bucket=b,
            )
            plot_success_rate_over_iterations(
                metrics,
                os.path.join(plots_dir, f"success_rate_iter_{args.split}_tasks{num_tasks}_seed{args.seed}_{name}.png"),
                bucket=b,
            )
            plot_avg_time_curves(
                metrics,
                os.path.join(plots_dir, f"avg_time_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_{name}.png"),
                bucket=b,
            )

        # 4) 若干散点图：all（你也可以按 bucket 复制一份）
        plot_scatter_x_cost(
            metrics,
            "runtime",
            os.path.join(plots_dir, f"rt_vs_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            "Runtime (s)",
            success_only=True,
            bucket=None,
        )
        plot_scatter_x_cost(
            metrics,
            "n_checks",
            os.path.join(plots_dir, f"checks_vs_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            "Collision checks",
            success_only=True,
            bucket=None,
        )
        plot_scatter_x_cost(
            metrics,
            "first_solution_iter",
            os.path.join(plots_dir, f"first_iter_vs_cost_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            "First solution iter (1-based)",
            success_only=True,
            bucket=None,
        )
        plot_scatter_xy(
            metrics,
            x_key="total_samples",
            y_key="runtime",
            out_path=os.path.join(plots_dir, f"rt_vs_samples_{args.split}_tasks{num_tasks}_seed{args.seed}_all.png"),
            x_label="Total samples",
            y_label="Runtime (s)",
            success_only=False,
            bucket=None,
        )

        # 5) 分布图（all）
        def _pref(stem):
            return os.path.join(
                plots_dir,
                f"{stem}_{args.split}_tasks{num_tasks}_seed{args.seed}_all"
            )

        plot_metric_hist_and_box(metrics, "first_solution_iter", _pref("first_iter_stats"), "Initial path iteration (1-based)", True, None)
        plot_metric_hist_and_box(metrics, "first_solution_cost", _pref("first_cost_stats"), "Initial path cost", True, None)
        plot_metric_hist_and_box(metrics, "first_solution_nodes", _pref("first_nodes_stats"), "Initial path nodes", True, None)
        plot_metric_hist_and_box(metrics, "final_iter_plus1", _pref("final_iter_stats"), "Final iteration (1-based)", True, None)
        plot_metric_hist_and_box(metrics, "best_cost", _pref("final_cost_stats"), "Final path cost", True, None)
        plot_metric_hist_and_box(metrics, "final_solution_nodes", _pref("final_nodes_stats"), "Final path nodes", True, None)
        plot_metric_hist_and_box(metrics, "runtime", _pref("runtime_stats"), "Runtime (s)", False, None)

        if args.rel_cost_factor is not None or args.cost_threshold is not None:
            plot_metric_hist_and_box(metrics, "time_to_threshold", _pref("time_to_threshold_stats"), "Time to threshold (s)", True, None)

        plot_metric_hist_and_box(metrics, "avg_iter_time", _pref("avg_iter_time_stats"), "Average iteration time (s)", False, None)

    else:
        print("[WARN] 没有任何有效任务被跑到，无法统计。")


if __name__ == "__main__":
    main()
