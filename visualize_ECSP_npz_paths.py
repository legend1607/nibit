import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 需要以启用 3D


def visualize_sample(npz_path, sample_idx=None, joints=None,
                     save_path=None, show=True,
                     path_near_thr=0.1):
    """
    可视化 + 统计一个样本：
      - 统计该样本的标签分布（collision/free/path soft label）
      - 全部点云（区分碰撞 / 自由 / 路径邻域点）
      - 最优路径轨迹
      - 随机选3个关节作为坐标轴

    参数
    ----
    npz_path : str
        train/val/test npz 文件路径
    sample_idx : int or None
        样本索引；None 时随机选
    joints : list[int] or np.ndarray or None
        长度为3的关节索引列表，例如 [0, 2, 5]；
        None 时随机选3个不重复关节
    save_path : str or None
        如果不为 None，则保存到该路径（如 "vis.png"）
    show : bool
        是否 plt.show()
    path_near_thr : float
        判定“路径邻域点”的 soft label 阈值，例如 0.1
    """

    # ========= 1. 读数据 =========
    data = np.load(npz_path, allow_pickle=True)

    # 当前生成脚本中的字段约定：
    # pc         : (N, npoints, D)   归一化后的关节空间点云
    # labels     : (N, npoints)      0: collision, 1: free
    # pathlabels : (N, npoints)      [0,1] 的 soft label，越大越靠近路径
    # paths      : (N,) object       每个是 (T, D) 的轨迹
    # starts/goals: (N, D)
    pc = data["pc"]
    labels = data["labels"]
    pathlabels = data["pathlabels"]
    paths = data["paths"]
    starts = data["starts"]
    goals = data["goals"]

    num_samples = pc.shape[0]

    # ========= 2. 选样本 =========
    if sample_idx is None:
        sample_idx = np.random.randint(num_samples)
    if not (0 <= sample_idx < num_samples):
        raise ValueError(f"sample_idx 超出范围: 0 ~ {num_samples-1}, got {sample_idx}")

    q_pc = pc[sample_idx]                 # (npoints, D)
    lab = labels[sample_idx]              # (npoints,)
    path_soft = pathlabels[sample_idx]    # (npoints,)
    path = np.array(paths[sample_idx])    # (T, D)
    start = starts[sample_idx]            # (D,)
    goal = goals[sample_idx]              # (D,)

    n_points, D = q_pc.shape

    print("====================================")
    print(f"样本 index = {sample_idx}")
    print(f"点数量 n_points = {n_points}")
    print("====================================\n")

    # ========= 2.1 统计 GT 标签分布 =========
    # 1) collision / free 统计
    num_collision = np.sum(lab == 0)
    num_free = np.sum(lab == 1)
    # 为防止潜在异常值，顺便数一下其它标签
    num_other = n_points - num_collision - num_free

    print("=== 碰撞标签分布 (labels) ===")
    print(f"collision (label=0): {num_collision} ({num_collision / n_points:.4f})")
    print(f"free      (label=1): {num_free} ({num_free / n_points:.4f})")
    if num_other > 0:
        print(f"其它标签(≠0/1):      {num_other} ({num_other / n_points:.4f})")
    print("====================================\n")

    # 2) path soft label 统计
    print("=== Path Soft Label 分布 (pathlabels) ===")
    print(f"min   = {path_soft.min():.6f}")
    print(f"max   = {path_soft.max():.6f}")
    print(f"mean  = {path_soft.mean():.6f}")
    print(f"median= {np.median(path_soft):.6f}")

    # 一些分位数
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    qs = np.percentile(path_soft, percentiles)
    print("\n分位数：")
    for p, q in zip(percentiles, qs):
        print(f"  {p:>3d}%: {q:.6f}")

    # 按区间统计
    bins = [0.0, 1e-3, 0.1, 0.5, 1.0 + 1e-6]  # 最后加一点点防止浮点边界问题
    bin_names = [
        "[0, 1e-3)",
        "[1e-3, 0.1)",
        "[0.1, 0.5)",
        "[0.5, 1.0]"
    ]
    hist = np.histogram(path_soft, bins=bins)[0]

    print("\n区间统计：")
    for name, cnt in zip(bin_names, hist):
        print(f"  {name:>10s}: {cnt:6d} ({cnt / n_points:.4f})")

    # near-path（基于阈值）统计
    mask_path_near = path_soft >= path_near_thr
    num_path_near = np.sum(mask_path_near)
    print(f"\n基于阈值 path_near_thr={path_near_thr}:")
    print(f"  near-path 点数 = {num_path_near} / {n_points} "
          f"({num_path_near / n_points:.4f})")
    print("====================================\n")

    # ========= 3. 随机选关节维度 =========
    if joints is None:
        if D < 3:
            raise ValueError(f"维度 D={D} < 3，无法随机选3个关节进行3D可视化")
        joints = np.random.choice(D, size=3, replace=False)
    else:
        joints = np.array(joints, dtype=int)
        if joints.shape[0] != 3:
            raise ValueError(f"joints 长度必须为3，当前为 {joints.shape[0]}")
        if np.any(joints < 0) or np.any(joints >= D):
            raise ValueError(f"关节索引越界，合法范围 0~{D-1}，得到 {joints}")

    print(f"可视化使用关节维度 (作为 XYZ) = {joints.tolist()}")

    # ========= 4. 投影到选中的3维 =========
    pc_proj = q_pc[:, joints]        # (npoints, 3)
    path_proj = path[:, joints]      # (T, 3)
    start_proj = start[joints]       # (3,)
    goal_proj = goal[joints]         # (3,)

    print("\n=== 数据范围检查（归一化空间） ===")
    print(f"PC[{sample_idx}] 范围:")
    for i, j in enumerate(joints):
        print(f"  joint {j}: pc_proj[:,{i}].min={pc_proj[:, i].min():.4f}, "
              f"max={pc_proj[:, i].max():.4f}")

    print(f"\nPATH[{sample_idx}] 范围:")
    for i, j in enumerate(joints):
        print(f"  joint {j}: path_proj[:,{i}].min={path_proj[:, i].min():.4f}, "
              f"max={path_proj[:, i].max():.4f}")
    print("====================================\n")
    print("=== 路径是否落在点云范围内（逐维检查） ===")
    for i, j in enumerate(joints):
        pc_min, pc_max = pc_proj[:, i].min(), pc_proj[:, i].max()
        path_min, path_max = path_proj[:, i].min(), path_proj[:, i].max()

        inside = (path_min >= pc_min) and (path_max <= pc_max)
        status = "✔ 在范围内" if inside else "✘ 超出范围!"

        print(
            f"joint {j}: "
            f"path_min={path_min:.4f}, path_max={path_max:.4f} | "
            f"pc_min={pc_min:.4f}, pc_max={pc_max:.4f} --> {status}"
        )
    print("====================================\n")

    # ========= 5. 构造 mask =========
    mask_collision = (lab == 0)
    mask_free = (lab == 1)
    mask_path = mask_path_near  # 由 soft label 阈值决定

    print(f"near-path (soft>={path_near_thr}) 点的数量: "
          f"{mask_path.sum()} / {n_points}\n")

    # ========= 6. 画图 =========
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 自由空间点
    if np.any(mask_free):
        ax.scatter(
            pc_proj[mask_free, 0],
            pc_proj[mask_free, 1],
            pc_proj[mask_free, 2],
            s=4, alpha=0.25, label="free space"
        )

    # 碰撞点
    if np.any(mask_collision):
        ax.scatter(
            pc_proj[mask_collision, 0],
            pc_proj[mask_collision, 1],
            pc_proj[mask_collision, 2],
            s=4, alpha=0.6, marker="x", label="collision"
        )

    # 路径附近点（基于 soft label 阈值）
    if np.any(mask_path):
        ax.scatter(
            pc_proj[mask_path, 0],
            pc_proj[mask_path, 1],
            pc_proj[mask_path, 2],
            s=8, alpha=0.9, label=f"path neighborhood (soft>={path_near_thr})"
        )

    # 最优路径轨迹
    ax.plot(
        path_proj[:, 0],
        path_proj[:, 1],
        path_proj[:, 2],
        linewidth=2.0,
        label="best path"
    )

    # 起点 / 终点
    ax.scatter(
        [start_proj[0]], [start_proj[1]], [start_proj[2]],
        s=60, marker="o", label="start"
    )
    ax.scatter(
        [goal_proj[0]], [goal_proj[1]], [goal_proj[2]],
        s=60, marker="^", label="goal"
    )

    ax.set_xlabel(f"joint {joints[0]}")
    ax.set_ylabel(f"joint {joints[1]}")
    ax.set_zlabel(f"joint {joints[2]}")
    ax.set_title(f"Sample {sample_idx} in {npz_path}")

    ax.legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"图已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="统计并可视化 npz 中某个样本的关节空间点云 + GT 标签分布 + 最优路径"
    )
    parser.add_argument(
        "--npz",
        default="data/liche/test/test.npz",
        help="npz 文件路径，例如 data/liche/train/train.npz"
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="样本索引（默认随机）"
    )
    parser.add_argument(
        "--joints",
        type=int,
        nargs=3,
        default=None,
        help="指定3个关节索引，例如 --joints 0 2 5；默认随机"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="如果指定则保存图片到该路径，而不是仅仅显示"
    )
    parser.add_argument(
        "--path_thr",
        type=float,
        default=0.1,
        help="判定 near-path 的 soft label 阈值，默认 0.1"
    )
    args = parser.parse_args()

    visualize_sample(
        npz_path=args.npz,
        sample_idx=args.idx,
        joints=args.joints,
        save_path=args.save,
        show=(args.save is None),  # 如果有保存路径，可以只保存不 show
        path_near_thr=args.path_thr
    )


if __name__ == "__main__":
    main()
