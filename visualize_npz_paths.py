import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 需要以启用 3D


def visualize_sample(npz_path, sample_idx=None, joints=None, save_path=None, show=True):
    """
    可视化一个样本：
      - 全部点云（区分碰撞 / 自由 / 路径点）
      - 最优路径轨迹
      - 随机选3个关节作为坐标轴

    参数
    ----
    npz_path : str
        你的 train/val/test npz 文件路径
    sample_idx : int or None
        样本索引；None 时随机选
    joints : list[int] or np.ndarray or None
        长度为3的关节索引列表，例如 [0, 2, 5]；
        None 时随机选3个不重复关节
    save_path : str or None
        如果不为 None，则保存到该路径（如 "vis.png"）
    show : bool
        是否 plt.show()
    """

    # ========= 1. 读数据 =========
    data = np.load(npz_path, allow_pickle=True)

    pc = data["pc"]          # (N, npoints, D)  归一化后的关节空间点云
    labels = data["labels"]  # (N, npoints)     0: 碰撞, 1: free, 2: path 邻域
    paths = data["paths"]    # (N,) object，每个是 (T, D) 的轨迹
    starts = data["starts"]  # (N, D)
    goals = data["goals"]    # (N, D)

    num_samples = pc.shape[0]

    # ========= 2. 选样本 =========
    if sample_idx is None:
        sample_idx = np.random.randint(num_samples)
    if not (0 <= sample_idx < num_samples):
        raise ValueError(f"sample_idx 超出范围: 0 ~ {num_samples-1}, got {sample_idx}")

    q_pc = pc[sample_idx]          # (npoints, D)
    lab = labels[sample_idx]       # (npoints,)
    path = np.array(paths[sample_idx])  # (T, D)
    start = starts[sample_idx]     # (D,)
    goal = goals[sample_idx]       # (D,)

    n_points, D = q_pc.shape

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

    print(f"可视化样本 index = {sample_idx}")
    print(f"使用关节维度 (作为 XYZ) = {joints.tolist()}")

    # ========= 4. 投影到选中的3维 =========
    pc_proj = q_pc[:, joints]        # (npoints, 3)
    path_proj = path[:, joints]      # (T, 3)
    start_proj = start[joints]       # (3,)
    goal_proj = goal[joints]         # (3,)
    print("\n=== 数据范围检查（归一化空间） ===")
    print(f"PC[{sample_idx}] 范围:")
    for i, j in enumerate(joints):
        print(f"  joint {j}: pc_proj[:,{i}].min={pc_proj[:,i].min():.4f}, max={pc_proj[:,i].max():.4f}")

    print(f"\nPATH[{sample_idx}] 范围:")
    for i, j in enumerate(joints):
        print(f"  joint {j}: path_proj[:,{i}].min={path_proj[:,i].min():.4f}, max={path_proj[:,i].max():.4f}")
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
    mask_path = (lab == 2)
    print(f"label=2 (path neighborhood) 点的数量: {mask_path.sum()} / {n_points}")

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
            s=4, alpha=0.5, marker="x", label="collision"
        )

    # 路径附近点（标签为 2）
    if np.any(mask_path):
        ax.scatter(
            pc_proj[mask_path, 0],
            pc_proj[mask_path, 1],
            pc_proj[mask_path, 2],
            s=8, alpha=0.9, label="path neighborhood"
        )

    # 最优路径轨迹 (BIT* 求得)
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
        description="可视化 npz 中的关节空间点云 + 最优路径（随机选3个关节维度）"
    )
    parser.add_argument("--npz", default="data/liche/test/test.npz",help="npz 文件路径，例如 data/liche/train/train.npz")
    parser.add_argument("--idx", type=int, default=None,
                        help="样本索引（默认随机）")
    parser.add_argument("--joints", type=int, nargs=3, default=None,
                        help="指定3个关节索引，例如 --joints 0 2 5；默认随机")
    parser.add_argument("--save", type=str, default=None,
                        help="如果指定则保存图片到该路径，而不是仅仅显示")
    args = parser.parse_args()

    visualize_sample(
        npz_path=args.npz,
        sample_idx=args.idx,
        joints=args.joints,
        save_path=args.save,
        show=(args.save is None)  # 如果有保存路径，可以只保存不 show
    )


if __name__ == "__main__":
    main()
