import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def analyze_npz(npz_path):
    data = np.load(npz_path)

    pc = data["pc"]          # (B, N, D)  normalized
    astar = data["astar"]    # (B, N)     0/1
    tokens = data["token"]   # (B,)

    B, N, D = pc.shape

    path_counts = astar.sum(axis=1)
    path_ratios = path_counts / N

    print(f"Loaded {B} samples, N={N}, D={D}\n")

    print("Path ratio statistics:")
    print(f"  mean   : {path_ratios.mean()*100:.2f}%")
    print(f"  std    : {path_ratios.std()*100:.2f}%")
    print(f"  min    : {path_ratios.min()*100:.2f}%")
    print(f"  25%    : {np.percentile(path_ratios, 25)*100:.2f}%")
    print(f"  median : {np.percentile(path_ratios, 50)*100:.2f}%")
    print(f"  75%    : {np.percentile(path_ratios, 75)*100:.2f}%")
    print(f"  max    : {path_ratios.max()*100:.2f}%")

    return pc, astar, tokens, path_ratios


def visualize_sample_3d(pc, astar, sample_idx=0, dims=(0, 1, 2)):
    """
    pc: (B, N, D)
    astar: (B, N)
    dims: which three dimensions to plot
    """
    pts = pc[sample_idx]
    mask = astar[sample_idx].astype(bool)

    d0, d1, d2 = dims

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pts[~mask, d0],
        pts[~mask, d1],
        pts[~mask, d2],
        s=4,
        c="lightgray",
        alpha=0.4,
        label="non-path",
    )

    ax.scatter(
        pts[mask, d0],
        pts[mask, d1],
        pts[mask, d2],
        s=10,
        c="red",
        alpha=0.9,
        label="path",
    )

    ax.set_xlabel(f"joint {d0} (normalized)")
    ax.set_ylabel(f"joint {d1} (normalized)")
    ax.set_zlabel(f"joint {d2} (normalized)")

    ax.set_title(
        f"Sample {sample_idx} | path ratio = {mask.mean()*100:.2f}%"
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


def find_extreme_samples(tokens, ratios, top_k=5):
    idx_sorted = np.argsort(ratios)

    print("\nLowest path ratios:")
    for i in idx_sorted[:top_k]:
        print(f"  {tokens[i]} : {ratios[i]*100:.2f}%")

    print("\nHighest path ratios:")
    for i in idx_sorted[-top_k:]:
        print(f"  {tokens[i]} : {ratios[i]*100:.2f}%")


if __name__ == "__main__":
    pc, astar, tokens, ratios = analyze_npz("data/liche/train.npz")

    # 示例：3D 可视化
    visualize_sample_3d(pc, astar, sample_idx=30, dims=(0, 1, 2))
    visualize_sample_3d(pc, astar, sample_idx=30, dims=(1, 2, 3))

    find_extreme_samples(tokens, ratios)
