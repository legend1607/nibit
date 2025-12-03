import math
import time
import heapq
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


INF = float("inf")


class IRRTStar:
    """
    Informed RRT* 实现（任意维），直接使用 env 的碰撞检测与采样接口。

    依赖的 env 接口约定（与 bit_star.py 一致）：
        - env.bound : list / array，可 reshape 为 (2, dim) 再转置得到 [ [xmin,xmax], [ymin,ymax], ... ]
        - env.config_dim : int，配置空间维度
        - env.sample_empty_points() -> np.ndarray (dim,)
              在可行空间内随机采样一个无碰撞点
        - env._state_fp(state: np.ndarray) -> bool
              状态是否在 free space
        - env._edge_fp(p1: np.ndarray, p2: np.ndarray) -> bool
              p1 -> p2 的线段是否无碰撞
        - env.collision_check_count : int
              碰撞检测计数器
        - （可选，用于可视化）env.rect_obstacles / env.circle_obstacles
    """

    def __init__(
        self,
        start,
        goal,
        environment,
        step_len: float,
        iter_max: int,
        search_radius: Optional[float] = None,
        plot_flag: bool = False,
    ):
        self.env = environment

        self.start = tuple(start)
        self.goal = tuple(goal)
        self.iter_max = iter_max
        self.step_len = float(step_len)

        # 空间信息
        self.bounds = np.array(self.env.bound).reshape((2, -1)).T  # shape: (dim, 2)
        self.dimension = self.env.config_dim
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]

        # 邻域搜索半径（RRT* 用）
        # 若未指定，则设为对角线的 1/5
        if search_radius is None:
            diag = np.linalg.norm(self.ranges)
            self.search_radius = diag / 5.0
        else:
            self.search_radius = float(search_radius)

        # 树结构：顶点列表 + 父指针 + 从 start 的 cost
        self.vertices: List[Tuple[float, ...]] = [self.start]
        self.parents: Dict[Tuple[float, ...], Optional[Tuple[float, ...]]] = {
            self.start: None
        }
        self.costs: Dict[Tuple[float, ...], float] = {self.start: 0.0}

        # 采样点记录（仅统计/可视化用）
        self.samples: List[Tuple[float, ...]] = []

        # 当前最优解
        self.c_best: float = INF
        self.best_goal: Optional[Tuple[float, ...]] = None

        # Informed 采样参数
        self.c_min = self.distance(self.start, self.goal)
        self.center_point: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.informed_sample_init()

        # 采样计数
        self.T = 0

        # 是否中间可视化
        self.plot_flag = plot_flag

    # ===================== 基本工具函数 =====================

    def distance(self, p, q) -> float:
        return float(np.linalg.norm(np.array(p) - np.array(q)))

    def is_state_free(self, p) -> bool:
        return bool(self.env._state_fp(np.array(p, dtype=float)))

    def is_edge_free(self, p, q) -> bool:
        return bool(self.env._edge_fp(np.array(p, dtype=float), np.array(q, dtype=float)))

    # ===================== Informed 采样相关 =====================

    def informed_sample_init(self):
        """
        初始化椭圆采样所需的旋转矩阵 C 和中心点，支持任意维度。
        与 bit_star 中 BITStar.informed_sample_init 的思路一致。
        """
        try:
            if not np.isfinite(self.c_min) or self.c_min <= 1e-12:
                self.center_point = None
                self.C = None
                return
        except Exception:
            self.center_point = None
            self.C = None
            return

        self.center_point = np.array(
            [(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)],
            dtype=float,
        )
        a1 = np.array(self.goal, dtype=float) - np.array(self.start, dtype=float)
        norm_a1 = np.linalg.norm(a1)
        if norm_a1 < 1e-12:
            self.C = None
            return
        a1 /= norm_a1  # unit direction

        # 构造旋转矩阵 C：第一列为 a1，其余列用随机矩阵 QR 正交化
        X = np.random.randn(self.dimension, self.dimension)
        X[:, 0] = a1
        Q, _ = np.linalg.qr(X)
        if np.dot(Q[:, 0], a1) < 0:
            Q[:, 0] *= -1
        self.C = Q

    def sample_unit_ball(self) -> np.ndarray:
        """
        在 n 维单位球内均匀采样一点。
        """
        u = np.random.normal(0, 1, self.dimension)
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            return self.sample_unit_ball()
        r = np.random.random() ** (1.0 / self.dimension)
        return r * u / norm

    def sample_informed_ellipsoid(self, c_best: float) -> np.ndarray:
        """
        在 Informed RRT* 的椭球内采样。
        若 c_best 不合法或比 c_min 小，则回退给 uniform 采样。
        """
        if (
            c_best is None
            or not np.isfinite(c_best)
            or c_best <= self.c_min
            or self.center_point is None
            or self.C is None
            or self.c_min is None
            or self.c_min <= 1e-12
        ):
            # 回退：全局 uniform 采样
            return np.array(self.env.sample_empty_points(), dtype=float)

        # 长轴 a、短轴 b（其余维度同 b）
        a = c_best / 2.0
        val = max(0.0, c_best * c_best - self.c_min * self.c_min)
        b = math.sqrt(val) / 2.0
        L = np.diag([a] + [b] * (self.dimension - 1))

        while True:
            x_ball = self.sample_unit_ball()
            x_ellipsoid = self.C @ (L @ x_ball) + self.center_point
            # 直接用 env._state_fp 检查可行性
            if self.env._state_fp(x_ellipsoid):
                return x_ellipsoid

    def sample_free(self) -> Tuple[float, ...]:
        """
        根据当前 c_best 决定采用 informed 椭球采样还是全局 uniform 采样。
        """
        if (
            self.c_best is not None
            and np.isfinite(self.c_best)
            and self.center_point is not None
            and self.C is not None
        ):
            p = self.sample_informed_ellipsoid(self.c_best)
        else:
            p = np.array(self.env.sample_empty_points(), dtype=float)
        self.T += 1
        return tuple(p.tolist())

    # ===================== RRT* 树操作 =====================

    def nearest_neighbor(self, x_rand: Tuple[float, ...]) -> Tuple[Tuple[float, ...], int]:
        d_min = INF
        idx_best = 0
        for i, v in enumerate(self.vertices):
            d = self.distance(v, x_rand)
            if d < d_min:
                d_min = d
                idx_best = i
        return self.vertices[idx_best], idx_best

    def steer(self, x_near: Tuple[float, ...], x_rand: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        从 x_near 朝向 x_rand 步进一步，步长不超过 step_len。
        """
        v_near = np.array(x_near, dtype=float)
        v_rand = np.array(x_rand, dtype=float)
        direction = v_rand - v_near
        dist = np.linalg.norm(direction)
        if dist < 1e-12:
            return x_near
        step = min(self.step_len, dist)
        x_new = v_near + step * direction / dist
        return tuple(x_new.tolist())

    def get_neighbors(self, x_new: Tuple[float, ...]) -> List[Tuple[float, ...]]:
        """
        以 search_radius 为邻域半径做暴力近邻搜索。
        """
        res = []
        for v in self.vertices:
            if v == x_new:
                continue
            if self.distance(v, x_new) <= self.search_radius:
                res.append(v)
        return res

    def is_in_goal_region(self, x: Tuple[float, ...]) -> bool:
        """
        这里简单地使用 step_len 作为 goal 区域半径。
        """
        return self.distance(x, self.goal) <= self.step_len

    def try_connect_goal(self, x_new: Tuple[float, ...]):
        """
        尝试从 x_new 连接到 goal，若成功且更优，则更新当前最优解。
        """
        if not self.is_in_goal_region(x_new):
            return
        if not self.is_edge_free(x_new, self.goal):
            return

        new_cost = self.costs[x_new] + self.distance(x_new, self.goal)

        if new_cost < self.c_best:
            self.c_best = new_cost
            goal_pt = self.goal
            if goal_pt not in self.vertices:
                self.vertices.append(goal_pt)
            self.parents[goal_pt] = x_new
            self.costs[goal_pt] = new_cost
            self.best_goal = goal_pt

    # ===================== 主规划 =====================

    def extract_path(self) -> List[Tuple[float, ...]]:
        """
        从最优 goal 回溯得到路径。
        """
        if self.best_goal is None or self.best_goal not in self.parents:
            return []
        path = []
        cur = self.best_goal
        while cur is not None:
            path.append(cur)
            cur = self.parents.get(cur, None)
        path.reverse()
        return path

    def planning(self, visualize: bool = False, refresh_interval: int = 10):
        """
        运行 Informed RRT* 搜索。

        返回:
            path: List[Tuple]         最优路径（start -> goal）
            samples: List[Tuple]      采样点列表
            edges: Dict[child] = parent  树的所有边
            n_collision_checks: int   新增的碰撞检测次数
            best_cost: float          最优路径 cost（若无解为 INF）
            T: int                    采样次数
            elapsed: float            规划用时（秒）
        """
        collision_checks0 = getattr(self.env, "collision_check_count", 0)
        t0 = time.time()
        iteration_costs = []

        # 初始化可视化
        if visualize and self.dimension >= 2:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
            ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
            ax.set_title("IRRT* Path Planning")
            ax.plot(self.start[0], self.start[1], "go", markersize=8, label="Start")
            ax.plot(self.goal[0], self.goal[1], "ro", markersize=8, label="Goal")
            for rx, ry, rw, rh in getattr(self.env, "rect_obstacles", []):
                ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color="gray", alpha=1.0))
            for cx, cy, r in getattr(self.env, "circle_obstacles", []):
                ax.add_patch(plt.Circle((cx, cy), r, color="gray", alpha=1.0))
            ax.legend()
        else:
            fig = ax = None

        for it in range(self.iter_max):
            # 1) 采样
            x_rand = self.sample_free()
            self.samples.append(x_rand)

            # 2) 最近邻
            x_near, _ = self.nearest_neighbor(x_rand)

            # 3) 扩展一步
            x_new = self.steer(x_near, x_rand)

            # 检查状态 + 边是否可行
            if (not self.is_state_free(x_new)) or (not self.is_edge_free(x_near, x_new)):
                continue

            # 4) 插入新节点，先设置默认父节点为 x_near
            self.vertices.append(x_new)
            self.parents[x_new] = x_near
            self.costs[x_new] = self.costs[x_near] + self.distance(x_near, x_new)

            # 5) 邻域内重新选择更优 parent
            neighbors = self.get_neighbors(x_new)
            best_parent = x_near
            best_cost = self.costs[x_new]

            for v in neighbors:
                if not self.is_edge_free(v, x_new):
                    continue
                new_cost = self.costs[v] + self.distance(v, x_new)
                if new_cost < best_cost:
                    best_parent = v
                    best_cost = new_cost

            # 更新 parent & cost
            self.parents[x_new] = best_parent
            self.costs[x_new] = best_cost

            # 6) Rewire：看看 x_new 能否作为别人的更优 parent
            for v in neighbors:
                if v == best_parent:
                    continue
                if not self.is_edge_free(x_new, v):
                    continue
                # 通过 x_new 到 v 的 cost
                new_cost = self.costs[x_new] + self.distance(x_new, v)
                if new_cost + 1e-9 < self.costs.get(v, INF):
                    self.parents[v] = x_new
                    self.costs[v] = new_cost
                    # 这里没有递归传播子树成本，简单版本

            # 7) 尝试连接 goal，更新 c_best
            self.try_connect_goal(x_new)
            iteration_costs.append(self.c_best)

            # 8) 可视化
            if visualize and it % refresh_interval == 0 and ax is not None:
                ax.clear()
                ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
                ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"IRRT* Iteration {it}/{self.iter_max}")

                # 障碍
                for rx, ry, rw, rh in getattr(self.env, "rect_obstacles", []):
                    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color="gray", alpha=1.0))
                for cx, cy, r in getattr(self.env, "circle_obstacles", []):
                    ax.add_patch(plt.Circle((cx, cy), r, color="gray", alpha=1.0))

                # 树边
                for child, parent in self.parents.items():
                    if parent is None:
                        continue
                    ax.plot(
                        [parent[0], child[0]],
                        [parent[1], child[1]],
                        linewidth=0.5,
                    )

                # 路径
                path = self.extract_path()
                if len(path) > 1:
                    arr = np.array(path)
                    ax.plot(arr[:, 0], arr[:, 1], "r-", lw=2, label="Best Path")

                # 起终点
                ax.plot(self.start[0], self.start[1], "go", markersize=8)
                ax.plot(self.goal[0], self.goal[1], "ro", markersize=8)

                # Informed 椭圆轮廓（仅展示前 2 维）
                if (
                    self.C is not None
                    and self.center_point is not None
                    and np.isfinite(self.c_best)
                    and self.c_best > self.c_min
                    and self.dimension >= 2
                ):
                    val = max(0.0, self.c_best**2 - self.c_min**2)
                    a = self.c_best / 2.0
                    b = math.sqrt(val) / 2.0
                    # 只画前两个维度的投影
                    L2 = np.diag([a, b])
                    theta = np.linspace(0, 2 * math.pi, 200)
                    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
                    # 只取 C 的前两行、前两列
                    C2 = self.C[:2, :2]
                    center2 = self.center_point[:2]
                    ellipse_pts = (C2 @ (L2 @ unit_circle)) + center2[:, None]
                    ax.plot(ellipse_pts[0, :], ellipse_pts[1, :], "m--", lw=1.5)

                ax.legend()
                plt.pause(0.001)

        elapsed = time.time() - t0
        n_collision = getattr(self.env, "collision_check_count", 0) - collision_checks0
        path = self.extract_path()

        # BIT* 返回的是 edges: child -> parent
        edges = {child: parent for child, parent in self.parents.items() if parent is not None}

        # 最后一次迭代编号（0-based；若一个迭代都没跑，则为 -1）
        final_iter = len(iteration_costs) - 1 if len(iteration_costs) > 0 else -1

        return (
            path,
            self.samples,
            edges,
            n_collision,
            self.c_best,
            self.T,
            elapsed,
            final_iter,
            iteration_costs,
        )


def get_irrt_planner(
    args,
    problem,
    neural_wrapper=None,
):
    """
    IRRT* 路径规划器工厂函数。
    用法风格参考 bit_star.get_bit_planner。
    """
    step_len = getattr(args, "step_len", 1.0)
    search_radius = getattr(args, "search_radius", None)

    planner = IRRTStar(
        problem["start"],
        problem["goal"],
        problem["env"],
        step_len=step_len,
        iter_max=args.iter_max,
        search_radius=search_radius,
        plot_flag=getattr(args, "plot_flag", False),
    )

    return planner
