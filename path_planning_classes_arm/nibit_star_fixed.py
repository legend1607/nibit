import numpy as np
import os
import sys
# 将项目根目录加入 Python 搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import math
import yaml
import heapq
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch
from shapely import affinity
import itertools
from environment.timer import Timer
import torch
import random

INF = float("inf")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 触发 3D 投影注册


class NIBITStar:
    def __init__(
        self,
        start,
        goal,
        environment,
        iter_max,
        batch_size,
        neural_wrapper,
        plot_flag=False,
        timer=None,
    ):
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.env = environment
        self.neural_wrapper = neural_wrapper

        # ---------- 关键：统一将 start/goal 转为可哈希 key ----------
        self.start = self.to_key(start)
        self.goal = self.to_key(goal)

        self.bounds = self.env.bound
        self.bounds = np.array(self.bounds).reshape((2, -1)).T
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.dimension = environment.config_dim

        # This is the tree (所有点均以 tuple key 表示)
        self.vertices = []
        self.edges = dict()  # key = point(tuple)， value = parent(tuple)
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()

        self.r = INF
        self.iter_max = iter_max
        self.batch_size = batch_size
        self.T = 0
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        # the parameters for informed sampling
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # whether plot the middle planning process
        self.plot_planning_process = plot_flag

        self.n_collision_points = 0
        self.n_free_points = 2
        self.path = []
        # ========= 自适应 near-path 距离阈值 =========
        self.free_th = 0.6   # P(free) 初始阈值（过严会在采样阶段自适应下调）
        self.free_th_min = 0.4
        self.free_th_relax = 0.90  # free 过滤为空时：free_th *= relax

        # ===== path 阈值（与 single_test 的 path_threshold 对齐）=====
        self.tau_init = 0.1      # 初始 path 阈值
        self.tau = 0.7      # 初始 path 阈值
        self.tau_max = 0.90   # 有初始解后逐步提高到这里
        self.tau_inc = 0.05   # 每次迭代提高的步长（可调）

        # 采样配比：从 (p_path>=tau) 与 (p_path<tau) 两侧按比例取点
        self.path_high_ratio = 0.40  # 阈值之上取点比例；阈值之下随机补齐
        self.has_solution = False
        self.vis_debug = False
        self.debug_vis = False      # 默认关
        self.vis_joints =  [0,1,2]      # 或固定 [0,1,2]
        self.vis_save_path = None   # 例如 "vis.png"

    def  to_key(self, point, ndigits=6):
        """
     将点转为 tuple(key)。
        """
        if point is None:
            raise ValueError("Cannot convert None to key")

        if isinstance(point, tuple):
            return tuple([float(round(x, ndigits)) for x in point])
        if isinstance(point, np.ndarray):
            arr = np.round(point.astype(float), ndigits)
            return tuple(arr.tolist())
        try:
            arr = np.round(np.array(point, dtype=float), ndigits)
            return tuple(arr.tolist())
        except Exception as e:
            raise ValueError(f"Cannot convert {type(point)} to key: {e}")

    # ---------------- planning setup ----------------
    def setup_planning(self):
        # add goal to the samples (as tuple key)
        if self.goal not in self.samples:
            self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree (as tuple key)
        if self.start not in self.vertices:
            self.vertices.append(self.start)
        self.g_scores[self.start] = 0

        # Computing the sampling space
        self.informed_sample_init()
        radius_constant = self.radius_init()

        return radius_constant

    def radius_init(self):
        from scipy import special

        # Hypersphere radius calculation
        n = self.dimension
        unit_ball_volume = np.pi ** (n / 2.0) / special.gamma(n / 2.0 + 1)
        # avoid division by zero
        denom = (self.n_collision_points + self.n_free_points)
        if denom == 0:
            denom = 1.0
        volume = np.abs(np.prod(self.ranges)) * self.n_free_points / denom
        gamma = (1.0 + 1.0 / n) * volume / unit_ball_volume
        radius_constant = 2 * self.eta * (gamma ** (1.0 / n))
        return radius_constant

    def informed_sample_init(self):
        """
        初始化椭圆采样所需的旋转矩阵 C 和中心点。
        支持任意维度。
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

        # center_point 使用数值形式（numpy array）
        self.center_point = np.array(
            [(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)]
        )
        # 长轴方向
        a1 = np.array(self.goal, dtype=float) - np.array(self.start, dtype=float)
        norm_a1 = np.linalg.norm(a1)
        if norm_a1 < 1e-12:
            self.C = None
            return
        a1 /= norm_a1  # 单位向量

        # 构造旋转矩阵 C
        X = np.random.randn(self.dimension, self.dimension)
        X[:, 0] = a1
        Q, _ = np.linalg.qr(X)
        if np.dot(Q[:, 0], a1) < 0:
            Q[:, 0] *= -1
        self.C = Q

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.dimension)
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def _vis_pointcloud_split(
        self,
        joints_np,
        p_free,
        p_path,
        free_mask,
        high_idx,
        low_idx,
        selected_idx=None,
        free_th=None,
        tau=None,
        title="nn split",
        block=False,
    ):
        """
        joints_np: (N, D) 至少前三维是 xyz
        p_free, p_path: (N,)
        free_mask: (N,) bool
        high_idx, low_idx: index array
        selected_idx: list[int] 最终选中点索引（可选）
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as e:
            print("[vis] matplotlib not available:", e)
            return

        pts = np.asarray(joints_np, dtype=float)
        if pts.shape[1] < 3:
            print("[vis] need at least 3 dims for xyz, got", pts.shape)
            return

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        p_free = np.asarray(p_free).reshape(-1)
        p_path = np.asarray(p_path).reshape(-1)

        free_th = float(self.free_th if free_th is None else free_th)
        tau = float(self.tau if tau is None else tau)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        # 1) 非 free 的点（灰色）
        nonfree = np.where(~free_mask)[0]
        if len(nonfree) > 0:
            ax.scatter(x[nonfree], y[nonfree], z[nonfree], s=6, alpha=0.15, label="not free")

        # 2) free & low（蓝色）
        if len(low_idx) > 0:
            ax.scatter(x[low_idx], y[low_idx], z[low_idx], s=10, alpha=0.5, label="free & low")

        # 3) free & high（橙色）
        if len(high_idx) > 0:
            ax.scatter(x[high_idx], y[high_idx], z[high_idx], s=14, alpha=0.8, label="free & high")

        # 4) 最终选中（红色，描边）
        if selected_idx is not None and len(selected_idx) > 0:
            sel = np.array(selected_idx, dtype=int)
            ax.scatter(
                x[sel], y[sel], z[sel],
                s=60, alpha=0.95, label="selected",
                edgecolors="k", linewidths=0.6
            )

        ax.set_title(
            f"{title}\nfree_th={free_th:.3f}, tau={tau:.3f}, "
            f"N={len(pts)}, free={int(np.sum(free_mask))}, high={len(high_idx)}, low={len(low_idx)}"
        )
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(loc="upper right")

        # 可选：把概率信息写在图外（避免挡住点云）
        txt = (
            f"p_free: min={p_free.min():.3f}, mean={p_free.mean():.3f}, max={p_free.max():.3f}\n"
            f"p_path: min={p_path.min():.3f}, mean={p_path.mean():.3f}, max={p_path.max():.3f}"
        )
        fig.text(0.02, 0.02, txt, fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    def sample_from_env(self, c_best, batch_size):
        """
        使用 informed sampling（椭圆 + 全局）产生候选关节点，
        不做碰撞检测，统一交给神经网络打分：
        - 使用 P(path) 做 Top-K 选点
        """

        def select_with_nn(tau,candidates):
            if len(candidates) == 0:
                return []

            joints_np = np.array(candidates, dtype=float)

            # 没网络：随机
            if self.neural_wrapper is None:
                if len(joints_np) > batch_size:
                    idx = np.random.choice(len(joints_np), size=batch_size, replace=False)
                    joints_np = joints_np[idx]
                return [self.to_key(p) for p in joints_np]

            K = min(batch_size, len(joints_np))

            # ===== 统一一次 forward 拿到概率（推荐 wrapper 提供 predict_probs）=====
            if hasattr(self.neural_wrapper, "predict_probs"):
                p_free, p_path = self.neural_wrapper.predict_probs(joints_np)
            else:
                # 兼容旧接口：分别算 mask 时会多跑一次 forward（但仍保持语义一致）
                free_mask_tmp, p_free = self.neural_wrapper.get_free_mask(joints_np, prob_th=0.0)
                _, p_path = self.neural_wrapper.get_path_mask(joints_np, prob_th=0.0)

            p_free = np.asarray(p_free, dtype=float).reshape(-1)
            p_path = np.asarray(p_path, dtype=float).reshape(-1)

            # ===== 1) 自适应 free 过滤：如果一个 free 都没有就下调 free_th =====
            free_th = float(self.free_th)
            free_mask = (p_free >= free_th)

            while (not np.any(free_mask)) and (free_th > self.free_th_min + 1e-9):
                free_th = max(self.free_th_min, free_th * self.free_th_relax)
                free_mask = (p_free >= free_th)

            # 持久化更新（下一轮采样继续用更“宽松”的阈值）
            self.free_th = free_th

            # 极端情况：仍无 free，就退化为不过滤
            if not np.any(free_mask):
                free_mask = np.ones_like(free_mask, dtype=bool)
            # ===== 2) 按 path 阈值分两侧取点：上侧 Top， 下侧 Random =====
            high_idx = np.where((p_path >= self.tau) & free_mask)[0]
            low_idx  = np.where((p_path <  self.tau) & free_mask)[0]
            if getattr(self, "vis_debug", False):
                self._vis_pointcloud_split(
                    joints_np=joints_np,
                    p_free=p_free,
                    p_path=p_path,
                    free_mask=free_mask,
                    high_idx=high_idx,
                    low_idx=low_idx,
                    selected_idx=None,
                    free_th=self.free_th,
                    tau=self.tau,
                    title="after threshold split",
                    block=False,
                )

            K = min(batch_size, len(joints_np))

            # 目标配比（先算目标，再根据可用数量调整）
            k_high_target = int(round(K * float(self.path_high_ratio)))
            k_low_target  = K - k_high_target

            # 高侧实际能取多少
            k_high = min(k_high_target, len(high_idx))
            # 高侧不足的缺口转给低侧（低侧目标增加）
            k_low = min(k_low_target + (k_high_target - k_high), len(low_idx))

            selected_idx = []

            # 高侧：按 p_path 从大到小取
            if k_high > 0:
                order_high = high_idx[np.argsort(-p_path[high_idx])]
                selected_idx.extend(order_high[:k_high].tolist())

            # 低侧：随机取
            if k_low > 0:
                if len(low_idx) <= k_low:
                    selected_idx.extend(low_idx.tolist())
                else:
                    chosen = np.random.choice(low_idx, size=k_low, replace=False)
                    selected_idx.extend([int(i) for i in chosen])

            # ===== 补齐策略（不改变你原来的兜底）=====
            # 仍不足：在 free 点里按 p_path 从大到小补
            if len(selected_idx) < K:
                free_idx = np.where(free_mask)[0]
                free_sorted = free_idx[np.argsort(-p_path[free_idx])]
                s = set(int(x) for x in selected_idx)
                for i in free_sorted:
                    if len(selected_idx) >= K:
                        break
                    ii = int(i)
                    if ii not in s:
                        selected_idx.append(ii)
                        s.add(ii)

            # 仍不足（极端）：退化到全体按 p_path 补
            if len(selected_idx) < K:
                all_sorted = np.argsort(-p_path)
                s = set(int(x) for x in selected_idx)
                for i in all_sorted:
                    if len(selected_idx) >= K:
                        break
                    ii = int(i)
                    if ii not in s:
                        selected_idx.append(ii)
                        s.add(ii)

            selected = joints_np[np.array(selected_idx, dtype=int)]

            # ===== 3) 调用点云可视化（可选开关，避免每次都弹窗）=====

            return [self.to_key(p) for p in selected]


        candidates = []
        oversample_factor = 10
        M = batch_size * oversample_factor

        # ---------- 情况 1：椭圆信息无效 → 全局 uniform 采样 ----------
        if (
            c_best is None
            or not np.isfinite(c_best)
            or self.c_min is None
            or self.c_min <= 1e-12
            or self.center_point is None
            or self.C is None
        ):
            max_trials = M * 10
            for _ in range(max_trials):
                q = self.env.uniform_sample()
                if q is not None:
                    candidates.append(np.array(q, dtype=float))
                if len(candidates) >= M:
                    break
            return select_with_nn(self.tau_init,candidates)

        # ---------- 情况 2：椭圆信息有效 ----------
        a = c_best / 2.0
        b = math.sqrt(max(0.0, c_best**2 - self.c_min**2)) / 2.0
        L = np.diag([a] + [b] * (self.dimension - 1))

        consecutive_failures = 0
        max_consecutive_failures = 50

        for _ in range(M * 10):
            try:
                x_ball = self.sample_unit_ball()
                x_ellipsoid = self.C @ (L @ x_ball) + self.center_point
                candidates.append(np.array(x_ellipsoid, dtype=float))
                consecutive_failures = 0
            except Exception:
                try:
                    q = self.env.uniform_sample()
                    if q is not None:
                        candidates.append(np.array(q, dtype=float))
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception:
                    consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                break
            if len(candidates) >= M:
                break

        # 不足就 uniform 补
        if len(candidates) < M:
            max_trials = (M - len(candidates)) * 10
            for _ in range(max_trials):
                q = self.env.uniform_sample()
                if q is not None:
                    candidates.append(np.array(q, dtype=float))
                if len(candidates) >= M:
                    break

        return select_with_nn(self.tau,candidates)

    def is_point_free(self, point):
        numeric = np.array(point, dtype=float)
        result = self.env._state_fp(numeric)
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        a = np.array(edge[0], dtype=float)
        b = np.array(edge[1], dtype=float)
        result = self.env._edge_fp(a, b)
        return result

    def get_g_score(self, point):
        if point == self.start:
            return 0
        return self.g_scores.get(point, INF)

    def get_f_score(self, point):
        g = self.get_g_score(point)
        if g == INF:
            g = self.heuristic_cost(self.start, point)
        return g + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        if not self.is_edge_free([point1, point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1, dtype=float) - np.array(point2, dtype=float))

    def get_edge_value(self, edge):
        return (
            self.get_g_score(edge[0])
            + self.heuristic_cost(edge[0], edge[1])
            + self.heuristic_cost(edge[1], self.goal)
        )

    def get_point_value(self, point):
        return self.get_g_score(point) + self.heuristic_cost(point, self.goal)

    def bestVertexQueueValue(self):
        if not self.vertex_queue:
            return INF
        else:
            return self.vertex_queue[0][0]

    def bestEdgeQueueValue(self):
        if not self.edge_queue:
            return INF
        else:
            return self.edge_queue[0][0]

    def get_current_path_points(self):
        """获取当前最优路径上的所有点（用于保护路径不被修剪）"""
        path_points = set()
        if self.get_g_score(self.goal) == INF:
            return path_points

        point = self.goal
        path_points.add(point)

        # 安全地回溯路径
        visited = set()
        max_iterations = len(self.vertices) + 100
        iteration = 0

        while point != self.start and iteration < max_iterations:
            if point in visited:  # 检测循环
                break
            if point not in self.edges:  # 路径断裂
                break

            visited.add(point)
            point = self.to_key(self.edges[point])
            path_points.add(point)
            iteration += 1

        return path_points

    def prune_edge(self, c_best):
        """修剪边，但保护当前最优路径"""
        path_points = self.get_current_path_points()

        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            point = self.to_key(point)
            parent = self.to_key(parent)

            if point in path_points:
                continue

            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point, None)

    def prune(self, c_best):
        """修剪顶点和边，但保护当前最优路径"""
        path_points = self.get_current_path_points()

        # 修剪样本点
        self.samples = [
            point for point in self.samples if self.get_f_score(point) < c_best
        ]

        # 修剪边
        self.prune_edge(c_best)

        # 修剪顶点
        vertices_temp = []
        for point in self.vertices:
            point = self.to_key(point)
            f = self.get_f_score(point)

            if point in path_points:
                vertices_temp.append(point)
                continue

            if f <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)

        self.vertices = vertices_temp

    def expand_vertex(self, point):
        self.timer.start()

        # neighbors among samples within radius
        neigbors_sample = [s for s in self.samples if self.distance(point, s) <= self.r]
        self.timer.finish(Timer.NN)

        self.timer.start()
        # push potential edges (point->neighbor) into edge_queue
        for neighbor in neigbors_sample:
            estimated_f_score = (
                self.get_g_score(point)
                + self.heuristic_cost(point, neighbor)
                + self.heuristic_cost(neighbor, self.goal)
            )
            if estimated_f_score < self.get_g_score(self.goal):
                heapq.heappush(
                    self.edge_queue,
                    (self.get_edge_value((point, neighbor)), (point, neighbor)),
                )

        # neighbors among existing vertices
        if point not in self.old_vertices:
            neigbors_vertex = [
                ver for ver in self.vertices if self.distance(point, ver) <= self.r
            ]
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = (
                        self.get_g_score(point)
                        + self.heuristic_cost(point, neighbor)
                        + self.heuristic_cost(neighbor, self.goal)
                    )
                    if estimated_f_score < self.get_g_score(self.goal):
                        estimated_g_score = (
                            self.get_g_score(point)
                            + self.heuristic_cost(point, neighbor)
                        )
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(
                                self.edge_queue,
                                (
                                    self.get_edge_value((point, neighbor)),
                                    (point, neighbor),
                                ),
                            )

        self.timer.finish(Timer.EXPAND)

    def get_best_path(self):
        """获取最优路径，带有完整的错误检查"""
        path = []
        if self.get_g_score(self.goal) == INF:
            return path

        path.append(self.goal)
        point = self.goal
        visited = set()
        max_iterations = len(self.vertices) + 100
        iteration = 0

        while point != self.start and iteration < max_iterations:
            if point in visited:
                print(f"Warning: Cycle detected at {point}")
                return []
            visited.add(point)

            if point not in self.edges:
                print(f"Warning: Point {point} has no parent in edges")
                print(f"Current point g_score: {self.g_scores.get(point, 'N/A')}")
                print(f"Is point in vertices: {point in self.vertices}")
                print(
                    f"Total edges: {len(self.edges)}, Total vertices: {len(self.vertices)}"
                )
                return []

            parent = self.edges[point]
            parent = self.to_key(parent)
            path.append(parent)
            point = parent
            iteration += 1

        if iteration >= max_iterations:
            print(f"Warning: Path too long ({iteration}), possible error")
            return []

        path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def planning(self, visualize=False, refresh_interval=1):
        """
        返回值格式与增强版 BITStar 对齐：
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
            iteration_times,          # ✅ 新增：每次迭代的累计时间
            first_solution_iter,
            first_solution_cost,
            first_solution_nodes,
            final_solution_nodes,
        )
        """
        no_improve_limit = 100
        no_improve_count = 0
        best_cost = self.get_g_score(self.goal)

        collision_checks = self.env.collision_check_count
        self.setup_planning()

        init_time = time()
        iteration_costs = []
        iteration_times = []   # ✅ 新增

        # 额外记录：第一次解 / 最终解信息
        first_solution_iter = None
        first_solution_cost = None
        first_solution_nodes = None
        final_solution_nodes = 0

        for k in range(self.iter_max):
            final_iter = k

            # 1. 如果队列为空 -> 新采样
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.get_g_score(self.goal)
                self.prune(c_best)
                new_samples = self.sample_from_env(c_best, self.batch_size)
                new_samples = [self.to_key(p) for p in new_samples]
                self.samples.extend(new_samples)
                self.T += self.batch_size

                self.timer.start()
                self.old_vertices = set([self.to_key(v) for v in self.vertices])
                self.vertex_queue = [
                    (self.get_point_value(v), self.to_key(v)) for v in self.vertices
                ]
                heapq.heapify(self.vertex_queue)
                q = len(self.vertices) + len(self.samples)
                if q > 0:
                    self.r = self.radius_init() * (
                        (math.log(q) / q) ** (1.0 / self.dimension)
                    )
                self.timer.finish(Timer.HEAP)

            # 2. 扩展节点
            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    self.timer.start()
                    _, point = heapq.heappop(self.vertex_queue)
                    point = self.to_key(point)
                    self.timer.finish(Timer.HEAP)
                    self.expand_vertex(point)
            except Exception as e:
                if (not self.edge_queue) and (not self.vertex_queue):
                    # 队列都空了，当前迭代无法扩展，记录 cost & time 后继续
                    iteration_costs.append(self.get_g_score(self.goal))
                    iteration_times.append(time() - init_time)   # ✅
                    continue
                else:
                    raise e

            # 3. 选取最优边并扩展树
            if not self.edge_queue:
                # 为了和 BITStar 一致，这里也记录一下当前 cost & time
                iteration_costs.append(self.get_g_score(self.goal))
                iteration_times.append(time() - init_time)       # ✅
                continue

            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)
            bestEdge = (self.to_key(bestEdge[0]), self.to_key(bestEdge[1]))

            if best_edge_value < self.get_g_score(self.goal):
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                self.timer.start()
                actual_f_edge = (
                    self.get_g_score(bestEdge[0])
                    + actual_cost_of_edge
                    + self.heuristic_cost(bestEdge[1], self.goal)
                )
                if actual_f_edge < self.get_g_score(self.goal):
                    actual_g_score_of_point = (
                        self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    )
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        point_key = self.to_key(bestEdge[1])
                        parent_key = self.to_key(bestEdge[0])

                        self.g_scores[point_key] = actual_g_score_of_point
                        self.edges[point_key] = parent_key

                        if point_key not in self.vertices:
                            try:
                                self.samples.remove(point_key)
                            except ValueError:
                                pass
                            self.vertices.append(point_key)
                            heapq.heappush(
                                self.vertex_queue,
                                (self.get_point_value(point_key), point_key),
                            )

                        # prune inconsistent edges
                        self.edge_queue = [
                            item
                            for item in self.edge_queue
                            if self.to_key(item[1][1]) != point_key
                            or self.get_g_score(self.to_key(item[1][0]))
                            + self.heuristic_cost(
                                self.to_key(item[1][0]), self.to_key(item[1][1])
                            )
                            < self.get_g_score(self.to_key(item[1][1]))
                        ]
                        heapq.heapify(self.edge_queue)
                self.timer.finish(Timer.HEAP)
            else:
                self.vertex_queue = []
                self.edge_queue = []

            # 4. Update path & 记录 cost / time / 第一次解
            self.path = self.get_best_path()
            g_goal = self.get_g_score(self.goal)
            iteration_costs.append(g_goal)
            iteration_times.append(time() - init_time)          # ✅

            current_cost = g_goal

            # 第一次可行解
            if first_solution_iter is None and np.isfinite(g_goal):
                first_solution_iter = k + 1  # 1-based
                first_solution_cost = g_goal
                if self.path:
                    first_solution_nodes = len(self.path)
                else:
                    tmp_path = self.get_best_path()
                    first_solution_nodes = len(tmp_path) if tmp_path else None

            if current_cost < best_cost:
                best_cost = current_cost
                no_improve_count = 0
            else:
                no_improve_count += 1
            # ---- 更新 path 阈值 tau（与 single_test 的 path_threshold 对齐）----
            if (not self.has_solution) and np.isfinite(g_goal):
                # 第一次找到可行解：开始逐步“收紧”到更靠近路径的区域（提高阈值）
                self.has_solution = True

            if self.has_solution:
                # 有初始解后：逐步提高阈值，直到 tau_max
                self.tau = min(self.tau_max, self.tau + self.tau_inc)
            else:
                # 没有解前：保持初始阈值（更探索）
                self.tau = float(self.tau)


            # 判定 1：找到直接连接的最优路径
            if (
                len(self.path) == 2
                and self.path[0] == self.start
                and self.path[1] == self.goal
            ):
                break

            # 判定 2：环境允许直接连线，并且路径长度接近直线距离
            direct_free = False
            try:
                direct_free = self.is_edge_free([self.start, self.goal])
            except Exception:
                direct_free = False

            if direct_free:
                current_len = (
                    self.path_length_calculate(self.path)
                    if len(self.path) > 1
                    else INF
                )
                straight_len = self.distance(self.start, self.goal)
                if abs(current_len - straight_len) <= 1e-8:
                    break

        # 最终路径节点数
        final_solution_nodes = len(self.path) if self.path else 0

        return (
            self.path,
            self.samples,
            self.edges,
            self.env.collision_check_count - collision_checks,
            self.get_g_score(self.goal),
            self.T,
            time() - init_time,
            final_iter,
            iteration_costs,
            iteration_times,         # ✅ 新增
            first_solution_iter,
            first_solution_cost,
            first_solution_nodes,
            final_solution_nodes,
        )

def get_bit_planner(
    args,
    problem,
    neural_wrapper=None,
):
    planner = NIBITStar(
        problem["start"],
        problem["goal"],
        problem["env"],
        args.iter_max,
        args.batch_size,
        neural_wrapper,
    )
    return planner
