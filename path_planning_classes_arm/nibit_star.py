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


def visualize_nn_predictions(
    joints_np,
    labels_pred,
    joints=None,
    title="NN predictions in joint space",
    save_path=None,
    show=True,
):
    """
    可视化一批关节空间采样点的“预测标签”：
      - label=0: 碰撞
      - label=1: 自由空间
      - label=2: 路径附近（path）
    """
    joints_np = np.asarray(joints_np, dtype=float)
    labels_pred = np.asarray(labels_pred, dtype=int)

    N, D = joints_np.shape
    if D < 3:
        raise ValueError(f"维度 D={D} < 3，无法做 3D 可视化")

    if joints is None:
        joints = np.random.choice(D, size=3, replace=False)
    else:
        joints = np.array(joints, dtype=int)
        if joints.shape[0] != 3:
            raise ValueError(f"joints 长度必须为3，当前为 {joints.shape[0]}")
        if np.any(joints < 0) or np.any(joints >= D):
            raise ValueError(f"关节索引越界，合法范围 0~{D-1}，得到 {joints}")
    joints = np.sort(joints)

    print(f"[NN VIS] 使用关节维度 (作为 XYZ) = {joints.tolist()}")

    # 投影到选中的 3 维
    pts_proj = joints_np[:, joints]  # (N,3)

    # 构造 mask
    mask_collision = (labels_pred == 0)
    mask_free      = (labels_pred == 1)
    mask_path      = (labels_pred == 2)

    print(f"[NN VIS] 预测 label=2(path) 点数: {mask_path.sum()} / {N}")
    print(f"[NN VIS] 预测 label=1(free) 点数: {mask_free.sum()} / {N}")
    print(f"[NN VIS] 预测 label=0(coll) 点数: {mask_collision.sum()} / {N}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 自由空间点
    if np.any(mask_free):
        ax.scatter(
            pts_proj[mask_free, 0],
            pts_proj[mask_free, 1],
            pts_proj[mask_free, 2],
            c='g', s=4, alpha=0.25, label="NN free (1)"
        )

    # 碰撞点
    if np.any(mask_collision):
        ax.scatter(
            pts_proj[mask_collision, 0],
            pts_proj[mask_collision, 1],
            pts_proj[mask_collision, 2],
            c='r', s=4, alpha=0.5, marker="x", label="NN collision (0)"
        )

    # 路径点
    if np.any(mask_path):
        ax.scatter(
            pts_proj[mask_path, 0],
            pts_proj[mask_path, 1],
            pts_proj[mask_path, 2],
            c='b', s=8, alpha=0.9, label="NN path (2)"
        )

    ax.set_xlabel(f"joint {joints[0]}")
    ax.set_ylabel(f"joint {joints[1]}")
    ax.set_zlabel(f"joint {joints[2]}")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[NN VIS] 图已保存到: {save_path}")

    if show:
        plt.show()


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
        self.tau_init = 1.0      # 初始“近路径”距离阈值（按你的距离尺度调）
        self.tau_min  = 0.05     # 最小阈值（越小越精细）
        self.tau_shrink = 0.98   # 每次迭代默认收缩比例
        self.tau_shrink_fast = 0.90  # 找到解后更快收缩

        self.tau = self.tau_init
        self.has_solution = False
        self.free_th = 0.6   # 判定为 free 的概率阈值，0.5~0.8 可调

    # ---------- helper: convert any point-like to hashable tuple ----------
    def to_key(self, point, ndigits=6):
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

    def sample_from_env(self, c_best, batch_size, vertices=None):
        """
        使用 informed sampling（椭圆 + 全局）产生候选关节点，
        不做碰撞检测，统一交给神经网络打分：
        - 使用 P(path) 做 Top-K 选点
        """

        def select_with_nn(candidates):
            if len(candidates) == 0:
                return []

            joints_np = np.array(candidates, dtype=float)

            # 没网络：随机
            if self.neural_wrapper is None:
                if len(joints_np) > batch_size:
                    idx = np.random.choice(len(joints_np), size=batch_size, replace=False)
                    joints_np = joints_np[idx]
                return [self.to_key(p) for p in joints_np]

            # ===== 1) 预测 collision/free logits 和 距离 =====
            logits = self.neural_wrapper.predict_logits(joints_np)   # (N,2)
            dist_pred = self.neural_wrapper.predict_dist(joints_np)  # (N,)

            logits = np.asarray(logits, dtype=float)
            dist_pred = np.asarray(dist_pred, dtype=float).reshape(-1)

            # ===== 2) softmax 得到 free 概率 =====
            # 避免数值溢出，减掉最大值
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(logits - logits_max)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)  # (N,2)
            p_free = probs[:, 1]  # 假设 class 1 = free

            free_mask = p_free >= self.free_th

            # 如果太严格导致一个 free 都没有，退回不做 free 过滤
            if not np.any(free_mask):
                free_mask = np.ones_like(free_mask, dtype=bool)

            # ===== 3) “近路径 & free” 的点：dist < tau 且 free =====
            near_mask = (dist_pred < self.tau) & free_mask
            near_pts = joints_np[near_mask]
            near_dist = dist_pred[near_mask]

            K = min(batch_size, len(joints_np))
            selected = []

            if len(near_pts) >= K:
                # 只在 near free 里按距离从小到大取 K 个
                order = np.argsort(near_dist)[:K]
                selected = near_pts[order]
            else:
                # 先把所有 near free 点放进去
                selected = list(near_pts)
                remain = K - len(selected)

                # ===== 4) 用“全体 free 点中的最小距离”补齐 =====
                # 先只在 free 里按距离排序
                free_idx = np.where(free_mask)[0]
                free_sorted = free_idx[np.argsort(dist_pred[free_idx])]

                for idx in free_sorted:
                    if len(selected) >= K:
                        break
                    if not near_mask[idx]:   # 避免与 near 重复
                        selected.append(joints_np[idx])

                # 仍然不够（极端情况：free 非常少），退化到全体里补
                if len(selected) < K:
                    order_all = np.argsort(dist_pred)
                    for idx in order_all:
                        if len(selected) >= K:
                            break
                        if joints_np[idx] not in selected:
                            selected.append(joints_np[idx])

                selected = np.array(selected, dtype=float)

            return [self.to_key(p) for p in selected]

        # ------------------------------------------------
        # 后面的主体逻辑不变
        # ------------------------------------------------
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
            return select_with_nn(candidates)

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

        return select_with_nn(candidates)

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
                new_samples = self.sample_from_env(c_best, self.batch_size, self.vertices)
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
            # ---- 更新自适应阈值 tau ----
            if (not self.has_solution) and np.isfinite(g_goal):
                # 第一次找到可行解
                self.has_solution = True
                self.tau = max(self.tau_min, self.tau * self.tau_shrink_fast)
            else:
                # 常规逐步收缩
                self.tau = max(self.tau_min, self.tau * self.tau_shrink)

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
            # iteration_times,         # ✅ 新增
            # first_solution_iter,
            # first_solution_cost,
            # first_solution_nodes,
            # final_solution_nodes,
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
