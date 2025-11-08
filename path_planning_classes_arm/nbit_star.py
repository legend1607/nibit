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

def get_point_cloud_mask_around_points(point_cloud, points, neighbor_radius=3):
    """
    - 自动支持 point_cloud 和 points 的任意维度
    - point_cloud: (n, C)
    - points: (m, C)
    - neighbor_radius: 半径阈值
    """
    point_cloud = np.asarray(point_cloud)
    points = np.asarray(points)

    # 检查维度一致
    assert point_cloud.shape[1] == points.shape[1], "point_cloud 和 points 维度不一致"

    # 计算欧氏距离
    diff = point_cloud[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, m, C)
    dist = np.linalg.norm(diff, axis=2)  # (n, m)

    # 判断是否在邻域半径内
    neighbor_mask = dist < neighbor_radius  # (n, m)
    around_points_mask = np.any(neighbor_mask, axis=1)  # (n,)

    return around_points_mask

class NBITStar:
    def __init__(self, 
                start,
                goal,
                environment,
                neural_wrapper,
                iter_max, 
                batch_size,
                pc_n_points, 
                plot_flag=False, timer=None):
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.env = environment
        self.neural_wrapper = neural_wrapper

        # ---------- 关键：统一将 start/goal 转为可哈希 key ----------
        # 使用 tuple 且对浮点做适度舍入以避免微小差异导致 key 不匹配
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
        self.pc_n_points=pc_n_points
        self.T = 0
        self.eta = 1.1  # tunable parameter
        self.obj_radius = 1
        self.resolution = 3

        # the parameters for informed sampling
        # NOTE: distance expects array-like inputs so to compute c_min use original numeric arrays
        self.c_min = self.distance(self.start, self.goal)
        self.center_point = None
        self.C = None

        # whether plot the middle planning process
        self.plot_planning_process = plot_flag

        self.n_collision_points = 0
        self.n_free_points = 2
        self.path=[]

    # ---------- helper: convert any point-like to hashable tuple ----------
    def to_key(self, point, ndigits=6):
        """
        将点转为 tuple(key)。若输入为 numpy array/list，先转换为 ndarray，再round到 ndigits。
        返回 tuple(float,...)
        """
        if isinstance(point, tuple):
            return tuple([float(round(x, ndigits)) for x in point])
        if isinstance(point, np.ndarray):
            arr = np.round(point.astype(float), ndigits)
            return tuple(arr.tolist())
        # try castable list-like
        try:
            arr = np.round(np.array(point, dtype=float), ndigits)
            return tuple(arr.tolist())
        except Exception:
            # fallback: return as-is cast to tuple
            return tuple(point)

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
                # disable informed sampling (will fallback to uniform)
                self.center_point = None
                self.C = None
                return
        except Exception:
            self.center_point = None
            self.C = None
            return

        # center_point 使用数值形式（numpy array）
        self.center_point = np.array([(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)])
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
        u = np.random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def sample_from_env(self, c_best, batch_size, vertices=None, bias_ratio=0.5):
        """
        生成固定 pc_n_points 个候选点作为点云输入网络，
        然后从网络预测的高概率点中按总分排序选择一部分，
        剩余通过椭圆/单位球采样补齐。
        """
        samples = []

        # --- 1️⃣ 生成固定候选点 ---
        candidate_points = []

        if (
            c_best is None
            or not np.isfinite(c_best)
            or self.c_min is None
            or self.c_min <= 1e-12
            or self.center_point is None
            or self.C is None
        ):
            for _ in range(self.pc_n_points):
                p = self.env.sample_empty_points()  # 返回 ndarray
                candidate_points.append(self.to_key(p))
        else:
            # 椭圆采样
            a = c_best / 2.0
            b = math.sqrt(max(0.0, c_best**2 - self.c_min**2)) / 2.0
            L = np.diag([a] + [b] * (self.dimension - 1))
            while len(candidate_points) < self.pc_n_points:
                x_ball = self.sample_unit_ball()
                x_ellipsoid = self.C @ (L @ x_ball) + self.center_point
                try:
                    if self.env._point_in_free_space(x_ellipsoid):
                        candidate_points.append(x_ellipsoid)
                except Exception:
                    continue

        candidate_points = np.array(candidate_points)

        # --- 2️⃣ 网络预测并选择 top n_bias 点 ---
        if self.neural_wrapper is not None:
            start_mask = get_point_cloud_mask_around_points(
                candidate_points,
                np.array([self.start]),
                neighbor_radius=0.1
            )
            goal_mask = get_point_cloud_mask_around_points(
                candidate_points,
                np.array([self.goal]),
                neighbor_radius=0.1
            )

            path_score, keypoint_score = self.neural_wrapper.classify_path_points(
                candidate_points.astype(np.float32),
                start_mask.astype(np.float32),
                goal_mask.astype(np.float32)
            )

            # 按总分排序
            combined_score = path_score + keypoint_score
            sorted_idx = np.argsort(-combined_score)  # 降序
            n_bias = int(batch_size * bias_ratio)
            n_bias = min(n_bias, len(candidate_points))
            if n_bias > 0:
                top_idx = sorted_idx[:n_bias]
                selected_predicted = candidate_points[top_idx]
                samples.extend([self.to_key(p) for p in selected_predicted])
                self.predicted_points = [self.to_key(p) for p in selected_predicted]

        # --- 3️⃣ 剩余通过采样补齐 ---
        n_remaining = batch_size - len(samples)
        while n_remaining > 0:
            if (
                c_best is not None
                and np.isfinite(c_best)
                and self.C is not None
                and self.center_point is not None
            ):
                # 椭圆采样
                a = c_best / 2.0
                b = math.sqrt(max(0.0, c_best**2 - self.c_min**2)) / 2.0
                L = np.diag([a] + [b] * (self.dimension - 1))
                x_ball = self.sample_unit_ball()
                x_sample = self.C @ (L @ x_ball) + self.center_point
            else:
                # 全局均匀采样
                x_sample = self.env.sample_empty_points()

            try:
                if self.env._point_in_free_space(x_sample):
                    samples.append(self.to_key(x_sample))
                    n_remaining -= 1
            except Exception:
                continue

        return samples

    def is_point_free(self, point):
        # point: tuple or array-like -> pass numeric array to env
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
        # point expected to be tuple key
        if point == self.start:
            return 0
        return self.g_scores.get(point, INF)

    def get_f_score(self, point):
        g = self.get_g_score(point)
        if g == INF:
            # 点还没被访问，用 start->point 的启发式代替 g_score
            g = self.heuristic_cost(self.start, point)
        return g + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        if not self.is_edge_free([point1, point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        return self.distance(point1, point2)
    
    def distance(self, point1, point2):
        # accepts tuple or array-like
        return np.linalg.norm(np.array(point1, dtype=float) - np.array(point2, dtype=float))

    def get_edge_value(self, edge):
        # edge: (p1_tuple, p2_tuple)
        return self.get_g_score(edge[0]) + self.heuristic_cost(edge[0], edge[1]) + self.heuristic_cost(edge[1], self.goal)

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

    def prune_edge(self, c_best):
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point, None)

    def prune(self, c_best):
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            f = self.get_f_score(point)
            if f <= c_best:
                # 如果 g_score 还没被计算过（INF），把点加入 samples
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
            estimated_f_score = self.get_g_score(point) + \
                                self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
            if estimated_f_score < self.get_g_score(self.goal):
                heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        # neighbors among existing vertices
        if point not in self.old_vertices:
            neigbors_vertex = [ver for ver in self.vertices if self.distance(point, ver) <= self.r]
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = self.get_g_score(point) + \
                                        self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
                    if estimated_f_score < self.get_g_score(self.goal):
                        estimated_g_score = self.get_g_score(point) + self.heuristic_cost(point, neighbor)
                        if estimated_g_score < self.get_g_score(neighbor):
                            heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        self.timer.finish(Timer.EXPAND)

    def get_best_path(self):
        path = []
        if self.g_scores[self.goal] != INF:
            path.append(self.goal)
            point = self.goal
            while point != self.start:
                point = self.edges[point]
                path.append(point)
            path.reverse()
        return path

    def path_length_calculate(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += self.distance(path[i], path[i + 1])
        return path_length

    def planning(self, visualize=False, refresh_interval=1):
        collision_checks = self.env.collision_check_count
        self.setup_planning()

        init_time = time()
        iteration_costs = []  # 记录每次迭代的代价

        for k in range(self.iter_max):
            final_iter=k
            # 1. 如果队列为空 -> 新采样
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.get_g_score(self.goal)
                self.prune(c_best)
                new_samples = self.sample_from_env(c_best, self.batch_size, self.vertices)
                self.samples.extend(new_samples)
                self.T += self.batch_size

                self.timer.start()
                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point), point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)
                q = len(self.vertices) + len(self.samples)
                if q > 0:
                    self.r = self.radius_init() * ((math.log(q) / q) ** (1.0 / self.dimension))
                self.timer.finish(Timer.HEAP)

            # 2. 扩展节点
            try:
                while self.bestVertexQueueValue() <= self.bestEdgeQueueValue():
                    self.timer.start()
                    _, point = heapq.heappop(self.vertex_queue)
                    self.timer.finish(Timer.HEAP)
                    self.expand_vertex(point)
            except Exception as e:
                if (not self.edge_queue) and (not self.vertex_queue):
                    continue
                else:
                    raise e

            # 3. 选取最优边并扩展树
            if not self.edge_queue:
                continue

            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)
            if best_edge_value < self.get_g_score(self.goal):
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                self.timer.start()
                actual_f_edge = (
                    self.get_g_score(bestEdge[0]) +
                    actual_cost_of_edge +
                    self.heuristic_cost(bestEdge[1], self.goal)
                )
                if actual_f_edge < self.get_g_score(self.goal):
                    actual_g_score_of_point = self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        # update g_score and parent (edges) using tuple keys
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]
                        if bestEdge[1] not in self.vertices:
                            try:
                                self.samples.remove(bestEdge[1])
                            except ValueError:
                                pass
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(self.vertex_queue, (self.get_point_value(bestEdge[1]), bestEdge[1]))

                        # prune inconsistent edges pointing to bestEdge[1]
                        self.edge_queue = [
                            item for item in self.edge_queue
                            if item[1][1] != bestEdge[1] or
                            self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0], item[1][1]) <
                            self.get_g_score(item[1][0])
                        ]
                        heapq.heapify(self.edge_queue)
                self.timer.finish(Timer.HEAP)
            else:
                self.vertex_queue = []
                self.edge_queue = []
            
            self.path = self.get_best_path()
            iteration_costs.append(self.get_g_score(self.goal))
            # 判定 1：找到直接连接的最优路径
            if len(self.path) == 2 and self.path[0] == self.start and self.path[1] == self.goal:
                print(f"可以直接连通起点和终点，退出迭代")
                break

            # 判定 2：环境允许直接连线，并且路径长度接近直线距离
            direct_free = False
            try:
                direct_free = self.is_edge_free([self.start, self.goal])
            except Exception:
                # 如果 env._edge_fp 不支持或报错，则忽略此条件
                direct_free = False

            if direct_free:
                current_len = self.path_length_calculate(self.path) if len(self.path) > 1 else INF
                straight_len = self.distance(self.start, self.goal)
                if abs(current_len - straight_len) <= 1e-5:
                    print("起点终点差", current_len-straight_len)
                    break
                
        return (self.path,
            self.samples,
            self.edges,
            self.env.collision_check_count - collision_checks,
            self.get_g_score(self.goal),
            self.T,
            time() - init_time,
            final_iter,
            iteration_costs
        )

def get_bit_planner(
    args,
    problem,
    neural_wrapper=None,
):
    planner = NBITStar(
        problem["start"],
        problem["goal"],
        problem['env'],
        neural_wrapper,
        args.iter_max,
        args.batch_size,
        args.pc_n_points,
    )
    return planner
