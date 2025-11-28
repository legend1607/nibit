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

class BITStar:
    def __init__(self, 
                start,
                goal,
                environment, 
                iter_max, 
                batch_size,
                pc_n_points, 
                plot_flag=False, timer=None):
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.env = environment

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
        self.pc_n_points = pc_n_points
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
        u = np.random.normal(0, 1, self.dimension)
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def sample_from_env(self, c_best, batch_size, vertices=None):
        """
        从椭圆域中进行 informed 采样。
        如果采样失败次数过多，抛出异常让上层处理。
        """
        samples = []

        # --- 回退到 uniform 采样 ---
        if (
            c_best is None
            or not np.isfinite(c_best)
            or self.c_min is None
            or self.c_min <= 1e-12
            or self.center_point is None
            or self.C is None
        ):
            consecutive_failures = 0
            max_consecutive_failures = 50  # 连续失败50次就放弃

            for _ in range(batch_size * 10):
                try:
                    p = self.env.sample_empty_points()
                    if p is not None:
                        samples.append(self.to_key(p))
                        consecutive_failures = 0  # 重置失败计数
                    else:
                        consecutive_failures += 1
                except Exception:
                    consecutive_failures += 1

                # ✅ 如果连续失败太多次，说明环境有问题
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Failed to sample {batch_size} points. "
                        f"Environment may be too crowded. "
                        f"Only sampled {len(samples)} points."
                    )

                if len(samples) >= batch_size:
                    break

            return samples

        # --- 椭圆采样 ---
        a = c_best / 2.0
        b = math.sqrt(max(0.0, c_best**2 - self.c_min**2)) / 2.0
        L = np.diag([a] + [b] * (self.dimension - 1))

        consecutive_failures = 0
        max_consecutive_failures = 50

        for _ in range(batch_size * 10):
            x_ball = self.sample_unit_ball()
            x_ellipsoid = self.C @ (L @ x_ball) + self.center_point

            try:
                if self.env._point_in_free_space(x_ellipsoid):
                    samples.append(self.to_key(x_ellipsoid))
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception:
                # 回退尝试
                try:
                    p = self.env.sample_empty_points()
                    if p is not None:
                        samples.append(self.to_key(p))
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception:
                    consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Failed to sample from ellipsoid. "
                    f"Environment may be too crowded. "
                    f"Only sampled {len(samples)} points."
                )

            if len(samples) >= batch_size:
                break

        return samples


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
        # 获取当前路径上的点（需要保护）
        path_points = self.get_current_path_points()
        
        edge_array = list(self.edges.items())
        for point, parent in edge_array:
            point = self.to_key(point)
            parent = self.to_key(parent)
            
            # 关键：不要删除当前最优路径上的边
            if point in path_points:
                continue
            
            # 删除超出椭圆域的边
            if self.get_f_score(point) > c_best or self.get_f_score(parent) > c_best:
                self.edges.pop(point, None)

    def prune(self, c_best):
        """修剪顶点和边，但保护当前最优路径"""
        # 先保护当前路径
        path_points = self.get_current_path_points()
        
        # 修剪样本点
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        
        # 修剪边（保护当前路径）
        self.prune_edge(c_best)
        
        # 修剪顶点
        vertices_temp = []
        for point in self.vertices:
            point = self.to_key(point)
            f = self.get_f_score(point)
            
            # 保护当前路径上的点
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
        """获取最优路径，带有完整的错误检查"""
        path = []
        if self.get_g_score(self.goal) == INF:
            return path
        
        path.append(self.goal)
        point = self.goal
        visited = set()  # 防止无限循环
        max_iterations = len(self.vertices) + 100
        iteration = 0
        
        while point != self.start and iteration < max_iterations:
            # 添加循环检测
            if point in visited:
                print(f"Warning: Cycle detected at {point}")
                return []
            visited.add(point)
            
            # 检查父节点是否存在
            if point not in self.edges:
                print(f"Warning: Point {point} has no parent in edges")
                print(f"Current point g_score: {self.g_scores.get(point, 'N/A')}")
                print(f"Is point in vertices: {point in self.vertices}")
                print(f"Total edges: {len(self.edges)}, Total vertices: {len(self.vertices)}")
                return []  # 返回空路径而不是崩溃
            
            parent = self.edges[point]
            parent = self.to_key(parent)  # 确保 key 一致性
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
        no_improve_limit = 50
        no_improve_count = 0
        best_cost = self.get_g_score(self.goal)

        collision_checks = self.env.collision_check_count
        self.setup_planning()

        init_time = time()
        iteration_costs = []

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
                self.vertex_queue = [(self.get_point_value(v), self.to_key(v)) for v in self.vertices]
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
                    point = self.to_key(point)
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
            bestEdge = (self.to_key(bestEdge[0]), self.to_key(bestEdge[1]))

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
                        # 确保所有 key 都经过 to_key 处理
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
                            heapq.heappush(self.vertex_queue, (self.get_point_value(point_key), point_key))

                        # prune inconsistent edges
                        self.edge_queue = [
                            item for item in self.edge_queue
                            if self.to_key(item[1][1]) != point_key or
                            self.get_g_score(self.to_key(item[1][0])) + self.heuristic_cost(
                                self.to_key(item[1][0]), self.to_key(item[1][1])
                            ) < self.get_g_score(self.to_key(item[1][1]))
                        ]
                        heapq.heapify(self.edge_queue)
                self.timer.finish(Timer.HEAP)
            else:
                self.vertex_queue = []
                self.edge_queue = []

            # 4. Update path
            self.path = self.get_best_path()
            iteration_costs.append(self.get_g_score(self.goal))

            current_cost = self.get_g_score(self.goal)

            if current_cost < best_cost:
                best_cost = current_cost
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= no_improve_limit:
                break

            # 判定 1：找到直接连接的最优路径
            if len(self.path) == 2 and self.path[0] == self.start and self.path[1] == self.goal:
                break

            # 判定 2：环境允许直接连线，并且路径长度接近直线距离
            direct_free = False
            try:
                direct_free = self.is_edge_free([self.start, self.goal])
            except Exception:
                direct_free = False

            if direct_free:
                current_len = self.path_length_calculate(self.path) if len(self.path) > 1 else INF
                straight_len = self.distance(self.start, self.goal)
                if abs(current_len - straight_len) <= 1e-8:
                    break

        return (
            self.path,
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
    planner = BITStar(
        problem["start"],
        problem["goal"],
        problem['env'],
        args.iter_max,
        args.batch_size,
        args.pc_n_points,
    )
    return planner
