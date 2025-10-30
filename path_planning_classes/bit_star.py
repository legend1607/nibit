import numpy as np
import os
import sys
# 将项目根目录加入 Python 搜索路径
# __file__ 是当前脚本路径，os.path.dirname(__file__) 是 algorithm/ 目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import math
import yaml
import heapq
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch
from shapely import affinity
import itertools
from time import time
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
                pc_n_points, 
                plot_flag=False,sampling=None, timer=None):
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.env = environment

        self.start = start
        self.goal = goal

        self.bounds = self.env.bound
        self.bounds = np.array(self.bounds).reshape((2, -1)).T
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.dimension = environment.config_dim

        # This is the tree
        self.vertices = []
        self.edges = dict()  # key = point，value = parent
        self.g_scores = dict()

        self.samples = []
        self.vertex_queue = []
        self.edge_queue = []
        self.old_vertices = set()

        self.r = INF
        self.iter_max=iter_max
        self.batch_size = pc_n_points
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

        if sampling is None:
            self.sampling = self.sample_from_env
        else:
            self.sampling = sampling

        self.n_collision_points = 0
        self.n_free_points = 2

    def setup_planning(self):
        # add goal to the samples
        if self.goal not in self.samples:
            self.samples.append(self.goal)
        self.g_scores[self.goal] = INF

        # add start to the tree
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
        volume = np.abs(np.prod(self.ranges)) * self.n_free_points / (self.n_collision_points + self.n_free_points)
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

        self.center_point = np.array([(self.start[i] + self.goal[i]) / 2.0 for i in range(self.dimension)])
        # 长轴方向
        a1 = np.array(self.goal, dtype=float) - np.array(self.start, dtype=float)
        norm_a1 = np.linalg.norm(a1)
        if norm_a1 < 1e-12:
            self.C = None
            return
        a1 /= norm_a1  # 单位向量


        # 构造旋转矩阵 C
        # 任意维，第一列为主轴方向，其余列通过随机矩阵QR分解生成正交基
        X = np.random.randn(self.dimension, self.dimension)
        X[:, 0] = a1
        Q, _ = np.linalg.qr(X)
        # 保证第一列与 a1 同向
        if np.dot(Q[:, 0], a1) < 0:
            Q[:, 0] *= -1
        self.C = Q

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.dimension)  # an array of d normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        r = np.random.random() ** (1.0 / self.dimension)
        x = r * u / norm
        return x

    def sample_from_env(self, c_best, sample_num, vertices=None):
        """
        从椭圆域中进行 informed 采样。
        若 c_best 无效，则回退为在环境中均匀采样。
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
            for _ in range(sample_num):
                p = self.env.sample_empty_points()
                samples.append(tuple(p))
            return samples

        # --- 椭圆参数 ---
        a = c_best / 2.0
        b = math.sqrt(max(0.0, c_best**2 - self.c_min**2)) / 2.0
        L = np.diag([a] + [b] * (self.dimension - 1))

        # --- 椭圆采样 ---
        while len(samples) < sample_num:
            x_ball = self.sample_unit_ball()  # 单位球采样
            # 从椭圆采样空间变换回真实坐标
            x_ellipsoid = self.C @ (L @ x_ball) + self.center_point

            # 检查碰撞（二维时取前两个坐标）
            if self.env._point_in_free_space(x_ellipsoid[:2]):
                samples.append(tuple(x_ellipsoid))

        return samples

    def get_random_point(self):
        return tuple(self.env.sample_empty_points())

    def is_point_free(self, point):
        result = self.env._state_fp(np.array(point))
        if result:
            self.n_free_points += 1
        else:
            self.n_collision_points += 1
        return result

    def is_edge_free(self, edge):
        result = self.env._edge_fp(np.array(edge[0]), np.array(edge[1]))
        # self.T += self.env.k
        return result

    def get_g_score(self, point):
        # gT(x)
        if point == self.start:
            return 0
        if point not in self.edges:
            return INF
        else:
            return self.g_scores.get(point)

    def get_f_score(self, point):
        # f^(x)
        return self.heuristic_cost(self.start, point) + self.heuristic_cost(point, self.goal)

    def actual_edge_cost(self, point1, point2):
        # c(x1,x2)
        if not self.is_edge_free([point1, point2]):
            return INF
        return self.distance(point1, point2)

    def heuristic_cost(self, point1, point2):
        # Euler distance as the heuristic distance
        return self.distance(point1, point2)

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_edge_value(self, edge):
        # sort value for edge
        return self.get_g_score(edge[0]) + self.heuristic_cost(edge[0], edge[1]) + self.heuristic_cost(edge[1],
                                                                                                       self.goal)

    def get_point_value(self, point):
        # sort value for point
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
                self.edges.pop(point)

    def prune(self, c_best):
        self.samples = [point for point in self.samples if self.get_f_score(point) < c_best]
        self.prune_edge(c_best)
        vertices_temp = []
        for point in self.vertices:
            if self.get_f_score(point) <= c_best:
                if self.get_g_score(point) == INF:
                    self.samples.append(point)
                else:
                    vertices_temp.append(point)
        self.vertices = vertices_temp

    def expand_vertex(self, point):
        self.timer.start()

        # get the nearest value in vertex for every one in samples where difference is less than the radius
        neigbors_sample = []
        for sample in self.samples:
            if self.distance(point, sample) <= self.r:
                neigbors_sample.append(sample)

        self.timer.finish(Timer.NN)

        self.timer.start()

        # add an edge to the edge queue is the path might improve the solution
        for neighbor in neigbors_sample:
            estimated_f_score = self.heuristic_cost(self.start, point) + \
                                self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
            if estimated_f_score < self.g_scores[self.goal]:
                heapq.heappush(self.edge_queue, (self.get_edge_value((point, neighbor)), (point, neighbor)))

        # add the vertex to the edge queue
        if point not in self.old_vertices:
            neigbors_vertex = []
            for ver in self.vertices:
                if self.distance(point, ver) <= self.r:
                    neigbors_vertex.append(ver)
            for neighbor in neigbors_vertex:
                if neighbor not in self.edges or point != self.edges.get(neighbor):
                    estimated_f_score = self.heuristic_cost(self.start, point) + \
                                        self.heuristic_cost(point, neighbor) + self.heuristic_cost(neighbor, self.goal)
                    if estimated_f_score < self.g_scores[self.goal]:
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

        if visualize:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
            ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
            ax.set_title("BIT* Path Planning")
            ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
            ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')
            for rx, ry, rw, rh in getattr(self.env, "rect_obstacles", []):
                ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='black', alpha=0.5))
            for cx, cy, r in getattr(self.env, "circle_obstacles", []):
                ax.add_patch(plt.Circle((cx, cy), r, color='black', alpha=0.5))
            ax.legend()
        init_time = time()

        for k in range(self.iter_max):
            # 1. 如果队列为空 -> 新采样
            if not self.vertex_queue and not self.edge_queue:
                c_best = self.g_scores[self.goal]
                self.prune(c_best)
                if math.isinf(c_best):
                    new_samples = [self.get_random_point() for _ in range(self.batch_size)]
                else:
                    new_samples = self.sampling(c_best, self.batch_size, self.vertices)
                self.samples.extend(new_samples)
                self.T += self.batch_size

                self.timer.start()
                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(point), point) for point in self.vertices]
                heapq.heapify(self.vertex_queue)
                q = len(self.vertices) + len(self.samples)
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
            best_edge_value, bestEdge = heapq.heappop(self.edge_queue)
            if best_edge_value < self.g_scores[self.goal]:
                actual_cost_of_edge = self.actual_edge_cost(bestEdge[0], bestEdge[1])
                self.timer.start()
                actual_f_edge = (
                    self.heuristic_cost(self.start, bestEdge[0]) +
                    actual_cost_of_edge +
                    self.heuristic_cost(bestEdge[1], self.goal)
                )
                if actual_f_edge < self.g_scores[self.goal]:
                    actual_g_score_of_point = self.get_g_score(bestEdge[0]) + actual_cost_of_edge
                    if actual_g_score_of_point < self.get_g_score(bestEdge[1]):
                        self.g_scores[bestEdge[1]] = actual_g_score_of_point
                        self.edges[bestEdge[1]] = bestEdge[0]
                        if bestEdge[1] not in self.vertices:
                            self.samples.remove(bestEdge[1])
                            self.vertices.append(bestEdge[1])
                            heapq.heappush(self.vertex_queue, (self.get_point_value(bestEdge[1]), bestEdge[1]))

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

            # 4. 实时可视化
            if visualize and k % refresh_interval == 0:
                ax.clear()
                ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"BIT* Iteration {k}/{self.iter_max}")

                # 绘制障碍
                for rx, ry, rw, rh in self.env.rect_obstacles:
                    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='gray', alpha=1.0))
                for cx, cy, r in self.env.circle_obstacles:
                    ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=1.0))
                # 绘制采样点
                if self.samples:
                    samples_arr = np.array(self.samples)
                    ax.scatter(samples_arr[:, 0], samples_arr[:, 1], c='lightgray', s=5, label='Samples')

                # 绘制树的边
                for child, parent in self.edges.items():
                    ax.plot([parent[0], child[0]], [parent[1], child[1]], c='skyblue', lw=0.5)

                # 绘制最优路径
                if len(self.path) > 1:
                    path_arr = np.array(self.path)
                    ax.plot(path_arr[:, 0], path_arr[:, 1], 'r-', lw=2, label='Best Path')

                # 绘制起点与目标点
                ax.plot(self.start[0], self.start[1], 'go', markersize=8)
                ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8)
                if self.C is not None and self.center_point is not None:
                    # 当前最优路径的 c_best
                    c_best = self.g_scores[self.goal]
                    if np.isfinite(c_best) and c_best > self.c_min:
                        val = max(0.0, c_best**2 - self.c_min**2)
                        a = c_best / 2.0                  # 长半轴
                        b = math.sqrt(val) / 2.0          # 短半轴

                        # 构造 L 矩阵（和采样一致）
                        L = np.diag([a, b])

                        # 单位圆边界
                        theta = np.linspace(0, 2*np.pi, 100)
                        unit_circle = np.vstack([np.cos(theta), np.sin(theta)])  # shape: 2 x 100

                        # 变换到椭圆
                        ellipse_points = (self.C @ L @ unit_circle) + self.center_point[:, None]  # shape: 2 x 100

                        # 绘制轮廓
                        ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'purple', linestyle='--', linewidth=1.5)
    
                ax.legend()

        return (self.path,
            self.samples,
                self.edges,
                self.env.collision_check_count - collision_checks,
                self.g_scores[self.goal],
                self.T,
                time() - init_time)

def get_bit_planner(
    args,
    problem,
    neural_wrapper=None,
):
    """
    BIT* 路径规划器工厂函数。
    自动在内部构造 RandomEnv，并返回初始化好的 BITStar 实例。
    """

    planner = BITStar(
        problem["start"],
        problem["goal"],
        problem['env'],
        args.iter_max,
        args.pc_n_points,
    )

    return planner