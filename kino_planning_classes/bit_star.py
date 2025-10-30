import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
import random
from collections import deque

INF = float("inf")

# -----------------------------
# Helper functions for double-integrator trajectory
# -----------------------------
def find_tau(x0, x1, max_acc=1.0):
    """简单估算双积分最优时间"""
    dx = np.array(x1[:2]) - np.array(x0[:2])
    v0 = np.array(x0[2:])
    v1 = np.array(x1[2:])
    tau = np.linalg.norm(dx) / max_acc
    return max(tau, 0.1)

def generate_poly_traj(x0, x1, tau):
    """
    生成二次多项式轨迹 x(t) = a0 + a1 t + a2 t^2
    x0, x1: [x, y, vx, vy]
    tau: 时间段
    返回 coeffs_x, coeffs_y
    """
    
    tau = max(tau, 0.1)  # 防止 tau 太小
    coeffs_x = np.zeros(3)
    coeffs_y = np.zeros(3)
    # x方向
    coeffs_x[0] = x0[0]
    coeffs_x[1] = x0[2]
    coeffs_x[2] = (2*(x1[0]-x0[0]-x0[2]*tau)) / (tau**2)
    # y方向
    coeffs_y[0] = x0[1]
    coeffs_y[1] = x0[3]
    coeffs_y[2] = (2*(x1[1]-x0[1]-x0[3]*tau)) / (tau**2)
    return coeffs_x, coeffs_y

def eval_poly(coeffs, t):
    """评估二次多项式"""
    return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2

# -----------------------------
# KinoBITStar Planner
# -----------------------------
class KinoBITStar:
    def __init__(self, start, goal, environment, iter_max=200, batch_size=50, plot_flag=False):
        self.env = environment

        if len(start) == 2:
            start = (start[0], start[1], 0.0, 0.0)
        self.start = tuple(start)

        if len(goal) == 2:
            goal = (goal[0], goal[1], 0.0, 0.0)
        self.goal = tuple(goal)
        self.iter_max = iter_max
        self.batch_size = batch_size
        self.plot_flag = plot_flag

        # Tree data structures
        self.vertices = [self.start]
        self.edges = dict()
        self.g_scores = {self.start: 0.0, self.goal: INF}
        self.samples = []
        self.vertex_queue = list(self.vertices)

        self.dimension = 4  # x, y, vx, vy
        self.r = INF
        self.path = []
        self.trajs = []  # 每条边的多项式轨迹

    # -----------------------------
    # 核心方法
    # -----------------------------
    def distance(self, a, b):
        return np.linalg.norm(np.array(a[:2])-np.array(b[:2]))

    def sample_point(self):
        return tuple(self.env.sample_empty_points())

    def is_edge_free(self, x0, x1):
        """离散化检测双积分轨迹是否碰撞"""
        dt = 0.05
        t_total = find_tau(x0, x1)
        steps = max(int(t_total/dt), 1)
        for i in range(steps+1):
            t = i*dt
            x = eval_poly(generate_poly_traj(x0, x1, t)[0], t)
            y = eval_poly(generate_poly_traj(x0, x1, t)[1], t)
            if not self.env._point_in_free_space([x, y]):
                return False
        return True

    def expand_vertex(self, vertex):
        vertex = tuple(vertex)
        neighbors = []
        # 在 samples 中找到半径 r 内的点
        for s in self.samples:
            if self.distance(vertex, s) <= self.r:
                neighbors.append(s)
        for neighbor in neighbors:
            if self.is_edge_free(vertex, neighbor):
                cost = self.g_scores[vertex] + self.distance(vertex, neighbor)
                if cost < self.g_scores.get(neighbor, INF):
                    self.g_scores[neighbor] = cost
                    self.edges[neighbor] = vertex
                    if neighbor not in self.vertices:
                        self.vertices.append(neighbor)
                        self.vertex_queue.append(neighbor)

    def get_best_path(self):
        path = []
        if self.g_scores.get(self.goal, INF) == INF:
            return path
        node = self.goal
        while node != self.start:
            path.append(node)
            node = self.edges[node]
        path.append(self.start)
        path.reverse()
        return path

    # -----------------------------
    # 主规划循环
    # -----------------------------
    def planning(self):
        for k in range(self.iter_max):
            # 队列为空 -> 新采样
            if not self.vertex_queue:
                new_samples = [self.sample_point() for _ in range(self.batch_size)]
                self.samples.extend(new_samples)
                self.vertex_queue = list(self.vertices)
                self.r = 3.0  # 可以使用 BIT* 半径公式

            # 扩展节点
            vertex = self.vertex_queue.pop(0)
            self.expand_vertex(vertex)

            # 更新最优路径
            path = self.get_best_path()
            if path:
                self.path = path

        # -----------------------------
        # 生成平滑多段多项式轨迹
        # -----------------------------
        self.trajs = []
        if hasattr(self, 'path') and len(self.path) > 1:
            for i in range(len(self.path)-1):
                x0 = self.path[i]
                x1 = self.path[i+1]
                tau = find_tau(x0, x1)
                coeffs_x, coeffs_y = generate_poly_traj(x0, x1, tau)
                self.trajs.append({'coeffs_x': coeffs_x, 'coeffs_y': coeffs_y, 'tau': tau})

        return self.path, self.trajs

    # -----------------------------
    # 可视化
    # -----------------------------
    def plot(self):
        if not hasattr(self, 'path') or not self.path:
            print("No path found")
            return
        plt.figure()
        # 绘制障碍
        for rx, ry, rw, rh in self.env.rect_obstacles:
            plt.gca().add_patch(plt.Rectangle((rx, ry), rw, rh, color='black', alpha=0.5))
        for cx, cy, r in self.env.circle_obstacles:
            plt.gca().add_patch(plt.Circle((cx, cy), r, color='black', alpha=0.5))

        # 绘制轨迹
        for traj in self.trajs:
            tau = traj['tau']
            t_samples = np.linspace(0, tau, 50)
            xs = [eval_poly(traj['coeffs_x'], t) for t in t_samples]
            ys = [eval_poly(traj['coeffs_y'], t) for t in t_samples]
            plt.plot(xs, ys, 'g-', lw=2)

        # 绘制路径顶点
        path_arr = np.array(self.path)
        plt.plot(path_arr[:,0], path_arr[:,1], 'ro', markersize=5)
        plt.plot(self.start[0], self.start[1], 'go', markersize=8)
        plt.plot(self.goal[0], self.goal[1], 'bo', markersize=8)

        plt.axis('equal')
        plt.show()

# -----------------------------
# 工厂函数保留
# -----------------------------
def get_bit_planner(args, problem, neural_wrapper=None):
    planner = KinoBITStar(
        problem["x_start"],
        problem["x_goal"],
        problem['env'],
        iter_max=args.iter_max,
        batch_size=args.pc_n_points,
        plot_flag=True
    )
    return planner
