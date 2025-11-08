"""
kinobitstar.py
--------------------------------
KinoBIT*: Time-Optimal BIT* with Kinodynamic Constraints
Author: ChatGPT (GPT-5)
Date: 2025-11-06
--------------------------------
依赖:
    numpy
    matplotlib (用于可视化)
"""

import numpy as np
import heapq
import math
from time import time
import matplotlib.pyplot as plt

INF = float("inf")


# ============================================================
# 环境类
# ============================================================
class SimpleEnv:
    def __init__(self, bounds=((0, 10), (0, 10)), obstacles=None):
        self.bound = bounds
        self.obstacles = obstacles or []

    def _point_in_free_space(self, p):
        """检测点是否在空闲区域"""
        if np.any(p < [self.bound[0][0], self.bound[1][0]]) or np.any(
            p > [self.bound[0][1], self.bound[1][1]]
        ):
            return False
        for (cx, cy, r) in self.obstacles:
            if np.linalg.norm(p - np.array([cx, cy])) <= r:
                return False
        return True

    def sample_empty_points(self):
        """随机采样自由空间内的位置"""
        while True:
            p = np.random.uniform(
                [b[0] for b in self.bound], [b[1] for b in self.bound]
            )
            if self._point_in_free_space(p):
                return p

    def plot(self):
        """绘制环境"""
        for (cx, cy, r) in self.obstacles:
            circle = plt.Circle((cx, cy), r, color="gray", alpha=0.6)
            plt.gca().add_patch(circle)
        plt.xlim(self.bound[0])
        plt.ylim(self.bound[1])
        plt.gca().set_aspect("equal")


# ============================================================
# KinoBIT* 主类
# ============================================================
class KinoBITStar:
    def __init__(self, start, goal, environment, iter_max=300, batch_size=200):
        self.env = environment
        self.start = tuple(start)  # (x, y, vx, vy)
        self.goal = tuple(goal)
        self.dimension = len(start)

        # 参数
        self.iter_max = iter_max
        self.batch_size = batch_size
        self.r = 3.0  # 邻域半径

        # 状态容器
        self.vertices = [self.start]
        self.edges = {}  # child -> (parent, coeff, T)
        self.g_scores = {self.start: 0.0}
        self.samples = []
        self.path = []

    # --------------------------------------------------------
    # 动力学轨迹生成
    # --------------------------------------------------------
    def calcOptimalTrajWithPartialState(self, s1, s2, v_max=3.0, a_max=3.0):
        p0 = np.array(s1[:2], dtype=float)
        v0 = np.array(s1[2:], dtype=float)
        pf = np.array(s2[:2], dtype=float)
        vf = np.array(s2[2:], dtype=float)

        dp = pf - p0
        dist = np.linalg.norm(dp)
        if dist < 1e-6:
            return False, 0.0, None

        Tmin = max(0.5, dist / max(v_max, 1e-3) * 0.5)
        Tmax = max(6.0, dist / 0.2 * 4.0)
        coarse_N = 8
        fine_N = 20

        def solve_coeff_for_T(T):
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6*T, 12*T**2, 20*T**3],
            ], dtype=float)

            # 🔧 关键修正：b 维度改为 (6,2)
            b = np.array([
                [p0[0], p0[1]],
                [pf[0], pf[1]],
                [v0[0], v0[1]],
                [vf[0], vf[1]],
                [0.0, 0.0],
                [0.0, 0.0]
            ])
            try:
                coeff = np.linalg.solve(A, b)
                return coeff
            except np.linalg.LinAlgError:
                return None

        def max_vel_acc_along(coeff, T, ncheck=50):
            ts = np.linspace(0, T, ncheck)
            maxv = 0.0
            maxa = 0.0
            for t in ts:
                vbias = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
                abias = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
                vel = coeff.T @ vbias
                acc = coeff.T @ abias
                maxv = max(maxv, np.linalg.norm(vel))
                maxa = max(maxa, np.linalg.norm(acc))
            return maxv, maxa

        # 搜索逻辑保持不变 ...
        Ts = np.linspace(Tmin, Tmax, coarse_N)
        for T in Ts:
            coeff = solve_coeff_for_T(T)
            if coeff is None:
                continue
            maxv, maxa = max_vel_acc_along(coeff, T, ncheck=30)
            if maxv <= v_max and maxa <= a_max:
                if self.checkCollision(coeff, T):
                    return True, T, coeff
        return False, None, None

    def updateStateFromCoeff(self, coeff, T):
        """由多项式系数计算状态(pos, vel, acc)"""
        bias = np.array([1, T, T**2, T**3, T**4, T**5])
        pos = coeff.T @ bias
        bias = np.array([0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4])
        vel = coeff.T @ bias
        bias = np.array([0, 0, 2, 6*T, 12*T**2, 20*T**3])
        acc = coeff.T @ bias
        return pos, vel, acc

    def checkCollision(self, coeff, T):
        """沿轨迹采样检测碰撞"""
        ts = np.linspace(0, T, 20)
        for t in ts:
            bias = np.array([1, t, t**2, t**3, t**4, t**5])
            pos = coeff.T @ bias
            if not self.env._point_in_free_space(pos):
                return False
        return True

    # --------------------------------------------------------
    # 辅助函数
    # --------------------------------------------------------
    def heuristic(self, s1, s2):
        return np.linalg.norm(np.array(s1[:2]) - np.array(s2[:2]))

    def get_g(self, s):
        return self.g_scores.get(s, INF)

    # --------------------------------------------------------
    # 核心 BIT* 扩展
    # --------------------------------------------------------
    def expand_vertex(self, vertex):
        """拓展当前顶点（动力学可行轨迹）"""
        for s in self.samples:
            if self.heuristic(vertex, s) > self.r:
                continue
            success, T, coeff = self.calcOptimalTrajWithPartialState(vertex, s)
            if not success:
                continue
            if not self.checkCollision(coeff, T):
                continue
            g_new = self.get_g(vertex) + T
            if g_new < self.get_g(s):
                self.g_scores[s] = g_new
                self.edges[s] = (vertex, coeff, T)

    # --------------------------------------------------------
    # 回溯路径
    # --------------------------------------------------------
    def get_best_path(self):
        if self.goal not in self.g_scores:
            return []
        path = [self.goal]
        s = self.goal
        while s != self.start:
            s = self.edges[s][0]
            path.append(s)
        path.reverse()
        return path

    # --------------------------------------------------------
    # 主规划函数
    # --------------------------------------------------------
    def planning(self):
        # 初始化采样
        for _ in range(self.batch_size):
            pos = self.env.sample_empty_points()
            vel = np.random.uniform(-1, 1, 2)
            self.samples.append(tuple(np.hstack((pos, vel))))

        t0 = time()
        for k in range(self.iter_max):
            v = self.vertices[np.random.randint(len(self.vertices))]
            self.expand_vertex(v)
            # 更新新节点
            new_nodes = [s for s in self.g_scores if s not in self.vertices]
            self.vertices.extend(new_nodes)
            if self.goal in self.g_scores:
                self.path = self.get_best_path()
                break
        return self.path, self.g_scores.get(self.goal, INF), time() - t0

    # --------------------------------------------------------
    # 可视化
    # --------------------------------------------------------
    def visualize(self):
        self.env.plot()
        for c, (p, coeff, T) in self.edges.items():
            ts = np.linspace(0, T, 30)
            traj = np.array([coeff.T @ np.array([1, t, t**2, t**3, t**4, t**5]) for t in ts])
            plt.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3)
        if self.path:
            path_xy = np.array([s[:2] for s in self.path])
            plt.plot(path_xy[:, 0], path_xy[:, 1], 'r-', linewidth=2, label='Optimal Path')
        plt.scatter(self.start[0], self.start[1], c='green', s=100, label='Start')
        plt.scatter(self.goal[0], self.goal[1], c='red', s=100, label='Goal')
        plt.legend()
        plt.show()


# ============================================================
# 测试主程序
# ============================================================
if __name__ == "__main__":
    env = SimpleEnv(obstacles=[(5, 2, 1.0)])
    start = (1, 1, 0, 0)
    goal = (9, 9, 0.5, 0.5)

    planner = KinoBITStar(start, goal, env, iter_max=250, batch_size=400)
    planner.r = 6.0  # 扩大邻域
    path, cost, runtime = planner.planning()

    print(f"Found path with cost={cost:.3f} in {runtime:.2f}s")
    if path:
        print("Path states:")
        for s in path:
            print("  →", np.round(s[:2], 2))
    planner.visualize()
