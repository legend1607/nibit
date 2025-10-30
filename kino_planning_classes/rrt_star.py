import numpy as np
import math
import random
import matplotlib.pyplot as plt

INF = float("inf")

class RRTStarDyn2D:
    def __init__(self, start, goal, env, iter_max=5000, step_len=0.5, search_radius=2.0, dt=0.1, vel_max=1.0):
        """
        动力学 RRT* 2D
        state = [x, y, vx, vy]
        """
        self.env = env
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.iter_max = iter_max
        self.step_len = step_len
        self.search_radius = search_radius
        self.dt = dt
        self.vel_max = vel_max

        # 树结构
        self.vertices = [self.start]
        self.vertex_parents = {tuple(self.start): None}
        self.g_scores = {tuple(self.start): 0.0}

        self.path_solutions = []

    # ------------------- 核心工具 -------------------

    def distance(self, s1, s2):
        return np.linalg.norm(np.array(s1[:2]) - np.array(s2[:2]))

    def nearest_neighbor(self, node):
        dists = [self.distance(node, v) for v in self.vertices]
        idx = np.argmin(dists)
        return self.vertices[idx]

    def steer(self, from_node, to_node):
        vec = np.array(to_node[:2]) - np.array(from_node[:2])
        dist = np.linalg.norm(vec)
        if dist > self.step_len:
            vec = vec / dist * self.step_len
        new_pos = np.array(from_node[:2]) + vec

        new_vel = vec / self.dt
        vel_norm = np.linalg.norm(new_vel)
        if vel_norm > self.vel_max:
            new_vel = new_vel / vel_norm * self.vel_max

        return np.array([new_pos[0], new_pos[1], new_vel[0], new_vel[1]])

    def find_near_neighbors(self, node):
        neighbors = [v for v in self.vertices if self.distance(node, v) <= self.search_radius]
        return neighbors

    def cost(self, node):
        cost = 0.0
        curr = tuple(node)
        while self.vertex_parents[curr] is not None:
            parent = self.vertex_parents[curr]
            cost += self.distance(np.array(curr), np.array(parent))
            curr = parent
        return cost

    def choose_parent(self, new_node, neighbors):
        best_parent = None
        min_cost = INF
        for n in neighbors:
            if self.env._edge_fp(n[:2], new_node[:2]):
                c = self.cost(n) + self.distance(n, new_node)
                if c < min_cost:
                    min_cost = c
                    best_parent = n
        if best_parent is not None:
            self.vertex_parents[tuple(new_node)] = tuple(best_parent)
            self.g_scores[tuple(new_node)] = min_cost
        else:
            nearest = self.nearest_neighbor(new_node)
            self.vertex_parents[tuple(new_node)] = tuple(nearest)
            self.g_scores[tuple(new_node)] = self.cost(nearest)

    def rewire(self, new_node, neighbors):
        for n in neighbors:
            if self.env._edge_fp(new_node[:2], n[:2]):
                c_new = self.g_scores[tuple(new_node)] + self.distance(new_node, n)
                if c_new < self.g_scores[tuple(n)]:
                    self.vertex_parents[tuple(n)] = tuple(new_node)
                    self.g_scores[tuple(n)] = c_new

    def extract_path(self, goal_node):
        path = [goal_node]
        curr = tuple(goal_node)
        while self.vertex_parents[curr] is not None:
            curr = self.vertex_parents[curr]
            path.append(np.array(curr))
        path.reverse()
        return path

    # ------------------- Informed RRT* 采样 -------------------

    def init_informed_sampling(self):
        self.c_min = self.distance(self.start, self.goal)
        self.center = (self.start[:2] + self.goal[:2]) / 2
        a1 = self.goal[:2] - self.start[:2]
        norm_a1 = np.linalg.norm(a1)
        if norm_a1 < 1e-6:
            self.C = None
            return
        a1 /= norm_a1
        X = np.random.randn(2, 2)
        X[:, 0] = a1
        Q, _ = np.linalg.qr(X)
        if np.dot(Q[:, 0], a1) < 0:
            Q[:, 0] *= -1
        self.C = Q

    def sample_unit_ball(self):
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x**2 + y**2 < 1:
                return np.array([x, y])

    def sample_informed(self, c_best):
        if c_best == INF or self.C is None:
            return self.env.sample_empty_points()
        a = c_best / 2
        b = math.sqrt(max(c_best**2 - self.c_min**2, 0)) / 2
        L = np.diag([a, b])
        x_ball = self.sample_unit_ball()
        node = self.C @ (L @ x_ball) + self.center
        return np.array([node[0], node[1], 0, 0])

    # ------------------- 主规划 -------------------

    def planning(self, visual=False):
        self.init_informed_sampling()
        for k in range(self.iter_max):
            if self.path_solutions:
                c_best = min([self.g_scores[tuple(p)] for p in self.path_solutions])
            else:
                c_best = INF

            node_rand = self.sample_informed(c_best)
            node_nearest = self.nearest_neighbor(node_rand)
            node_new = self.steer(node_nearest, node_rand)

            if self.env._edge_fp(node_nearest[:2], node_new[:2]):
                self.vertices.append(node_new)
                neighbors = self.find_near_neighbors(node_new)
                self.choose_parent(node_new, neighbors)
                self.rewire(node_new, neighbors)

                if self.distance(node_new, self.goal) < self.step_len and self.env._edge_fp(node_new[:2], self.goal[:2]):
                    self.vertex_parents[tuple(self.goal)] = tuple(node_new)
                    self.g_scores[tuple(self.goal)] = self.cost(node_new) + self.distance(node_new, self.goal)
                    self.path_solutions.append(self.goal)

        if self.path_solutions:
            best_cost = INF
            best_path = None
            for p in self.path_solutions:
                c = self.g_scores[tuple(p)]
                if c < best_cost:
                    best_cost = c
                    best_path = self.extract_path(p)
            self.path = best_path
        else:
            self.path = []

        if visual:
            self.visualize_tree()

        return self.path

    # ------------------- 可视化 -------------------

    def visualize_tree(self):
        plt.figure(figsize=(8,8))
        plt.xlim(self.env.bound[0][0], self.env.bound[1][0])
        plt.ylim(self.env.bound[0][1], self.env.bound[1][1])
        plt.gca().set_aspect('equal', adjustable='box')
        # 绘制障碍
        for rx, ry, rw, rh in self.env.rect_obstacles:
            plt.gca().add_patch(plt.Rectangle((rx, ry), rw, rh, color='gray', alpha=0.5))
        for cx, cy, r in self.env.circle_obstacles:
            plt.gca().add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))
        # 树
        for child, parent in self.vertex_parents.items():
            if parent is not None:
                plt.plot([child[0], parent[0]], [child[1], parent[1]], c='skyblue', lw=0.5)
        # 最优路径
        if self.path:
            path_arr = np.array(self.path)
            plt.plot(path_arr[:,0], path_arr[:,1], 'r-', lw=2)
        # 起点/终点
        plt.plot(self.start[0], self.start[1], 'go', markersize=8)
        plt.plot(self.goal[0], self.goal[1], 'ro', markersize=8)
        plt.show()
def get_rrt_planner(args, problem, neural_wrapper=None):
    """
    动力学 RRT* 规划器工厂函数
    自动将位置 state 拼接为 [x, y, 0, 0] 作为初始状态和目标状态
    """
    # 自动补速度
    x_start = np.array(problem['x_start'], dtype=float)
    if len(x_start) == 2:
        x_start = np.hstack([x_start, [0.0, 0.0]])

    x_goal = np.array(problem['x_goal'], dtype=float)
    if len(x_goal) == 2:
        x_goal = np.hstack([x_goal, [0.0, 0.0]])

    planner = RRTStarDyn2D(
        start=x_start,
        goal=x_goal,
        env=problem['env'],
        iter_max=args.iter_max,
        step_len=args.step_len,
        search_radius=args.search_radius,
        dt=getattr(args, 'dt', 0.1),
        vel_max=getattr(args, 'vel_max', 1.0)
    )
    return planner
