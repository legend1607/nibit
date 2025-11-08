import numpy as np
from math import hypot

class Random2DEnv:
    """
    适配 KinoBITStar 的随机 2D 环境。
    支持：
      - 几何边界和障碍物；
      - 动力学状态 [x, y, vx, vy]；
      - 速度 / 加速度约束；
      - 碰撞检测（点与线段）。
    """

    def __init__(self, env_dict, mode="train"):
        self.mode = mode
        self.envs = env_dict
        self.episode_i = 0
        self.collision_check_count = 0

        # 环境边界与障碍物
        width, height = env_dict['env_dims']
        self.bound = [(0, 0), (width, height)]
        self.rect_obstacles = env_dict.get('rectangle_obstacles', [])
        self.circle_obstacles = env_dict.get('circle_obstacles', [])

        # 维度与配置
        self.config_dim = 2          # 平面位置 (x, y)
        self.dof = 2                 # 两个自由度
        self.q_bounds = np.array([[0, width], [0, height]], dtype=float)

        # 动力学约束参数
        self.vmax = np.array([1.0, 1.0])  # 最大速度
        self.amax = np.array([5.0, 5.0])  # 最大加速度
        self.dt = env_dict.get("dt", 0.1)

        # 起点与目标
        self.start = env_dict.get('start', np.array([5.0, 5.0]))
        self.goal = env_dict.get('goal', np.array([width - 5.0, height - 5.0]))

        # 起点、终点扩展为 [q, dq]
        if len(self.start) == 2:
            self.start = np.hstack([self.start, np.zeros(2)])
        if len(self.goal) == 2:
            self.goal = np.hstack([self.goal, np.zeros(2)])

    # ==================== 基本接口 ====================

    def __str__(self):
        return f"Random2DEnv-Kino({self.mode})"

    def get_problem(self):
        """提供 KinoBITStar 兼容的 problem dict"""
        return dict(
            start=self.start,
            goal=self.goal,
            env=self,
            rect_obstacles=self.rect_obstacles,
            circle_obstacles=self.circle_obstacles,
            bound=self.bound
        )

    # ==================== 采样 ====================

    def sample_physical_pair(self):
        """
        采样一对物理一致状态 (x_from, x_to)
        满足动力学方程与加速度约束。
        """
        q1 = self.sample_empty_points()
        dq1 = np.random.uniform(-self.vmax, self.vmax)

        # 采样加速度 a 满足 |a| ≤ amax
        a = np.random.uniform(-self.amax, self.amax)

        # 预测下一个状态
        q2 = q1 + dq1 * self.dt + 0.5 * a * (self.dt ** 2)
        dq2 = dq1 + a * self.dt

        # 检查边界与碰撞
        if not self._edge_fp(q1, q2):
            return self.sample_physical_pair()  # 递归重采样
        if np.any(q2 < self.q_bounds[:, 0]) or np.any(q2 > self.q_bounds[:, 1]):
            return self.sample_physical_pair()

        return np.hstack([q1, dq1]), np.hstack([q2, dq2])

    def uniform_sample(self):
        x = np.random.uniform(self.bound[0][0], self.bound[1][0])
        y = np.random.uniform(self.bound[0][1], self.bound[1][1])
        return np.array([x, y], dtype=float)

    def sample_empty_points(self):
        """无碰撞的几何位置采样"""
        while True:
            p = self.uniform_sample()
            if self._point_in_free_space(p):
                return p

    def sample_state(self):
        """采样动力学状态 [x, y, vx, vy]"""
        q = self.sample_empty_points()
        dq = np.random.uniform(-self.vmax, self.vmax)
        return np.hstack([q, dq])

    # ==================== 碰撞检测 ====================

    def _point_in_free_space(self, q):
        """几何位置 (x, y) 是否在自由空间"""
        self.collision_check_count += 1
        x, y = q
        if x < self.bound[0][0] or x > self.bound[1][0] or y < self.bound[0][1] or y > self.bound[1][1]:
            return False
        for rx, ry, rw, rh in self.rect_obstacles:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return False
        for cx, cy, r in self.circle_obstacles:
            if hypot(x - cx, y - cy) <= r:
                return False
        return True

    def _edge_fp(self, a, b, step_size=0.5):
        """线段碰撞检测"""
        a, b = np.array(a), np.array(b)
        dist = np.linalg.norm(a - b)
        steps = max(int(dist / step_size), 1)
        for i in range(steps + 1):
            pt = a + (b - a) * (i / steps)
            if not self._point_in_free_space(pt):
                return False
        return True

    def _state_fp(self, state):
        """用于 BIT* 检查的接口"""
        q = np.array(state[:2], dtype=float)
        return self._point_in_free_space(q)

    # ==================== 动力学辅助 ====================

    def kin_reachable(self, x_from, x_to, tol=1e-2):
        """判断 x_to 是否可由 x_from 在加速度约束下到达"""
        x_from, x_to = np.array(x_from), np.array(x_to)
        q1, dq1 = x_from[:2], x_from[2:]
        q2, dq2 = x_to[:2], x_to[2:]

        # 计算所需加速度
        a_req = (dq2 - dq1) / self.dt
        if np.any(np.abs(a_req) > self.amax):
            return False

        # 预测下一个位置
        q_pred = q1 + dq1 * self.dt + 0.5 * a_req * (self.dt ** 2)
        if np.linalg.norm(q_pred - q2) > tol:
            return False

        # 检查速度和范围
        if np.any(np.abs(dq2) > self.vmax):
            return False
        if np.any(q2 < self.q_bounds[:, 0]) or np.any(q2 > self.q_bounds[:, 1]):
            return False

        # 碰撞检测
        return self._edge_fp(q1, q2)

    def kin_edge_free(self, x_from, x_to, steps=10):
        """沿动力学轨迹检查碰撞"""
        x_from, x_to = np.array(x_from), np.array(x_to)
        q1, dq1 = x_from[:2], x_from[2:]
        q2, dq2 = x_to[:2], x_to[2:]
        a_req = (dq2 - dq1) / self.dt
        for i in range(steps + 1):
            t = (i / steps) * self.dt
            q_t = q1 + dq1 * t + 0.5 * a_req * (t ** 2)
            if not self._point_in_free_space(q_t):
                return False
        return True

    # ==================== 其他 ====================

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def in_goal_region(self, state, eps=2.0):
        q = np.array(state[:2])
        return np.linalg.norm(q - np.array(self.goal[:2])) < eps

    def set_random_init_goal(self):
        """随机起止点（确保距离合理）"""
        self.start = np.hstack([self.sample_empty_points(), np.zeros(2)])
        self.goal = np.hstack([self.sample_empty_points(), np.zeros(2)])
        while self.distance(self.start[:2], self.goal[:2]) < 5.0:
            self.goal = np.hstack([self.sample_empty_points(), np.zeros(2)])
        return self.get_problem()
