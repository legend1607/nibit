import numpy as np
from math import hypot

class Random2DEnv:
    """
    随机2D环境类（连续空间 + 矩形/圆形障碍）
    """
    def __init__(self, env_dict, mode="train"):
        self.mode = mode
        self.envs = env_dict
        self.episode_i = 0
        self.collision_check_count = 0
        width, height = env_dict['env_dims']
        self.bound = [(0, 0), (width, height)]
        self.env_info = None
        self.start = env_dict.get('start', [None])[0]
        self.goal = env_dict.get('goal', [None])[0]
        self.config_dim = len(env_dict['env_dims'])  # 2D
        self.rect_obstacles = env_dict.get('rectangle_obstacles', [])
        self.circle_obstacles = env_dict.get('circle_obstacles', [])

    def __str__(self):
        return f"Random2DEnv({self.mode})"

    # ================= 核心接口 =================
    def get_problem(self):
        return dict(
            start=self.start,
            goal=self.goal,
            rect_obstacles=self.rect_obstacles,
            circle_obstacles=self.circle_obstacles,
            bound=self.bound
        )

    def uniform_sample(self):
        """连续空间随机采样"""
        x = np.random.uniform(self.bound[0][0], self.bound[1][0])
        y = np.random.uniform(self.bound[0][1], self.bound[1][1])
        return np.array([x, y], dtype=float)

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def interpolate(self, a, b, ratio):
        a, b = np.array(a), np.array(b)
        return a + ratio * (b - a)

    def in_goal_region(self, state, eps=5):
        return self.distance(state, self.goal) < eps and self._point_in_free_space(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        if action is not None:
            new_state = state + action

        # 限制在边界内
        new_state[0] = np.clip(new_state[0], self.bound[0][0], self.bound[1][0])
        new_state[1] = np.clip(new_state[1], self.bound[0][1], self.bound[1][1])

        if not check_collision:
            return new_state, action, True, self.in_goal_region(new_state)

        no_collision = self._edge_fp(state, new_state)
        done = no_collision and self.in_goal_region(new_state)
        return new_state, action, no_collision, done

    # ================= 碰撞检测 =================

    def _point_in_free_space(self, state):
        self.collision_check_count += 1
        x, y = state
        # 检查边界
        if x < self.bound[0][0] or x > self.bound[1][0] or y < self.bound[0][1] or y > self.bound[1][1]:
            return False
        # 检查矩形障碍
        for rx, ry, rw, rh in self.rect_obstacles:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return False
        # 检查圆形障碍
        for cx, cy, r in self.circle_obstacles:
            if hypot(x - cx, y - cy) <= r:
                return False
        return True

    def _edge_fp(self, a, b, step_size=0.5):
        """连续空间线段碰撞检测"""
        a, b = np.array(a), np.array(b)
        dist = self.distance(a, b)
        steps = max(int(dist / step_size), 1)
        for i in range(steps + 1):
            pt = a + (b - a) * (i / steps)
            if not self._point_in_free_space(pt):
                return False
        return True

    def _state_fp(self, state):
        return self._point_in_free_space(state)

    def sample_empty_points(self):
        while True:
            p = self.uniform_sample()
            if self._point_in_free_space(p):
                return p

    def set_random_init_goal(self):
        self.start = self.sample_empty_points()
        self.goal = self.sample_empty_points()
        while self.distance(self.start, self.goal) < 1.0:
            self.goal = self.sample_empty_points()
        return self.get_problem()
