import numpy as np
from math import hypot

class Random2DKinodynamicEnv:
    """
    Kinodynamic 2D 随机环境类（位置 + 速度，双积分动力学）
    """
    def __init__(self, env_dict, max_vel=2.0, mode="train"):
        """
        env_dict: dict 包含：
            - 'rect_obstacles': [[x, y, w, h], ...]
            - 'circle_obstacles': [[x, y, r], ...]
            - 'start': [x0, y0]
            - 'goal': [xg, yg]
            - 'env_dims': [width, height]
        max_vel: 最大速度，用于速度采样
        """
        self.mode = mode
        self.env_dict = env_dict
        self.max_vel = max_vel

        width, height = env_dict['env_dims']
        self.bound = [(0, 0), (width, height)]
        self.start = np.array(env_dict.get('start', [0,0]), dtype=float)
        self.goal = np.array(env_dict.get('goal', [width-1, height-1]), dtype=float)

        self.rect_obstacles = env_dict.get('rect_obstacles', [])
        self.circle_obstacles = env_dict.get('circle_obstacles', [])

        self.state_dim = 4  # [x, y, vx, vy]
        self.collision_check_count = 0

        print(f"Initialized kinodynamic 2D environment in mode={mode}")

    # ================= 核心接口 =================
    def get_problem(self):
        return {
            "start": self.start,
            "goal": self.goal,
            "bound": self.bound,
            "rect_obstacles": self.rect_obstacles,
            "circle_obstacles": self.circle_obstacles
        }

    def uniform_sample(self):
        """采样位置 + 速度"""
        x = np.random.uniform(self.bound[0][0], self.bound[1][0])
        y = np.random.uniform(self.bound[0][1], self.bound[1][1])
        vx = np.random.uniform(-self.max_vel, self.max_vel)
        vy = np.random.uniform(-self.max_vel, self.max_vel)
        return np.array([x, y, vx, vy], dtype=float)

    def distance(self, a, b):
        """只考虑位置距离"""
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def in_goal_region(self, state, eps=0.5):
        pos = state[:2]
        return self.distance(pos, self.goal) < eps and self._point_in_free_space(pos)

    def step(self, state, action, dt=0.1, check_collision=True):
        """
        双积分动力学步进：
        state: [x, y, vx, vy]
        action: [ax, ay] 加速度
        dt: 时间步长
        """
        new_state = np.zeros_like(state)
        new_state[:2] = state[:2] + state[2:] * dt + 0.5 * np.array(action) * dt**2
        new_state[2:] = state[2:] + np.array(action) * dt

        # 限制在边界内
        new_state[0] = np.clip(new_state[0], self.bound[0][0], self.bound[1][0])
        new_state[1] = np.clip(new_state[1], self.bound[0][1], self.bound[1][1])

        if not check_collision:
            return new_state, True, self.in_goal_region(new_state)

        no_collision = self._trajectory_fp(state, new_state, dt)
        done = no_collision and self.in_goal_region(new_state)
        return new_state, no_collision, done

    # ================= 碰撞检测 =================
    def _point_in_free_space(self, pos):
        x, y = pos
        # 边界检查
        if x < self.bound[0][0] or x > self.bound[1][0] or y < self.bound[0][1] or y > self.bound[1][1]:
            return False
        # 矩形障碍
        for rx, ry, rw, rh in self.rect_obstacles:
            if rx <= x <= rx+rw and ry <= y <= ry+rh:
                return False
        # 圆形障碍
        for cx, cy, r in self.circle_obstacles:
            if hypot(x - cx, y - cy) <= r:
                return False
        self.collision_check_count += 1
        return True

    def _trajectory_fp(self, state, new_state, dt, step_size=0.1):
        """
        离散化动力学轨迹，检测碰撞
        state: [x, y, vx, vy]
        new_state: [x, y, vx, vy]
        """
        pos0 = np.array(state[:2])
        pos1 = np.array(new_state[:2])
        dist = np.linalg.norm(pos1 - pos0)
        steps = max(int(dist / step_size), 1)
        for i in range(steps + 1):
            t = i / steps
            # 双积分公式
            pos = pos0 + state[2:] * t * dt + 0.5 * (new_state[2:] - state[2:]) * (t*dt)**2 / dt
            if not self._point_in_free_space(pos):
                return False
        return True

    # ================= 其他工具 =================
    def sample_empty_points(self):
        while True:
            p = self.uniform_sample()
            if self._point_in_free_space(p[:2]):
                return p

    def set_random_init_goal(self):
        self.start = self.sample_empty_points()
        self.goal = self.sample_empty_points()
        while self.distance(self.start, self.goal) < 1.0:
            self.goal = self.sample_empty_points()
        return self.get_problem()
