import numpy as np
import pybullet as p
import pybullet_data
import pickle
from time import time
from environment.timer import Timer

class RobotArmEnv:
    """
    通用机械臂环境类 (UR5, KUKA等)
    """
    RRT_EPS = 0.5
    voxel_r = 0.1

    def __init__(self, arm_file, map_file=None, GUI=False, base_position=[0,0,0]):
        self.arm_file = arm_file
        self.base_position = base_position

        # pybullet连接
        if GUI:
            p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1 --background_color_blue=1')
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0,0,0.1])

        # 初始化环境
        self.reset_env()

        # 任务集
        self.maps = {}
        self.episode_i = 0
        if map_file is not None:
            with open(map_file, 'rb') as f:
                self.problems = pickle.load(f)
        else:
            self.problems = []

        self.timer = Timer()
        self.collision_check_count = 0
        self.collision_time = 0
        self.collision_point = None
        self.obstacles = []

    def reset_env(self):
        """重置仿真环境"""
        p.resetSimulation()
        self.obs_ids = []
        self.armId = p.loadURDF(self.arm_file, self.base_position, [0,0,0,1], useFixedBase=True)
        plane = p.createCollisionShape(p.GEOM_PLANE)
        self.plane = p.createMultiBody(0, plane)
        p.setGravity(0, 0, -9.8)

        # 关节信息
        n_joints = p.getNumJoints(self.armId)
        joints = [p.getJointInfo(self.armId, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        self.pose_range = [(p.getJointInfo(self.armId, j)[8], p.getJointInfo(self.armId, j)[9]) for j in self.joints]
        self.config_dim = len(self.joints)
        self.bound = np.array(self.pose_range).T.reshape(-1)

        # link 索引
        _link_name_to_index = {p.getBodyInfo(self.armId)[0].decode('UTF-8'): -1}
        for _id in range(n_joints):
            _name = p.getJointInfo(self.armId, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id
        self.tip_index = _link_name_to_index.get('ee_link', self.config_dim-1)

    # =================== 初始化任务 ===================

    def init_new_problem(self, index=None):
        if index is None:
            self.index = self.episode_i
        else:
            self.index = index

        obstacles, start, goal, path = self.problems[self.index]

        self.episode_i = (self.episode_i + 1) % len(self.problems)
        self.collision_check_count = 0

        self.reset_env()
        self.collision_point = None

        self.obstacles = obstacles
        self.init_state = start
        self.goal_state = goal
        self.path = path
        self.obs_ids = []

        for halfExtents, basePosition in obstacles:
            self.obs_ids.append(self.create_voxel(halfExtents, basePosition))

        return self.get_problem()

    def init_new_problem_with_config(self, start, goal, obstacles):
        self.index = 0
        self.collision_check_count = 0
        self.reset_env()
        self.collision_point = None

        self.obstacles = obstacles
        self.init_state = start
        self.goal_state = goal
        self.obs_ids = []

        for halfExtents, basePosition in obstacles:
            self.obs_ids.append(self.create_voxel(halfExtents, basePosition))

        return self.get_problem()

    # =================== 核心功能 ===================

    def uniform_sample(self, n=1):
        sample = np.random.uniform(np.array(self.pose_range)[:,0], np.array(self.pose_range)[:,1], size=(n,self.config_dim))
        if n == 1:
            return sample.reshape(-1)
        return sample

    def distance(self, from_state, to_state):
        diff = to_state - from_state
        return np.sqrt(np.sum(diff**2))

    def interpolate(self, from_state, to_state, ratio):
        new_state = from_state + ratio*(to_state - from_state)
        new_state = np.maximum(new_state, np.array(self.pose_range)[:,0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:,1])
        return new_state

    def in_goal_region(self, state):
        return self.distance(state, self.goal_state) < self.RRT_EPS and self._state_fp(state)

    def set_config(self, config, armId=None):
        if armId is None:
            armId = self.armId
        for i, jointId in enumerate(self.joints):
            p.resetJointState(armId, jointId, config[i])
        p.performCollisionDetection()
    
    def sample_n_points(self, n, need_negative=False):
        """
        采样 n 个有效关节状态，可选同时返回无效状态
        :param n: 采样数量
        :param need_negative: 是否返回无效状态
        :return: 有效状态列表，若 need_negative=True 同时返回无效状态列表
        """
        samples = []
        negative = [] if need_negative else None

        while len(samples) < n:
            sample = self.uniform_sample()
            if self._state_fp(sample):
                samples.append(sample)
            elif need_negative:
                negative.append(sample)

        if need_negative:
            return samples, negative
        else:
            return samples
    
    def step(self, state, action=None, new_state=None, check_collision=True):
        """
        状态更新 + 碰撞检测
        :param state: 当前关节状态
        :param action: 关节动作 (可选)
        :param new_state: 指定下一状态 (可选)
        :param check_collision: 是否进行碰撞检测
        :return: new_state, action [, no_collision, done]
        """
        if action is not None:
            new_state = state + action

        # 保证状态在关节范围内
        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        action = new_state - state

        if not check_collision:
            return new_state, action

        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done
    # =================== 碰撞检测 ===================

    def _valid_state(self, state):
        return (state >= np.array(self.pose_range)[:,0]).all() and (state <= np.array(self.pose_range)[:,1]).all()

    def _point_in_free_space(self, state):
        t0 = time()
        if not self._valid_state(state):
            return False
        self.set_config(state)
        in_free = len(p.getContactPoints(self.armId)) == 0
        if not in_free:
            self.collision_point = state
        self.collision_check_count += 1
        self.collision_time += time()-t0
        return in_free

    def _state_fp(self, state):
        return self._point_in_free_space(state)

    def _edge_fp(self, state, new_state):
        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        disp = new_state - state
        d = self.distance(state, new_state)
        K = max(int(d / self.RRT_EPS), 1)
        for k in range(K+1):
            c = state + k*1./K*disp
            if not self._point_in_free_space(c):
                return False
        return True

    def _iterative_check_segment(self, left, right, tol=0.1):
        """
        递归二分检查线段上的碰撞
        :param left: 线段起点
        :param right: 线段终点
        :param tol: 阈值，小于该距离不再递归
        :return: True 表示线段无碰撞, False 表示有碰撞
        """
        if np.linalg.norm(right - left) > tol:
            mid = (left + right) / 2.0
            if not self._state_fp(mid):
                self.collision_point = mid
                return False
            return self._iterative_check_segment(left, mid, tol) and \
                   self._iterative_check_segment(mid, right, tol)
        return True
    # =================== 障碍物 ===================

    def create_voxel(self, halfExtents, basePosition):
        colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        visId = p.createVisualShape(p.GEOM_BOX, rgbaColor=[0.5,0.5,0.5,0.8], halfExtents=halfExtents)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colId, baseVisualShapeIndex=visId, basePosition=basePosition)

    # =================== 获取任务信息 ===================

    def get_problem(self, width=15, index=None):
        if index is None:
            problem = {
                "map": np.array(self.obs_map(width)).astype(float),
                "init_state": self.init_state,
                "goal_state": self.goal_state
            }
            self.maps[self.index] = problem
            return problem
        else:
            return self.maps[index]

    def obs_map(self, num=15):
        resolution = 2./(num-1)
        grid_pos = [np.linspace(-1,1,num=num) for _ in range(3)]
        points_pos = np.stack(np.meshgrid(*grid_pos), axis=-1).reshape(-1,3)
        points_obs = np.zeros(points_pos.shape[0], dtype=bool)

        for obstacle in self.obstacles:
            size, base = obstacle
            low, high = base-size, base+size
            mask = np.all((points_pos >= low) & (points_pos <= high), axis=1)
            points_obs = np.logical_or(points_obs, mask)
        return points_obs.reshape((num,num,num))

    # =================== 路径操作 ===================

    def aug_path(self):
        """路径插值增强"""
        result = [self.init_state]
        path = np.array(self.path)
        agent = np.array(path[0])
        next_index = 1
        while next_index < len(path):
            if np.linalg.norm(path[next_index]-agent) <= self.RRT_EPS:
                agent = path[next_index]
                next_index += 1
            else:
                agent = agent + self.RRT_EPS * (path[next_index]-agent) / np.linalg.norm(path[next_index]-agent)
            result.append(np.array(agent))
        return result

    def set_random_init_goal(self):
        while True:
            points = self.uniform_sample(n=2)
            init, goal = points[0], points[1]
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    # =================== 动画绘制 ===================

    def plot_path(self, path):
        p.resetSimulation()
        self.set_config(path[0])
        prev_pos = p.getLinkState(self.armId, self.tip_index)[0]
        for state in path[1:]:
            self.set_config(state)
            new_pos = p.getLinkState(self.armId, self.tip_index)[0]
            p.addUserDebugLine(prev_pos, new_pos, [1,0,0], 2, 0)
            prev_pos = new_pos
