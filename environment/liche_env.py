import time
from typing import List, Optional, Sequence, Tuple, Dict, Set, Any

import numpy as np
import pybullet as p
import pybullet_data


def interpolate_path(path: Sequence[Sequence[float]], step_size: float = 0.1):
    """
    对路径进行线性插值，使关节或坐标变化平滑
    :param path: 原始路径 (list of configs)
    :param step_size: 相邻节点最大欧式间距
    :return: 插值后的平滑路径 (list of np.ndarray)
    """
    if len(path) < 2:
        return list(path)

    interpolated = [np.array(path[0], dtype=float)]
    for i in range(len(path) - 1):
        q1, q2 = np.array(path[i], dtype=float), np.array(path[i + 1], dtype=float)
        dist = np.linalg.norm(q2 - q1)
        if dist <= 0:
            continue
        n_steps = max(int(dist / step_size), 1)
        for j in range(1, n_steps + 1):
            q_interp = q1 + (q2 - q1) * (j / n_steps)
            interpolated.append(q_interp)
    return interpolated


class LicheEnv:
    """
    LicheEnv - 基于 PyBullet 的 LICHE 机械臂关节空间规划与碰撞检测环境。

    特性：
    - 支持随机障碍物（方块 / 球体）添加与清除
    - 关节空间采样 / 距离 / 插值 / step 接口，兼容 Random2DEnv 风格
    - 高效碰撞检测：
        * 优先对 (arm, obstacle) 使用 getClosestPoints
        * 支持 ignored_body_ids（例如平面等）
    - set_random_init_goal / sample_empty_points 直接用于随机规划任务
    - 支持 with 上下文管理（with LicheEnv(...) as env:）
    """

    EPS: float = 0.05          # goal 区域半径
    RRT_EPS: float = 0.08      # edge discretization 步长
    CLOSEST_DIST: float = 1e-3 # getClosestPoints 距离阈值
    MAX_SAMPLE_TRIALS: int = 10000

    def __init__(self, GUI: bool = False, arm_file: str = "liche/urdf/liche.urdf"):
        """
        :param GUI: 是否以 GUI 模式启动 PyBullet
        :param arm_file: LICHE URDF 路径（相对于 pybullet_data 的路径）
        """
        self.GUI = GUI
        self.arm_file = arm_file

        # 碰撞统计
        self.collision_check_count: int = 0
        self.collision_time: float = 0.0

        # 障碍体记录（body ids）
        self.obstacles: List[int] = []

        # 忽略碰撞的 body id 集合（例如平面）
        self.ignored_body_ids: Set[int] = set()

        # 连接 PyBullet
        if GUI:
            self.cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=5, cameraYaw=90, cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        self.arm_id: int = p.loadURDF(
            self.arm_file,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=True,
        )

        # 获取机械臂关节维度与范围
        self.config_dim: int = p.getNumJoints(self.arm_id)
        pr = []
        for j in range(self.config_dim):
            info = p.getJointInfo(self.arm_id, j)
            lower = info[8]  # jointLowerLimit
            upper = info[9]  # jointUpperLimit
            # 某些 URDF 关节可能给出 lower>upper（未知/连续关节），做保护处理：
            if lower > upper:
                # 如果是连续关节，使用大范围作为近似
                lower, upper = -np.pi, np.pi
            pr.append([lower, upper])
        self.pose_range: np.ndarray = np.array(pr, dtype=float)  # shape: (config_dim, 2)
        # bound 展平为 [l0, u0, l1, u1, ...]
        self.bound: np.ndarray = self.pose_range.T.reshape(-1)
        self.end_effector_index: int = self.config_dim - 1

        # 起点 / 目标
        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None

    # -------------------------
    # 基础工具方法
    # -------------------------
    def __str__(self) -> str:
        return f"LicheEnv({self.config_dim}D)"

    def __enter__(self) -> "LicheEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """清理 PyBullet 资源（可安全重复调用）"""
        try:
            p.removeAllUserDebugItems()
        except Exception:
            pass
        try:
            if p.isConnected(self.cid):
                p.disconnect(self.cid)
        except Exception:
            pass

    # -------------------------
    # 障碍物管理
    # -------------------------
    def add_box_obstacle(
        self,
        half_extents: Sequence[float],
        base_pos: Sequence[float],
        rgba: Sequence[float] = (0, 0, 0, 1),
    ) -> int:
        """添加静态方块障碍，返回 body id 并记录到 self.obstacles"""
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=base_pos,
        )
        self.obstacles.append(body_id)
        return body_id

    def add_sphere_obstacle(
        self,
        radius: float,
        base_pos: Sequence[float],
        rgba: Sequence[float] = (0, 0, 1, 1),
    ) -> int:
        """添加静态球体障碍，返回 body id 并记录到 self.obstacles"""
        collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=base_pos,
        )
        self.obstacles.append(body_id)
        return body_id

    def remove_obstacle(self, body_id: int) -> None:
        """在场景中移除指定障碍（若存在）并从 self.obstacles 中清除"""
        try:
            p.removeBody(body_id)
        except Exception:
            pass
        if body_id in self.obstacles:
            self.obstacles.remove(body_id)
        if body_id in self.ignored_body_ids:
            self.ignored_body_ids.discard(body_id)

    def clear_obstacles(self) -> None:
        """删除场景中所有已记录的障碍并清空 self.obstacles（保留 plane 与 arm）"""
        for obs_id in list(self.obstacles):
            try:
                p.removeBody(obs_id)
            except Exception:
                pass
        self.obstacles = []

    def ignore_body(self, body_id: int) -> None:
        """将某个 body id 加入忽略集合（碰撞检测时忽略）"""
        self.ignored_body_ids.add(body_id)

    def unignore_body(self, body_id: int) -> None:
        """将某个 body id 从忽略集合中移除"""
        self.ignored_body_ids.discard(body_id)

    # -------------------------
    # 状态 / 动作相关
    # -------------------------
    def set_config(self, joint_values: Sequence[float], robot_id: Optional[int] = None) -> None:
        """
        安全地设置机械臂关节角（重置状态，不做动力学模拟）
        - 会检查长度并将值裁剪到 self.pose_range 范围内
        :param joint_values: 长度应为 config_dim 的序列
        :param robot_id: 可选，指定 robot body id（默认 self.arm_id）
        """
        if robot_id is None:
            robot_id = self.arm_id
        arr = np.asarray(joint_values, dtype=float).flatten()
        if arr.size != self.config_dim:
            raise ValueError(f"[LicheEnv] 状态维度不匹配: got {arr.size}, expected {self.config_dim}")
        clipped = np.clip(arr, self.pose_range[:, 0], self.pose_range[:, 1])
        for j in range(self.config_dim):
            p.resetJointState(robot_id, j, float(clipped[j]))

    def get_config(self, robot_id: Optional[int] = None) -> np.ndarray:
        """
        获取当前机械臂关节角（从 pybullet 读取）
        :return: numpy array length config_dim
        """
        if robot_id is None:
            robot_id = self.arm_id
        vals = []
        for j in range(self.config_dim):
            state = p.getJointState(robot_id, j)
            vals.append(float(state[0]))  # state[0] 是位置
        return np.array(vals, dtype=float)

    # -------------------------
    # Random2DEnv 兼容接口
    # -------------------------
    def get_problem(self) -> Dict[str, Any]:
        """返回当前问题信息（供外部算法使用）"""
        return dict(
            start=self.start,
            goal=self.goal,
            bound=self.bound.copy(),
            obstacles=list(self.obstacles),
        )

    def uniform_sample(self) -> np.ndarray:
        """在关节空间内均匀采样（返回 numpy 数组）"""
        s = np.random.uniform(self.pose_range[:, 0], self.pose_range[:, 1])
        return s.astype(float)

    def distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        """欧几里得距离（关节空间）"""
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        return float(np.linalg.norm(a_arr - b_arr))

    def interpolate(self, a: Sequence[float], b: Sequence[float], ratio: float) -> np.ndarray:
        """线性插值 + 范围裁剪"""
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        s = a_arr + ratio * (b_arr - a_arr)
        return np.clip(s, self.pose_range[:, 0], self.pose_range[:, 1])

    def step(
        self,
        state: Sequence[float],
        action: Optional[Sequence[float]] = None,
        new_state: Optional[Sequence[float]] = None,
        check_collision: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool, bool]:
        """
        与 Random2DEnv 一致接口：
        - 输入 state + action 或直接 new_state
        - 返回: (new_state, action, collision_free, done)
        """
        if (action is None) == (new_state is None):
            raise ValueError("Provide either 'action' or 'new_state', not both or neither.")

        state_arr = np.asarray(state, dtype=float)
        if action is not None:
            new_state_arr = state_arr + np.asarray(action, dtype=float)
            action_out = np.asarray(action, dtype=float)
        else:
            new_state_arr = np.asarray(new_state, dtype=float)
            action_out = None

        new_state_arr = np.clip(new_state_arr, self.pose_range[:, 0], self.pose_range[:, 1])

        if not check_collision:
            return new_state_arr, action_out, True, self.in_goal_region(new_state_arr)

        no_collision = self._edge_fp(state_arr, new_state_arr)
        done = no_collision and self.in_goal_region(new_state_arr)
        return new_state_arr, action_out, no_collision, done

    # -------------------------
    # 碰撞检测相关
    # -------------------------
    def _point_in_free_space(self, state: Sequence[float]) -> bool:
        """
        检查在给定 joint config 下机械臂是否无碰撞。
        - 优先使用 getClosestPoints 与 self.obstacles（若存在）
        - 否则使用 getContactPoints 并以 ignored_body_ids 过滤
        """
        t0 = time.time()

        s_arr = np.asarray(state, dtype=float).flatten()
        if s_arr.size != self.config_dim:
            raise ValueError(f"[LicheEnv] 状态维度不匹配: got {s_arr.size}, expected {self.config_dim}")

        # 将关节直接设置到该状态（resetJointState）
        for j in range(self.config_dim):
            p.resetJointState(self.arm_id, j, float(s_arr[j]))

        self.collision_check_count += 1

        collision_found = False
        if len(self.obstacles) > 0:
            # 有显式障碍时：对每个障碍体检查最近点
            for obs_id in self.obstacles:
                if obs_id in self.ignored_body_ids:
                    continue
                pts = p.getClosestPoints(self.arm_id, obs_id, distance=self.CLOSEST_DIST)
                if pts:  # 非空 → 存在接触/重叠
                    collision_found = True
                    break
        else:
            # 没有显式障碍时，回退到 getContactPoints
            contacts = p.getContactPoints(bodyA=self.arm_id)
            for c in contacts:
                try:
                    bodyA = int(c[1])
                    bodyB = int(c[2])
                except Exception:
                    collision_found = True
                    break

                if (bodyA in self.ignored_body_ids) and (bodyB in self.ignored_body_ids):
                    continue

                other = (
                    bodyB
                    if bodyA == self.arm_id
                    else bodyA
                    if bodyB == self.arm_id
                    else None
                )
                if other is None:
                    collision_found = True
                    break
                if other in self.ignored_body_ids:
                    continue
                collision_found = True
                break

        free = (not collision_found)
        self.collision_time += time.time() - t0
        return free

    def _edge_fp(self, a: Sequence[float], b: Sequence[float]) -> bool:
        """
        线段碰撞检测（配置空间）
        - 默认假设 a 已经是无碰撞（典型 RRT/BIT* 前提），只检查 b 和中间点
        - 对线段内部按 RRT_EPS 步长进行插值检查（使用 ceil 保证覆盖）
        - 返回 True 表示整条线段无碰撞
        """
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)

        # 终点检查
        if not self._point_in_free_space(b_arr):
            return False

        d = self.distance(a_arr, b_arr)
        if d <= 0.0:
            return True

        steps = max(int(np.ceil(d / self.RRT_EPS)), 1)
        # 已检查终点，只需检查中间点
        for i in range(1, steps):
            ratio = i / steps
            s = a_arr + ratio * (b_arr - a_arr)
            if not self._point_in_free_space(s):
                return False
        return True

    def _state_fp(self, state: Sequence[float]) -> bool:
        """alias"""
        return self._point_in_free_space(state)

    # -------------------------
    # 采样 / 初始化 / 目标判定
    # -------------------------
    def sample_empty_points(self) -> Optional[np.ndarray]:
        """循环采样直到得到一个无碰撞的配置；若尝试上限仍失败则返回 None"""
        for _ in range(self.MAX_SAMPLE_TRIALS):
            s = self.uniform_sample()
            if self._point_in_free_space(s):
                return s
        return None

    def set_random_init_goal(self) -> Dict[str, Any]:
        """随机生成有效的 start / goal（不会太接近）。若失败则返回 start/goal 为 None。"""
        start = self.sample_empty_points()
        if start is None:
            self.start = None
            self.goal = None
            return self.get_problem()

        goal = self.sample_empty_points()
        if goal is None:
            self.start = start
            self.goal = None
            return self.get_problem()

        # 保证 start 与 goal 不太接近
        max_retry = 50
        retry = 0
        while self.distance(start, goal) < 0.1 and retry < max_retry:
            g = self.sample_empty_points()
            if g is None:
                break
            goal = g
            retry += 1

        if self.distance(start, goal) < 0.1:
            # 仍然过近，视为失败
            self.start = start
            self.goal = None
            return self.get_problem()

        self.start = start
        self.goal = goal
        return self.get_problem()

    def in_goal_region(self, state: Sequence[float], eps: float = EPS) -> bool:
        """判断是否到达目标（默认仍要求当前状态无碰撞）"""
        if self.goal is None:
            return False
        state_arr = np.asarray(state, dtype=float)
        return (self.distance(state_arr, self.goal) < eps) and self._point_in_free_space(state_arr)

    # -------------------------
    # 可视化 / 其他工具
    # -------------------------
    def get_end_effector_pos(self, config: Sequence[float]) -> np.ndarray:
        """
        给定关节配置，返回末端坐标（xyz），以 numpy.ndarray 返回
        注意：此函数会把关节状态 reset 到给定配置（不会做动力学推进）
        """
        self.set_config(config)
        pos = p.getLinkState(self.arm_id, self.end_effector_index)[0]
        return np.array(pos, dtype=float)

    def get_collision_stats(self) -> Dict[str, float]:
        """返回碰撞检测统计信息"""
        avg_time = (
            self.collision_time / self.collision_check_count
            if self.collision_check_count > 0
            else 0.0
        )
        return dict(
            collision_check_count=self.collision_check_count,
            total_collision_time=self.collision_time,
            avg_collision_check_time=avg_time,
        )

    def render_path(
        self,
        path: Sequence[Sequence[float]],
        color: Sequence[float] = (1, 0, 0),
        line_width: float = 2.0,
        life_time: float = 0.0,
        clear_previous: bool = True,
        show_nodes: bool = True,
        gradient: bool = False,
        show_robots: bool = True,
        pose_interval: int = 10,
        sleep_interval: float = 0.05,
        interp_step: float = 0.05,
        cleanup_bodies: bool = True,
        save_curve: bool = True,
        save_file: str = "traj_001.csv",
    ) -> None:
        """
        渲染路径（带插值、渐变、半透明机械臂显示）

        :param interp_step: 插值步长，越小轨迹越平滑
        :param cleanup_bodies: 是否在渲染结束后删除临时加载的 URDF（推荐 True，避免资源泄露）
        """
        if not path or len(path) < 2:
            print("[LicheEnv] render_path(): path too short, skipping.")
            return

        # === 路径插值 ===
        path = interpolate_path(path, step_size=interp_step)

        if clear_previous:
            try:
                p.removeAllUserDebugItems()
            except Exception:
                pass

        pts = [self.get_end_effector_pos(q) for q in path]
        n = len(pts)
        temp_body_ids: List[int] = []

        for i in range(n - 1):
            if gradient:
                t = i / max(n - 2, 1)
                c = (1 - t, t, 0)  # 红→绿渐变
            else:
                c = color

            p.addUserDebugLine(
                pts[i],
                pts[i + 1],
                lineColorRGB=c,
                lineWidth=line_width,
                lifeTime=life_time,
            )

            # 半透明机械臂展示
            if show_robots and (i % pose_interval == 0):
                new_robot = p.loadURDF(
                    self.arm_file,
                    [0, 0, 0],
                    [0, 0, 0, 1],
                    useFixedBase=True,
                    flags=p.URDF_IGNORE_COLLISION_SHAPES,
                )
                temp_body_ids.append(new_robot)
                self.set_config(path[i], new_robot)
                for data in p.getVisualShapeData(new_robot):
                    rgba = list(data[-1])
                    rgba[-1] = 0.5
                    p.changeVisualShape(new_robot, data[1], rgbaColor=rgba)
        
        # === 保存轨迹到 CSV ===
        if save_curve and save_file:
            import csv
            import os

            os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)

            # path: (T, dof)
            dof = len(path[0])
            header = ["t"] + [f"q{i}" for i in range(dof)] + ["ee_x", "ee_y", "ee_z"]

            with open(save_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for t, q in enumerate(path):
                    ee = self.get_end_effector_pos(q)  # (x,y,z)
                    w.writerow([t, *list(q), *list(ee)])

            print(f"[LicheEnv] Trajectory saved to: {save_file}")

            if self.GUI:
                time.sleep(sleep_interval)

        if show_nodes:
            for pos in pts[::pose_interval]:
                sphere_id = p.loadURDF(
                    "sphere2red.urdf",
                    pos,
                    globalScaling=0.04,
                    flags=p.URDF_IGNORE_COLLISION_SHAPES,
                )
                temp_body_ids.append(sphere_id)

        # 默认把临时生成的 URDF 清理掉，避免 GUI 场景堆积过多物体
        if cleanup_bodies:
            for bid in temp_body_ids:
                try:
                    p.removeBody(bid)
                except Exception:
                    pass
