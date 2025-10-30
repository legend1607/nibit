import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import List, Optional, Sequence, Tuple, Dict, Set, Any, Union
import numpy as np
import pybullet as p
import time
from typing import Sequence

def interpolate_path(path: Sequence[Sequence[float]], step_size: float = 0.1):
    """
    对路径进行线性插值，使关节或坐标变化平滑
    :param path: 原始路径 (list of configs)
    :param step_size: 相邻节点最大欧式间距
    :return: 插值后的平滑路径
    """
    if len(path) < 2:
        return path

    interpolated = [np.array(path[0], dtype=float)]
    for i in range(len(path) - 1):
        q1, q2 = np.array(path[i]), np.array(path[i + 1])
        dist = np.linalg.norm(q2 - q1)
        n_steps = max(int(dist / step_size), 1)
        for j in range(1, n_steps + 1):
            q_interp = q1 + (q2 - q1) * (j / n_steps)
            interpolated.append(q_interp)
    return interpolated


class LicheEnv:
    """
    LicheEnv - 用于基于 PyBullet 的 LICHE关节空间规划与碰撞检测环境。
    接口兼容 Random2DEnv（step / sample / distance / interpolate 等）。
    中文注释与详细 docstring 保留，便于直接阅读与维护。

    主要特性:
    - 支持随机障碍物（方块、球体），并可清理
    - 更快的碰撞检测路径：当 self.obstacles 非空时，优先使用 getClosestPoints(kuka, obs, distance=0.0)
    - 可忽略某些 body（ignored_body_ids），避免平面等造成误报
    - 更安全的 set_config / get_config（长度检查、范围裁剪）
    - get_end_effector_pos 返回 numpy.ndarray
    - 支持 with 上下文管理：with LicheEnv(...) as env:
    - 提供碰撞统计 get_collision_stats()
    - 修复 _edge_fp 对步数与端点的处理（RRT_EPS 逻辑）
    """

    EPS: float = 0.05
    RRT_EPS: float = 0.5

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
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=30, cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # 加载平面与机械臂
        # self.plane_id: int = p.loadURDF("plane.urdf")
        # 默认忽略平面造成的接触
        # self.ignored_body_ids.add(self.plane_id)

        self.arm_id: int = p.loadURDF(self.arm_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

        # 获取机械臂关节维度与范围
        self.config_dim: int = p.getNumJoints(self.arm_id)
        # pose_range shape: (config_dim, 2)
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
        self.pose_range: np.ndarray = np.array(pr, dtype=float)
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
            # 注意：disconnect 在 DIRECT/GUI 都适用
            p.disconnect()
        except Exception:
            pass

    # -------------------------
    # 障碍物管理
    # -------------------------
    def add_box_obstacle(self, half_extents: Sequence[float], base_pos: Sequence[float], rgba: Sequence[float] = [0, 0, 0, 1]) -> int:
        """添加静态方块障碍，返回 body id 并记录到 self.obstacles"""
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id, basePosition=base_pos)
        self.obstacles.append(body_id)
        return body_id

    def add_sphere_obstacle(self, radius: float, base_pos: Sequence[float], rgba: Sequence[float] = [0, 0, 1, 1]) -> int:
        """添加静态球体障碍，返回 body id 并记录到 self.obstacles"""
        collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id, basePosition=base_pos)
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
        # 也从 ignored 中移除（若存在）
        if body_id in self.ignored_body_ids:
            self.ignored_body_ids.discard(body_id)

    def clear_obstacles(self) -> None:
        """删除场景中所有已记录的障碍并清空 self.obstacles（保留 plane 与 kuka）"""
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
        # 裁剪范围
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
            # state[0] 是位置
            vals.append(float(state[0]))
        return np.array(vals, dtype=float)

    # -------------------------
    # Random2DEnv 兼容接口
    # -------------------------
    def get_problem(self) -> Dict[str, Any]:
        """返回当前问题信息（供外部算法使用）"""
        return dict(
            start=self.start,
            goal=self.goal,
            bound=self.bound,
            obstacles=list(self.obstacles)
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

    def step(self, state: Sequence[float], action: Optional[Sequence[float]] = None,
             new_state: Optional[Sequence[float]] = None, check_collision: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], bool, bool]:
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

        # 限制范围
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

        # 类型与长度检查
        s_arr = np.asarray(state, dtype=float).flatten()
        if s_arr.size != self.config_dim:
            raise ValueError(f"[LicheEnv] 状态维度不匹配: got {s_arr.size}, expected {self.config_dim}")

        # 将关节直接设置到该状态（resetJointState）
        for j in range(self.config_dim):
            p.resetJointState(self.arm_id, j, float(s_arr[j]))

        # 开始计数
        self.collision_check_count += 1

        # 如果有显式障碍列表，优先使用 getClosestPoints（每个障碍体一次查询，distance=0.0）
        collision_found = False
        if len(self.obstacles) > 0:
            # 对每个障碍体检查最近点，若有最近点距离<=0 则视为碰撞（返回非自由）
            for obs_id in self.obstacles:
                # skip ignored obstacles if user flagged them
                if obs_id in self.ignored_body_ids:
                    continue
                pts = p.getClosestPoints(self.arm_id, obs_id, distance=0.0)
                if pts:
                    # 若 pts 非空，则存在接触或重叠 -> 碰撞
                    collision_found = True
                    break
        else:
            # 回退到 getContactPoints（过滤 ignored_body_ids）
            contacts = p.getContactPoints(bodyA=self.arm_id)
            # contact tuple 结构中通常 bodyUniqueIdA 在 index 1，bodyUniqueIdB 在 index 2
            # 我们需要检查是否存在与非忽略体的接触
            for c in contacts:
                try:
                    bodyA = int(c[1])
                    bodyB = int(c[2])
                except Exception:
                    # 若解析失败则保守处理为存在接触
                    collision_found = True
                    break
                # 如果两者都在 ignored，则跳过；否则视为碰撞
                if (bodyA in self.ignored_body_ids) and (bodyB in self.ignored_body_ids):
                    continue
                # 如果与 kuka 自身接触（bodyA==kuka或者bodyB==kuka）且另一个体不是 ignored -> 碰撞
                other = bodyB if bodyA == self.arm_id else bodyA if bodyB == self.arm_id else None
                if other is None:
                    # 未能确定另一方，保守认为碰撞
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
        - 先检查端点 a, b
        - 对线段内部按 RRT_EPS 步长进行插值检查（使用 ceil 保证覆盖）
        - 返回 True 表示整条线段无碰撞
        """
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)

        # 端点检查
        if not (self._point_in_free_space(a_arr) and self._point_in_free_space(b_arr)):
            return False

        d = self.distance(a_arr, b_arr)
        # 计算需要的步数：确保每段长度不超过 RRT_EPS
        # 若 d == 0 则没有内部点
        if d <= 0.0:
            return True

        steps = max(int(np.ceil(d / self.RRT_EPS)), 1)
        # 我们已经检查了端点，所以只需检查 1..steps-1 的内部点
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
    def sample_empty_points(self) -> np.ndarray:
        """循环采样直到得到一个无碰撞的配置"""

        for _ in range(10000):  # 防止死循环，最多尝试 10000 次
            s = self.uniform_sample()
            if self._point_in_free_space(s):
                return s

    def set_random_init_goal(self) -> Dict[str, Any]:
        """随机生成有效的 start / goal（不会太接近）"""
        self.start = self.sample_empty_points()
        self.goal = self.sample_empty_points()
        while self.distance(self.start, self.goal) < 0.1:
            self.goal = self.sample_empty_points()
        return self.get_problem()

    def in_goal_region(self, state: Sequence[float], eps: float = EPS) -> bool:
        """判断是否到达目标（并且当前状态也是无碰撞）"""
        if self.goal is None:
            return False
        return (self.distance(state, self.goal) < eps) and self._point_in_free_space(state)

    # -------------------------
    # 可视化 / 其他工具
    # -------------------------
    def get_end_effector_pos(self, config: Sequence[float]) -> np.ndarray:
        """
        给定关节配置，返回末端坐标（xyz），以 numpy.ndarray 返回
        注意：此函数会把关节状态 reset 到给定配置（不会做动力学推进）
        """
        self.set_config(config)
        # getLinkState 返回元组，位置在索引 0
        pos = p.getLinkState(self.arm_id, self.end_effector_index)[0]
        return np.array(pos, dtype=float)

    # -------------------------
    # 状态统计
    # -------------------------
    def get_collision_stats(self) -> Dict[str, float]:
        """返回碰撞检测统计信息"""
        avg_time = (self.collision_time / self.collision_check_count) if self.collision_check_count > 0 else 0.0
        return dict(
            collision_check_count=self.collision_check_count,
            total_collision_time=self.collision_time,
            avg_collision_check_time=avg_time
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
        sleep_interval: float = 0.02,
        interp_step: float = 0.1
    ) -> None:
        """
        渲染路径（带插值、渐变、半透明机械臂显示）
        -------------------------------------------------
        :param interp_step: 插值步长，越小轨迹越平滑
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

        for i in range(n - 1):
            if gradient:
                t = i / (n - 2)
                c = (1 - t, t, 0)  # 红→绿渐变
            else:
                c = color

            p.addUserDebugLine(
                pts[i],
                pts[i + 1],
                lineColorRGB=c,
                lineWidth=line_width,
                lifeTime=life_time
            )

            # 半透明机械臂展示
            if show_robots and (i % pose_interval == 0):
                new_robot = p.loadURDF(
                    self.arm_file,
                    [0, 0, 0],
                    [0, 0, 0, 1],
                    useFixedBase=True,
                    flags=p.URDF_IGNORE_COLLISION_SHAPES
                )
                self.set_config(path[i], new_robot)
                for data in p.getVisualShapeData(new_robot):
                    rgba = list(data[-1])
                    rgba[-1] = 0.5
                    p.changeVisualShape(new_robot, data[1], rgbaColor=rgba)

            if self.GUI:
                time.sleep(sleep_interval)

        if show_nodes:
            for pos in pts[::pose_interval]:
                p.loadURDF(
                    "sphere2red.urdf",
                    pos,
                    globalScaling=0.04,
                    flags=p.URDF_IGNORE_COLLISION_SHAPES
                )
