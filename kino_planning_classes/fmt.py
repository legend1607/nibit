import math
import random
from collections import deque
from time import time

# --------------------- Node 和 Edge ---------------------
class Node:
    def __init__(self, state):
        self.state = state  # [x, y, vx, vy]
        self.cost = 0
        self.parent = None
        self.children = set()
        self.N_out = None
        self.N_in = None

class Edge:
    def __init__(self, src, tgt):
        self.source = src
        self.target = tgt

# --------------------- KinoFMT* 类 ---------------------
class KinoFMTStar:
    def __init__(self, env, Jth, n_samples, svm_classifier=None):
        """
        env: Random2DKinodynamicEnv 实例
        Jth: 阈值代价
        n_samples: 采样节点数量
        svm_classifier: 可达性预测函数，输入 state1, state2 返回 True/False
        """
        self.env = env
        self.Jth = Jth
        self.n_samples = n_samples
        self.svm = svm_classifier
        self.V = set()
        self.E = set()
        self.H = set()  # frontier
        self.W = set()  # 未访问
        self.COST = {}  # (n1.state, n2.state) -> (J_opt, tau)

    # -------------------- 工具函数 --------------------
    @staticmethod
    def distance(n1, n2):
        x1, y1 = n1.state[0], n1.state[1]
        x2, y2 = n2.state[0], n2.state[1]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def cost(self, n1, n2):
        key = (tuple(n1.state), tuple(n2.state))
        if key in self.COST:
            return self.COST[key][0]
        else:
            J_opt = self.distance(n1, n2)**2  # 简化：距离平方作为代价
            tau = self.distance(n1, n2)       # 最优时间近似
            self.COST[key] = (J_opt, tau)
            return J_opt

    def reachable_set(self, x, forward=True):
        if forward:
            if x.N_out is not None:
                return x.N_out
            x.N_out = set()
            for s in self.V:
                if s == x:
                    continue
                reachable = False
                if self.svm:
                    reachable = self.svm(x.state, s.state)
                else:
                    reachable = self.cost(x, s) < self.Jth
                if reachable:
                    x.N_out.add(s)
            return x.N_out
        else:
            if x.N_in is not None:
                return x.N_in
            x.N_in = set()
            for s in self.V:
                if s == x:
                    continue
                reachable = False
                if self.svm:
                    reachable = self.svm(s.state, x.state)
                else:
                    reachable = self.cost(s, x) < self.Jth
                if reachable:
                    x.N_in.add(s)
            return x.N_in

    def nearest(self, S, node):
        """返回 Ynear 中 cost-to-come 最小节点"""
        min_node = None
        min_cost = float('inf')
        for s in S:
            c = s.cost + self.cost(s, node)
            if c < min_cost:
                min_cost = c
                min_node = s
        return min_node, min_cost

    # -------------------- 核心规划 --------------------

    def planning(self, visual=False, grid_size=20, delay=0.05):
        xinit = Node(self.env.start)
        xinit.cost = 0
        self.V.add(xinit)
        for _ in range(self.n_samples):
            s_state = self.env.sample_empty_points()
            node = Node(s_state)
            node.cost = float('inf')
            self.V.add(node)

        self.H = set([xinit])
        self.W = self.V - set([xinit])
        z = xinit
        goal_node = Node(self.env.goal)
        goal_reached = False

        while not goal_reached and self.H:
            N_zout = self.reachable_set(z, forward=True) & self.W
            for x in N_zout:
                Ynear = self.reachable_set(x, forward=False) & self.H
                if not Ynear:
                    continue
                ymin, min_cost = self.nearest(Ynear, x)
                no_collision = self.env._trajectory_fp(ymin.state, x.state, dt=0.1)
                if no_collision:
                    x.parent = ymin
                    x.cost = min_cost
                    ymin.children.add(x)
                    self.E.add(Edge(ymin, x))
                    self.H.add(x)
                    self.W.remove(x)
                    if self.env.distance(x.state, self.env.goal) < 0.5:
                        goal_node.parent = x
                        goal_node.cost = x.cost + self.cost(x, goal_node)
                        goal_reached = True

            self.H.remove(z)
            if not self.H:
                break
            z = min(self.H, key=lambda n: n.cost)

            if visual:
                self._visual(grid_size)
                time.sleep(delay)

        if goal_reached:
            path = self.get_path(goal_node)
            smooth_path = self.smooth_path(path)
            if visual:
                print("\nFinal trajectory:")
                self._visual(grid_size, path=smooth_path)
            return smooth_path
        else:
            return None

    # -------------------- 动态可视化 --------------------
    def _visual(self, grid_size=20, path=None):
        """简单 ASCII 动态显示 2D 环境"""
        # 创建网格
        grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

        # 障碍物
        for rx, ry, rw, rh in self.env.rect_obstacles:
            x0 = int(rx / self.env.bound[1][0] * grid_size)
            y0 = int(ry / self.env.bound[1][1] * grid_size)
            x1 = min(grid_size-1, int((rx+rw) / self.env.bound[1][0] * grid_size))
            y1 = min(grid_size-1, int((ry+rh) / self.env.bound[1][1] * grid_size))
            for i in range(x0, x1+1):
                for j in range(y0, y1+1):
                    grid[j][i] = '#'

        # 节点
        for node in self.V:
            x, y = node.state[0], node.state[1]
            i = int(x / self.env.bound[1][0] * grid_size)
            j = int(y / self.env.bound[1][1] * grid_size)
            grid[j][i] = 'o'

        # 起点和终点
        sx, sy = self.env.start[0], self.env.start[1]
        gx, gy = self.env.goal[0], self.env.goal[1]
        grid[int(sy / self.env.bound[1][1] * grid_size)][int(sx / self.env.bound[1][0] * grid_size)] = 'S'
        grid[int(gy / self.env.bound[1][1] * grid_size)][int(gx / self.env.bound[1][0] * grid_size)] = 'G'

        # 路径
        if path:
            for state in path:
                x, y = state[0], state[1]
                i = int(x / self.env.bound[1][0] * grid_size)
                j = int(y / self.env.bound[1][1] * grid_size)
                grid[j][i] = '*'

        # 打印网格
        print("\033c", end='')  # 清屏
        for row in reversed(grid):
            print(''.join(row))
    # -------------------- 路径处理 --------------------
    def get_path(self, end_node):
        path = deque()
        node = end_node
        while node:
            path.appendleft(node)
            node = node.parent
        return list(path)

    def smooth_path(self, path, NUM=20):
        """二次多项式平滑"""
        if len(path) < 3:
            return [n.state for n in path]
        smooth_states = []
        for i in range(len(path)-2):
            p0, p1, p2 = path[i].state, path[i+1].state, path[i+2].state
            for t_idx in range(NUM):
                t = t_idx / (NUM-1)
                x = (1-t)**2*p0[0] + 2*(1-t)*t*(p0[0]+p1[0])/2 + t**2*p2[0]
                y = (1-t)**2*p0[1] + 2*(1-t)*t*(p0[1]+p1[1])/2 + t**2*p2[1]
                vx = (1-t)**2*p0[2] + 2*(1-t)*t*(p0[2]+p1[2])/2 + t**2*p2[2]
                vy = (1-t)**2*p0[3] + 2*(1-t)*t*(p0[3]+p1[3])/2 + t**2*p2[3]
                # 碰撞检测
                if not self.env._trajectory_fp([x, y, vx, vy], [x, y, vx, vy], dt=0.05):
                    continue
                smooth_states.append([x, y, vx, vy])
        smooth_states.append(path[-2].state)
        smooth_states.append(path[-1].state)
        return smooth_states

# --------------------- 工厂函数 ---------------------
def get_planner(
    args,
    problem,
    neural_wrapper=None,
):
    """
    创建 KinoFMTStar 规划器实例
    
    args: 包含规划器参数的对象，至少包括：
        - iter_max: 最大迭代次数
        - pc_n_points: 采样点数量
        - Jth: 动力学代价阈值
    problem: dict，至少包含：
        - 'env': Random2DKinodynamicEnv 实例
        - 'x_start': 起点状态 [x, y, vx, vy]
        - 'x_goal': 目标状态 [x, y, vx, vy]
    neural_wrapper: 可选，SVM 或神经网络可达性预测接口
    """
    planner = KinoFMTStar(
        env=problem['env'],
        Jth=getattr(args, "Jth", 5.0),
        n_samples=getattr(args, "pc_n_points", 100),
        svm_classifier=neural_wrapper
    )

    # 设置起点和目标
    planner.env.start = problem["x_start"]
    planner.env.goal = problem["x_goal"]

    return planner
