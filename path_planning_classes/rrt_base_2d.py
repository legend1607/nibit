import math
import numpy as np

class RRTBase2D:
    """
    基于RRT的2D路径规划器，适配Random2DEnv
    """
    def __init__(
        self,
        start,
        goal,
        step_len,
        search_radius,
        iter_max,
        env,  # Random2DEnv实例
        clearance,
    ):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.step_len = step_len
        self.search_radius = search_radius
        self.iter_max = iter_max

        # 存储节点和父节点索引
        self.vertices = np.zeros((1 + iter_max, 2))
        self.vertex_parents = np.full(1 + iter_max, -1, dtype=int)  # 根节点父节点为 -1
        self.vertices[0] = self.start
        self.num_vertices = 1

        self.env = env
        self.clearance = clearance

        self.x_range = [self.env.bound[0][0], self.env.bound[1][0]]
        self.y_range = [self.env.bound[0][1], self.env.bound[1][1]]

    # ================= 核心采样 =================
    def sample_free(self):
        """从环境中采样空闲点"""
        return self.env.sample_empty_points()

    def cost(self, vertex_index):
        cost = 0.
        while vertex_index != 0:
            vertex_parent_index = self.vertex_parents[vertex_index]
            dx, dy = self.vertices[:self.num_vertices][vertex_index] - self.vertices[:self.num_vertices][vertex_parent_index]
            cost += math.hypot(dx, dy)
            vertex_index = vertex_parent_index
        return cost
    # ================= 辅助函数 =================
    @staticmethod
    def get_distance_and_angle(start, end):
        dx, dy = end - start
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def nearest_neighbor(self, n):
        """找到最近邻节点"""
        distances = np.linalg.norm(self.vertices[:self.num_vertices] - n, axis=1)
        nearest_index = np.argmin(distances)
        return self.vertices[nearest_index], nearest_index

    def is_collision_free(self, start, end):
        """检查从 start 到 end 是否碰撞"""
        return self.env._edge_fp(start, end, step_size=self.step_len)

    def in_goal_region(self, node, eps=0.5):
        return self.env.in_goal_region(node, eps=eps)

    # ================= 树扩展 =================
    def steer(self, from_node, to_node):
        dist, angle = self.get_distance_and_angle(from_node, to_node)
        if dist <= self.step_len:
            return to_node
        else:
            new_x = from_node[0] + self.step_len * math.cos(angle)
            new_y = from_node[1] + self.step_len * math.sin(angle)
            return np.array([new_x, new_y])

    # ================= 路径提取 =================
    def extract_path(self, goal_index):
        path = [self.vertices[goal_index]]
        index = self.vertex_parents[goal_index]
        while index != -1:
            path.append(self.vertices[index])
            index = self.vertex_parents[index]
        path.reverse()
        return np.array(path)

    # ================= RRT算法 =================
    def planning(self, goal_sample_rate=0.1, eps=0.5):
        for i in range(self.iter_max):
            # 1. 采样
            if np.random.rand() < goal_sample_rate:
                sampled_point = self.goal
            else:
                sampled_point = self.sample_free()

            # 2. 找到最近节点
            nearest_node, nearest_index = self.nearest_neighbor(sampled_point)

            # 3. 向采样点扩展
            new_node = self.steer(nearest_node, sampled_point)

            # 4. 检查碰撞
            if self.is_collision_free(nearest_node, new_node):
                self.vertices[self.num_vertices] = new_node
                self.vertex_parents[self.num_vertices] = nearest_index
                current_index = self.num_vertices
                self.num_vertices += 1

                # 5. 检查是否到达目标区域
                if self.in_goal_region(new_node, eps=eps):
                    return self.extract_path(current_index)

        # 如果没有找到路径，返回None
        return None

    # ================= 路径长度 =================
    @staticmethod
    def get_path_len(path):
        if path is None or len(path) == 0:
            return np.inf
        return np.linalg.norm(path[1:] - path[:-1], axis=1).sum()

    def check_success(self, path):
        if path is None or len(path) == 0:
            return False
        return np.allclose(path[0], self.start) and np.allclose(path[-1], self.goal)


    @staticmethod
    def Line(start, goal):
        dx, dy = goal - start
        return math.hypot(dx, dy)