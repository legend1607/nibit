import math
import time
from matplotlib import pyplot as plt
import numpy as np
from path_planning_classes.rrt_base_2d import RRTBase2D

class RRTStar2D(RRTBase2D):
    def __init__(self, start, goal, step_len, search_radius, iter_max, env, clearance):
        super().__init__(
            start,
            goal,
            step_len,
            search_radius,
            iter_max,
            env,
            clearance,
        )

    # =================== 核心规划 ===================
    def planning(self, visualize=False, eps=0.5):
        if visualize:
            plt.ion()  # 开启交互式绘图
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
            ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
            ax.set_title("BIT* Path Planning")
            start_point, goal_point = np.array(self.start), np.array(self.goal)
            ax.plot(start_point[0], start_point[1], 'go', markersize=8, label='Start')
            ax.plot(goal_point[0], goal_point[1], 'ro', markersize=8, label='Goal')
            for rx, ry, rw, rh in self.env.rect_obstacles:
                rect = plt.Rectangle((rx, ry), rw, rh, color='black', alpha=0.5)
                ax.add_patch(rect)

            # 绘制圆形障碍
            for cx, cy, r in self.env.circle_obstacles:
                circle = plt.Circle((cx, cy), r, color='black', alpha=0.5)
                ax.add_patch(circle)
            ax.legend()

        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
            node_nearest, node_nearest_index = self.nearest_neighbor(node_rand)
            node_new = self.new_state(node_nearest, node_rand)

            if self.is_collision_free(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices) > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)
            if visualize and k % 10 == 0:
                ax.clear()
                ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"BIT* Iteration {k}/{self.iter_max}")

                # 绘制障碍
                for rx, ry, rw, rh in self.env.rect_obstacles:
                    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='gray', alpha=1.0))
                for cx, cy, r in self.env.circle_obstacles:
                    ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=1.0))
                # 绘制起点和终点
                ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
                ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')

                # 绘制树的连线
                for idx, parent_idx in enumerate(self.vertex_parents[:self.num_vertices]):
                    if parent_idx != -1:
                        p1 = self.vertices[idx]
                        p2 = self.vertices[parent_idx]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', alpha=0.5)

                # 绘制最优路径
                if x_best is not None:
                    path = self.extract_path(x_best)
                    path_points = np.array(path)
                    ax.plot(path_points[:, 0], path_points[:, 1], '-r', linewidth=2, label='Best Path')


        goal_parent_index = self.search_goal_parent(eps)

        self.path = self.extract_path(goal_parent_index)
        return self.path

    # =================== 节点生成 ===================
    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_len, dist)
        node_new = node_start + dist * np.array([math.cos(theta), math.sin(theta)])
        return node_new

    def generate_random_node(self):
        return self.sample_free()

    # =================== 碰撞和邻居 ===================
    def is_collision_free(self, start, end):
        return self.env._edge_fp(start, end, step_size=self.step_len)

    def find_near_neighbors(self, node_new, node_new_index=None):
        r = min(self.search_radius * math.sqrt(math.log(self.num_vertices)/self.num_vertices), self.step_len)
        vec_to_node_new = node_new - self.vertices[:self.num_vertices]
        dist_to_node_new = np.hypot(vec_to_node_new[:,0], vec_to_node_new[:,1])
        indices_vertex_within_r = np.where(dist_to_node_new <= r)[0]

        neighbor_indices = []
        for vertex_index in indices_vertex_within_r:
            if node_new_index is not None and vertex_index == node_new_index:
                continue
            if self.is_collision_free(node_new, self.vertices[vertex_index]):
                neighbor_indices.append(vertex_index)
        return np.array(neighbor_indices)

    # =================== 选择父节点 ===================
    def choose_parent(self, node_new, neighbor_indices, node_new_index, curr_node_new_cost):
        vec_neighbors_to_new = node_new - self.vertices[:self.num_vertices][neighbor_indices]
        dist_neighbors_to_new = np.hypot(vec_neighbors_to_new[:,0], vec_neighbors_to_new[:,1])
        neighbor_costs = np.array([self.cost(idx) for idx in neighbor_indices])
        node_new_cost_candidates = neighbor_costs + dist_neighbors_to_new
        best_idx = np.argmin(node_new_cost_candidates)
        if node_new_cost_candidates[best_idx] < curr_node_new_cost:
            self.vertex_parents[node_new_index] = neighbor_indices[best_idx]

    # =================== 重连 ===================
    def rewire(self, node_new, neighbor_indices, node_new_index):
        node_new_cost = self.cost(node_new_index)
        vec_new_to_neighbors = self.vertices[:self.num_vertices][neighbor_indices] - node_new
        dist_new_to_neighbors = np.hypot(vec_new_to_neighbors[:,0], vec_new_to_neighbors[:,1])
        for i, neighbor_index in enumerate(neighbor_indices):
            if self.cost(neighbor_index) > node_new_cost + dist_new_to_neighbors[i]:
                self.vertex_parents[neighbor_index] = node_new_index

    # =================== 搜索目标父节点 ===================
    def search_goal_parent(self, eps=0.5):
        vec_to_goal = self.goal - self.vertices[:self.num_vertices]
        dist_to_goal = np.hypot(vec_to_goal[:,0], vec_to_goal[:,1])
        indices_within_step = np.where(dist_to_goal <= self.step_len)[0]
        if len(indices_within_step) == 0:
            return None

        costs = []
        for idx in indices_within_step:
            if self.is_collision_free(self.vertices[idx], self.goal):
                costs.append(self.cost(idx) + dist_to_goal[idx])
            else:
                costs.append(np.inf)
        best_idx = indices_within_step[np.argmin(costs)]
        if costs[np.argmin(costs)] == np.inf:
            return None
        return best_idx

    # =================== planning_random ===================
    def planning_random(self, iter_after_initial):
        """
        Two-stage RRT*: first find initial path, then refine with additional iterations.
        Supports dynamic expansion of vertices array.
        """
        path_len_list = []
        time_list = []

        def ensure_capacity(index_needed):
            """动态扩容以避免越界"""
            if index_needed < len(self.vertices):
                return
            old_size = len(self.vertices)
            new_size = max(old_size * 2, index_needed + 1)
            print(f"[Info] Expanding vertices from {old_size} to {new_size}")

            # 扩容 vertices
            dim = self.vertices.shape[1]
            new_vertices = np.zeros((new_size, dim), dtype=self.vertices.dtype)
            new_vertices[:old_size] = self.vertices
            self.vertices = new_vertices

            # 扩容 vertex_parents
            new_parents = np.full((new_size,), -1, dtype=self.vertex_parents.dtype)
            new_parents[:old_size] = self.vertex_parents
            self.vertex_parents = new_parents

        # ===================== 第一阶段：寻找初始路径 =====================
        for k in range(self.iter_max):
            t0 = time.time()
            node_rand = self.generate_random_node()
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)

            if self.is_collision_free(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    ensure_capacity(node_new_index)
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices) > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)

            goal_parent_index = self.search_goal_parent()
            current_path = [] if goal_parent_index is None else self.extract_path(goal_parent_index)
            current_path_len = self.get_path_len(current_path)
            path_len_list.append(current_path_len)
            time_list.append(time.time() - t0)

            if current_path_len < np.inf:
                print(f"[Info] Initial path found at iteration {k+1}, length: {current_path_len:.2f}")
                break

        # ===================== 第二阶段：路径优化 =====================
        for k in range(iter_after_initial):
            t0 = time.time()
            node_rand = self.generate_random_node()
            node_nearest, node_nearest_index = self.nearest_neighbor(self.vertices[:self.num_vertices], node_rand)
            node_new = self.new_state(node_nearest, node_rand)

            if self.is_collision_free(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    ensure_capacity(node_new_index)
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if len(neighbor_indices) > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)

            goal_parent_index = self.search_goal_parent()
            current_path = [] if goal_parent_index is None else self.extract_path(goal_parent_index)
            current_path_len = self.get_path_len(current_path)
            path_len_list.append(current_path_len)
            time_list.append(time.time() - t0)

        return path_len_list, time_list


# =================== 工厂函数 ===================
def get_path_planner(args, problem, neural_wrapper=None):
    """
    Factory function to create a RRT* planner for Random2DEnv
    """
    return RRTStar2D(
        start=problem['start'],
        goal=problem['goal'],
        step_len=args.step_len,
        search_radius=problem.get('search_radius', 5.0),
        iter_max=args.iter_max,
        env=problem['env'],
        clearance=args.clearance
    )
