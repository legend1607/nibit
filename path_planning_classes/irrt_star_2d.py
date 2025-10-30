import math
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from path_planning_classes.rrt_star_2d import RRTStar2D

class IRRTStar2D(RRTStar2D):
    def __init__(self, start, goal, step_len, search_radius, iter_max, env, clearance):
        super().__init__(start, goal, step_len, search_radius, iter_max, env, clearance)
        # store solution indices
        self.path_solutions = []
        # safe bounds handling: env.bound can be [xmin, xmax, ymin, ymax] or [[xmin,xmax],[ymin,ymax]]
        bound_arr = np.array(self.env.bound)
        if bound_arr.ndim == 1 and bound_arr.size == 4:
            self.bounds = bound_arr.reshape((2, 2)).T  # [[xmin, xmax], [ymin, ymax]] -> transpose -> shape (2,2)
        elif bound_arr.ndim == 2 and bound_arr.shape == (2, 2):
            self.bounds = bound_arr.T
        else:
            # fall back: try to coerce to 2x2
            try:
                self.bounds = bound_arr.reshape((2, 2)).T
            except Exception as e:
                raise ValueError("env.bound must be 4-element or 2x2 array-like") from e
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]

    # =================== 动态扩容 ===================
    def _ensure_capacity(self, required_index):
        # ensure self.vertices and self.vertex_parents exist and have sensible initial capacity
        if getattr(self, "vertices", None) is None or getattr(self, "vertex_parents", None) is None:
            # create minimal initial capacity
            init_cap = max(required_index + 1, 16)
            self.vertices = np.empty((init_cap,), dtype=object)
            self.vertex_parents = np.full((init_cap,), -1, dtype=int)
            return

        old_capacity = self.vertices.shape[0]
        if required_index < old_capacity:
            return
        new_capacity = max(old_capacity * 2, required_index + 1)
        # debug print
        print(f"[IRRT*] 扩容 vertices: {old_capacity} -> {new_capacity}")

        new_vertices = np.empty((new_capacity,), dtype=object)
        new_vertices[:old_capacity] = self.vertices
        self.vertices = new_vertices

        new_parents = np.full((new_capacity,), -1, dtype=int)
        new_parents[:old_capacity] = self.vertex_parents
        self.vertex_parents = new_parents

    # =================== 初始化 ===================
    def init(self):
        cMin, theta = self.get_distance_and_angle(self.start, self.goal)
        # Rotation (3x3) from ellipse frame to world frame
        C = self.RotationToWorldFrame(self.start, self.goal, cMin)
        x_center = np.zeros((3, 1))
        x_center[:2, 0] = (self.start + self.goal) / 2.0
        return theta, cMin, x_center, C

    # =================== 规划入口 ===================
    def planning(self, visualize=False):
        theta, start_goal_dist, x_center, C = self.init()
        c_best = np.inf

        if visualize:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
            ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
            ax.set_title("IRRT* Path Planning")
            ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
            ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')
            for rx, ry, rw, rh in getattr(self.env, "rect_obstacles", []):
                ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='black', alpha=0.5))
            for cx, cy, r in getattr(self.env, "circle_obstacles", []):
                ax.add_patch(plt.Circle((cx, cy), r, color='black', alpha=0.5))
            ax.legend()

        for k in range(self.iter_max):
            if k % 100 == 0:
                print(f"[IRRT*] iter {k}/{self.iter_max}")
            if self.path_solutions:
                c_best, x_best = self.find_best_path_solution()

            node_rand = self.generate_random_node(c_best, start_goal_dist, x_center, C)
            node_nearest, node_nearest_index = self.nearest_neighbor(node_rand)
            node_new = self.new_state(node_nearest, node_rand)

            # proceed only if new edge is collision-free
            if self.is_collision_free(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self._ensure_capacity(node_new_index)
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if neighbor_indices is not None and neighbor_indices.size > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                # use a consistent eps or the parent method default
                if self.in_goal_region(node_new, eps=5.0):
                    self.path_solutions.append(node_new_index)

            # visualization (redraw every 10 iters)
            if visualize and (k % 10 == 0 or k == self.iter_max - 1):
                ax.clear()
                ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
                ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"IRRT* Iter {k}/{self.iter_max}")

                # draw obstacles
                for rx, ry, rw, rh in getattr(self.env, "rect_obstacles", []):
                    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='gray', alpha=1.0))
                for cx, cy, r in getattr(self.env, "circle_obstacles", []):
                    ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=1.0))

                # draw start / goal
                ax.plot(self.start[0], self.start[1], 'go', markersize=8)
                ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8)

                # draw tree edges
                for idx in range(self.num_vertices):
                    parent_idx = self.vertex_parents[idx]
                    if parent_idx != -1:
                        p1 = self.vertices[idx]
                        p2 = self.vertices[parent_idx]
                        if p1 is None or p2 is None:
                            continue
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], alpha=0.5)

                # draw informed ellipse if available
                if self.path_solutions and c_best < np.inf:
                    radicand = max(0.0, c_best**2 - start_goal_dist**2)
                    a = c_best / 2.0
                    b = math.sqrt(radicand) / 2.0
                    theta_ellipse = math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])
                    center_x, center_y = (self.start + self.goal) / 2.0
                    t = np.linspace(0, 2 * np.pi, 200)
                    ellipse_x = a * np.cos(t)
                    ellipse_y = b * np.sin(t)
                    R = np.array([[math.cos(theta_ellipse), -math.sin(theta_ellipse)],
                                  [math.sin(theta_ellipse), math.cos(theta_ellipse)]])
                    ellipse_points = R @ np.vstack((ellipse_x, ellipse_y))
                    ellipse_x, ellipse_y = ellipse_points[0, :] + center_x, ellipse_points[1, :] + center_y
                    ax.plot(ellipse_x, ellipse_y, '--', linewidth=1.2, label='Informed Region')

                # draw best path so far
                if self.path_solutions:
                    _, best_idx = self.find_best_path_solution()
                    path = self.extract_path(best_idx)
                    if len(path) > 0:
                        path_points = np.array(path)
                        ax.plot(path_points[:, 0], path_points[:, 1], '-r', linewidth=2, label='Best Path')

                # add common legend once
                ax.plot([], [], 'go', label='Start')
                ax.plot([], [], 'ro', label='Goal')
                ax.legend()
                plt.pause(0.001)
                plt.draw()

        # finalize path
        if self.path_solutions:
            _, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []

    # =================== 寻找最优路径 ===================
    def find_best_path_solution(self):
        # guard: remove invalid indices
        valid_solutions = [idx for idx in self.path_solutions if 0 <= idx < self.num_vertices and self.vertices[idx] is not None]
        if not valid_solutions:
            raise ValueError("No valid path solutions present")
        path_costs = [self.cost(idx) + self.Line(self.vertices[idx], self.goal) for idx in valid_solutions]
        best_local = int(np.argmin(path_costs))
        return path_costs[best_local], valid_solutions[best_local]

    # =================== 随机节点生成 ===================
    def generate_random_node(self, c_max, c_min, x_center, C):
        if c_max < np.inf:
            return self.SampleInformedSubset(c_max, c_min, x_center, C)
        else:
            return self.sample_free()

    def SampleInformedSubset(self, c_max, c_min, x_center, C, max_attempts=500):
        # numeric stability for radius calculation
        radicand = max(0.0, float(c_max)**2 - float(c_min)**2)
        r = [c_max / 2.0, math.sqrt(radicand) / 2.0, math.sqrt(radicand) / 2.0]
        L = np.diag(r)
        attempt = 0
        while attempt < max_attempts:
            x_ball = self.SampleUnitBall()   # 3x1
            node_rand3 = C @ L @ x_ball + x_center  # 3x1
            node_rand = node_rand3[:2, 0]
            # robust check against environment method
            try:
                if self.env._point_in_free_space(tuple(node_rand)):
                    return node_rand
            except Exception:
                # try array-style call
                if self.env._point_in_free_space(node_rand):
                    return node_rand
            attempt += 1
        # fallback to uniform sampling in free space
        return self.sample_free()

    @staticmethod
    def SampleUnitBall():
        # sample uniformly in unit disk (radius distribution sqrt(u))
        u = random.random()
        r = math.sqrt(u)
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return np.array([[x], [y], [0.0]])

    @staticmethod
    def RotationToWorldFrame(start, goal, L):
        if L <= 0:
            L = 1e-9
        a1 = np.zeros((3, 1))
        a1[:2, 0] = (goal - start) / L
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M)
        # ensure rotation has proper determinant
        det_term = np.linalg.det(U) * np.linalg.det(V_T.T)
        C = U @ np.diag([1.0, 1.0, det_term]) @ V_T
        return C

    # =================== planning_random (two-phase) ===================
    def planning_random(self, iter_after_initial):
        path_len_list = []
        time_list = []
        theta, start_goal_dist, x_center, C = self.init()
        c_best = np.inf

        # Phase 1: find any initial path
        for k in range(self.iter_max):
            t0 = time.time()
            if self.path_solutions:
                c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)

            node_rand = self.generate_random_node(c_best, start_goal_dist, x_center, C)
            node_nearest, node_nearest_index = self.nearest_neighbor(node_rand)
            node_new = self.new_state(node_nearest, node_rand)

            if self.is_collision_free(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self._ensure_capacity(node_new_index)
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if neighbor_indices is not None and neighbor_indices.size > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                if self.in_goal_region(node_new):
                    self.path_solutions.append(node_new_index)

            time_list.append(time.time() - t0)

            if self.path_solutions:
                break

        # Phase 2: optimize for iter_after_initial iterations
        for k in range(iter_after_initial):
            t0 = time.time()
            if not self.path_solutions:
                time_list.append(time.time() - t0)
                path_len_list.append(np.inf)
                continue

            c_best, x_best = self.find_best_path_solution()
            path_len_list.append(c_best)

            node_rand = self.generate_random_node(c_best, start_goal_dist, x_center, C)
            node_nearest, node_nearest_index = self.nearest_neighbor(node_rand)
            node_new = self.new_state(node_nearest, node_rand)

            if self.is_collision_free(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    node_new_index = node_nearest_index
                    curr_node_new_cost = self.cost(node_nearest_index)
                else:
                    node_new_index = self.num_vertices
                    self._ensure_capacity(node_new_index)
                    self.vertices[node_new_index] = node_new
                    self.vertex_parents[node_new_index] = node_nearest_index
                    self.num_vertices += 1
                    curr_node_new_cost = self.cost(node_nearest_index) + self.Line(node_nearest, node_new)

                neighbor_indices = self.find_near_neighbors(node_new, node_new_index)
                if neighbor_indices is not None and neighbor_indices.size > 0:
                    self.choose_parent(node_new, neighbor_indices, node_new_index, curr_node_new_cost)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                if self.in_goal_region(node_new):
                    self.path_solutions.append(node_new_index)

            time_list.append(time.time() - t0)

        # finalize best path if any
        if self.path_solutions:
            _, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []

        return path_len_list, time_list

# =================== 工厂函数 ===================
def get_path_planner(args, problem, neural_wrapper=None):
    return IRRTStar2D(
        problem['start'],
        problem['goal'],
        args.step_len,
        problem.get('search_radius', 5.0),
        args.iter_max,
        problem['env'],
        args.clearance,
    )
