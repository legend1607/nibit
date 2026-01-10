"""Merged n-D RRT* / Informed RRT* (IRRT*) implementation.

This is an n-dimensional generalization of the uploaded 3D version. It keeps the
same environment API assumptions:

- env._state_fp(x: np.ndarray) -> bool     # True if point is in free space
- env._edge_fp(a: np.ndarray, b: np.ndarray) -> bool  # True if segment is collision-free
- env.collision_check_count (int)

Sampling bounds:

- Preferred: env.bound = [(min0,max0), (min1,max1), ...] for all dimensions.
- Back-compat (3D only): env.x_range, env.y_range, env.z_range.

Factories at the bottom preserve the original entrypoints.
"""

import math
from time import time

import numpy as np

from scipy.spatial import cKDTree  # KD-Tree acceleration

# Optional neural wrapper is passed in by caller (see get_nirrtstar_planner)

def _normalize_bounds(bound, dim: int):
    """
    Normalize env.bound to shape (dim,2).
    Supports:
      - flat (2*dim,)  -> [l0,u0,l1,u1,...]
      - (dim,2)
      - (2,dim)
      - list[(l,u), ...]
    """
    if bound is None:
        raise ValueError("env.bound is None")

    b = np.asarray(bound, dtype=float)

    # flat: [l0,u0,l1,u1,...]
    if b.ndim == 1 and b.size == 2 * dim:
        return b.reshape((2, dim)).T   # 与 BITStar 完全一致

    # (2,dim)
    if b.ndim == 2 and b.shape == (2, dim):
        return b.T

    # (dim,2)
    if b.ndim == 2 and b.shape == (dim, 2):
        return b

    raise ValueError(
        f"Bad env.bound shape {b.shape}; expected flat (2*dim,) or (dim,2). dim={dim}"
    )


class Utils:
    """Collision / validity helpers using the same environment API as BIT*."""

    def __init__(self, env, clearance: float = 0.0):
        self.env = env
        self.clearance = float(clearance)

    @staticmethod
    def _to_np(p):
        return np.array(p, dtype=float)

    def is_collision(self, start, end) -> bool:
        """Return True if the edge (start,end) is in collision."""
        a = self._to_np(start)
        b = self._to_np(end)
        # env._edge_fp returns True when edge is free.
        return not bool(self.env._edge_fp(a, b))

    def is_inside_obs(self, node) -> bool:
        """Return True if point is inside obstacle (i.e., not free)."""
        p = self._to_np(node)
        return not bool(self.env._state_fp(p))

    def is_valid(self, node) -> bool:
        """Validity is delegated to env._state_fp."""
        p = self._to_np(node)
        return bool(self.env._state_fp(p))


class RRTBaseND:
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env,
        clearance,
        path_planner_name,
    use_kdtree: bool = True,
    kdtree_rebuild_interval: int = 64,
    ):
        self.x_start = np.array(x_start, dtype=np.float64)
        self.x_goal = np.array(x_goal, dtype=np.float64)
        if self.x_start.ndim != 1 or self.x_goal.ndim != 1:
            raise ValueError("x_start and x_goal must be 1D state vectors")
        if self.x_start.shape != self.x_goal.shape:
            raise ValueError("x_start and x_goal must have the same dimensionality")

        self.dim = int(self.x_start.shape[0])
        self.step_len = float(step_len)
        self.search_radius = float(search_radius)
        self.iter_max = int(iter_max)

        # Pre-allocate arrays for speed (avoid repeated vstack)
        self.vertices = np.zeros((1 + self.iter_max, self.dim), dtype=np.float64)
        self.vertex_parents = np.zeros(1 + self.iter_max, dtype=int)
        self.vertices[0] = self.x_start  # start parent is itself
        self.num_vertices = 1
        self.path = []

        self.env = env
        self.utils = Utils(env, clearance)
        self.clearance = float(clearance)

        raw_bound = getattr(env, "bound", None)
        self.bound = _normalize_bounds(raw_bound, self.dim)   # shape: (dim,2)


        self.path_planner_name = path_planner_name

        # ---------------- KD-Tree acceleration ----------------
        # We keep a cKDTree built on a *prefix* of vertices for fast queries,
        # and brute-force over the "tail" (recently added vertices not yet in the tree)
        # to preserve correctness without rebuilding every insertion.
        self.use_kdtree = bool(use_kdtree)
        self.kdtree_rebuild_interval = int(kdtree_rebuild_interval)
        self._kdtree = None
        self._kdtree_valid_n = 0  # number of vertices included in _kdtree

    def _rebuild_kdtree(self):
        """Rebuild the KD-Tree over all current vertices."""
        if (not self.use_kdtree) or self.num_vertices <= 0:
            self._kdtree = None
            self._kdtree_valid_n = 0
            return
        self._kdtree = cKDTree(self.vertices[: self.num_vertices])
        self._kdtree_valid_n = int(self.num_vertices)

    def _maybe_rebuild_kdtree(self, force: bool = False):
        if not self.use_kdtree:
            return
        if self.num_vertices <= 0:
            self._kdtree = None
            self._kdtree_valid_n = 0
            return
        if force or (self._kdtree is None) or (self.num_vertices - self._kdtree_valid_n >= self.kdtree_rebuild_interval):
            self._rebuild_kdtree()

    def _nearest_index(self, q: np.ndarray) -> int:
        """Return index of nearest vertex to q (KD-Tree + tail brute-force)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if self.num_vertices == 1 or (not self.use_kdtree):
            # fallback: brute force
            vec = q - self.vertices[: self.num_vertices]
            return int(np.argmin(np.linalg.norm(vec, axis=1)))

        self._maybe_rebuild_kdtree(force=False)

        # Query KD-Tree over prefix
        best_idx = 0
        best_dist = float("inf")
        if self._kdtree is not None and self._kdtree_valid_n > 0:
            d, idx = self._kdtree.query(q, k=1)
            best_idx = int(idx)
            best_dist = float(d)

        # Brute-force tail vertices not yet indexed by KD-Tree (correctness)
        tail_start = int(self._kdtree_valid_n)
        if tail_start < self.num_vertices:
            tail = self.vertices[tail_start : self.num_vertices]
            vec = q - tail
            dists = np.linalg.norm(vec, axis=1)
            j = int(np.argmin(dists))
            d_tail = float(dists[j])
            if d_tail < best_dist:
                best_dist = d_tail
                best_idx = tail_start + j

        return int(best_idx)

    def _radius_indices(self, q: np.ndarray, r: float) -> np.ndarray:
        """Return indices of vertices within radius r of q (KD-Tree + tail brute-force)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        r = float(r)

        if (not self.use_kdtree) or self.num_vertices <= 1:
            vec = q - self.vertices[: self.num_vertices]
            d = np.linalg.norm(vec, axis=1)
            return np.where(d <= r)[0].astype(int)

        self._maybe_rebuild_kdtree(force=False)

        idxs = []
        if self._kdtree is not None and self._kdtree_valid_n > 0:
            idxs = self._kdtree.query_ball_point(q, r)
        idxs = list(map(int, idxs)) if idxs is not None else []

        # Tail brute-force
        tail_start = int(self._kdtree_valid_n)
        if tail_start < self.num_vertices:
            tail = self.vertices[tail_start : self.num_vertices]
            vec = q - tail
            d = np.linalg.norm(vec, axis=1)
            tail_hits = np.where(d <= r)[0]
            idxs.extend((tail_start + int(j)) for j in tail_hits)

        if not idxs:
            return np.array([], dtype=int)

        # unique + sorted for stability
        return np.array(sorted(set(idxs)), dtype=int)

    def _knn_indices(self, q: np.ndarray, k: int) -> np.ndarray:
        """Return indices of k nearest vertices to q (KD-Tree + tail brute-force)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        k = int(k)
        if k <= 0 or self.num_vertices <= 0:
            return np.array([], dtype=int)

        # Cap k to existing vertices
        k = min(k, int(self.num_vertices))

        # Fallback brute force if KDTree disabled or too few points
        if (not self.use_kdtree) or self.num_vertices <= 1:
            vec = q - self.vertices[: self.num_vertices]
            d = np.linalg.norm(vec, axis=1)
            return np.argsort(d)[:k].astype(int)

        self._maybe_rebuild_kdtree(force=False)

        # 1) KDTree over prefix
        idxs = []
        if self._kdtree is not None and self._kdtree_valid_n > 0:
            kk = min(k, int(self._kdtree_valid_n))
            d, idx = self._kdtree.query(q, k=kk)
            idx = np.atleast_1d(idx)
            idxs.extend([int(i) for i in idx.tolist()])

        # 2) Tail brute force then merge by distance
        tail_start = int(self._kdtree_valid_n)
        if tail_start < self.num_vertices:
            tail = self.vertices[tail_start : self.num_vertices]
            vec = q - tail
            d_tail = np.linalg.norm(vec, axis=1)
            # take candidates from tail (up to k)
            take = min(k, len(d_tail))
            tail_local = np.argsort(d_tail)[:take]
            idxs.extend([tail_start + int(j) for j in tail_local.tolist()])

        if not idxs:
            return np.array([], dtype=int)

        # Unique + sort by true distance, then take top-k
        idxs = sorted(set(idxs))
        pts = self.vertices[idxs]
        d_all = np.linalg.norm(pts - q, axis=1)
        order = np.argsort(d_all)[:k]
        return np.array([idxs[int(i)] for i in order], dtype=int)

    @staticmethod
    def _get_sampling_bound(env):
        """Return a list[(min,max)] per dimension."""
        b = getattr(env, "bound", None)
        if b is not None and len(b) >= 1:
            return list(b)

        # 3D backward compatibility
        xr = getattr(env, "x_range", None)
        yr = getattr(env, "y_range", None)
        zr = getattr(env, "z_range", None)
        if xr is not None and yr is not None and zr is not None:
            return [xr, yr, zr]

        raise AttributeError(
            "Environment must provide 'bound'=[(min,max),...] (preferred) or x_range/y_range/z_range (3D legacy)."
        )

    def planning(self, *args, **kwargs):
        raise NotImplementedError

    def SampleGlobally(self):
        lo = np.array([mn for mn, _ in self.bound], dtype=float) + self.clearance
        hi = np.array([mx for _, mx in self.bound], dtype=float) - self.clearance
        return np.random.uniform(lo, hi)

    def SampleFree(self):
        while True:
            p = self.SampleGlobally()
            if not self.utils.is_inside_obs(p):
                return p

    def cost(self, vertex_index: int) -> float:
        c = 0.0
        while vertex_index != 0:
            parent = int(self.vertex_parents[vertex_index])
            c += float(np.linalg.norm(self.vertices[vertex_index] - self.vertices[parent]))
            vertex_index = parent
        return c

    def extract_path(self, goal_parent_index: int):
        path = [self.x_goal]
        idx = int(goal_parent_index)
        while idx != 0:
            path.append(self.vertices[idx])
            idx = int(self.vertex_parents[idx])
        path.append(self.vertices[idx])
        path.reverse()
        return np.stack(path, axis=0)

    def check_success(self, path) -> bool:
        if path is None or len(path) == 0:
            return False
        return np.all(path[0] == self.x_start) and np.all(path[-1] == self.x_goal)

    def get_path_len(self, path) -> float:
        if path is None or len(path) == 0:
            return np.inf
        path = np.array(path, dtype=float)
        disp = path[1:] - path[:-1]
        return float(np.linalg.norm(disp, axis=1).sum())

    def InGoalRegion(self, node) -> bool:
        return self.Line(node, self.x_goal) < self.step_len and (not self.utils.is_collision(node, self.x_goal))

    def get_path_planner_name(self):
        return self.path_planner_name

    @staticmethod
    def nearest_neighbor(node_list: np.ndarray, n: np.ndarray):
        vec_to_n = n - node_list
        nearest_index = int(np.argmin(np.linalg.norm(vec_to_n, axis=1)))
        return node_list[nearest_index], nearest_index

    @staticmethod
    def get_distance_and_direction(node_start: np.ndarray, node_end: np.ndarray):
        delta = node_end - node_start
        dist = float(np.linalg.norm(delta))
        if dist == 0.0:
            return 0.0, np.zeros_like(delta)
        return dist, delta / dist

    @staticmethod
    def Line(x_start: np.ndarray, x_goal: np.ndarray) -> float:
        return float(np.linalg.norm(x_goal - x_start))


class RRTStarND(RRTBaseND):
    def __init__(
        self,
        x_start,
        x_goal,
        step_len,
        search_radius,
        iter_max,
        env,
        clearance,
    ):
        super().__init__(
            x_start,
            x_goal,
            step_len,
            search_radius,
            iter_max,
            env,
            clearance,
            f"RRT* {len(x_start)}D",
        )
        self.visualizer = None

    def planning(self, visualize=False):
        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
            node_nearest_index = self._nearest_index(node_rand)
            node_nearest = self.vertices[node_nearest_index]
            node_new = self.new_state(node_nearest, node_rand)

            if not self.utils.is_collision(node_nearest, node_new):
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

            if (k + 1) % 1000 == 0:
                print(k + 1)

        goal_parent_index = self.search_goal_parent()
        if goal_parent_index is None:
            if visualize:
                self.visualize()
            return

        self.path = self.extract_path(goal_parent_index)
        if visualize:
            self.visualize()

    def new_state(self, node_start, node_goal):
        dist, direction = self.get_distance_and_direction(node_start, node_goal)
        dist = min(self.step_len, dist)
        return node_start + dist * direction

    def choose_parent(self, node_new, neighbor_indices, node_new_index, curr_node_new_cost):
        vec_neighbors_to_new = node_new - self.vertices[: self.num_vertices][neighbor_indices]
        dist_neighbors_to_new = np.linalg.norm(vec_neighbors_to_new, axis=-1)
        neighbor_costs = [self.cost(int(nei)) for nei in neighbor_indices]
        node_new_cost_candidates = np.array(neighbor_costs, dtype=float) + dist_neighbors_to_new
        best_idx = int(np.argmin(node_new_cost_candidates))
        if float(node_new_cost_candidates[best_idx]) < float(curr_node_new_cost):
            self.vertex_parents[node_new_index] = int(neighbor_indices[best_idx])

    def rewire(self, node_new, neighbor_indices, node_new_index):
        vec_new_to_neighbors = self.vertices[: self.num_vertices][neighbor_indices] - node_new
        dist_new_to_neighbors = np.linalg.norm(vec_new_to_neighbors, axis=-1)
        node_new_cost = self.cost(node_new_index)
        for i, neighbor_index in enumerate(neighbor_indices):
            neighbor_index = int(neighbor_index)
            if self.cost(neighbor_index) > node_new_cost + float(dist_new_to_neighbors[i]):
                self.vertex_parents[neighbor_index] = int(node_new_index)

    def search_goal_parent(self):
        vec_to_goal = self.x_goal - self.vertices[: self.num_vertices]
        dist_to_goal = np.linalg.norm(vec_to_goal, axis=-1)
        indices = np.where(dist_to_goal <= self.step_len)[0]
        if len(indices) == 0:
            return None

        total_cost_candidates = []
        for vertex_index, vertex, d in zip(indices, self.vertices[: self.num_vertices][indices], dist_to_goal[indices]):
            if not self.utils.is_collision(vertex, self.x_goal):
                total_cost_candidates.append(self.cost(int(vertex_index)) + float(d))
            else:
                total_cost_candidates.append(np.inf)

        return int(indices[int(np.argmin(total_cost_candidates))])

    def generate_random_node(self):
        return self.SampleFree()

    def find_near_neighbors(self, node_new, node_new_index=None):
        # RRT* near radius scaling with dimensionality
        n = max(self.num_vertices, 2)
        r = min(self.search_radius * (math.log(n) / n) ** (1.0 / float(self.dim)), self.step_len)

        indices = self._radius_indices(node_new, r)
        if len(indices) == 0:
            return np.array([], dtype=int)

        neighbor_indices = []
        for vertex_index in indices:
            vertex_index = int(vertex_index)
            if node_new_index is not None and vertex_index == int(node_new_index):
                continue
            vertex = self.vertices[vertex_index]
            if not self.utils.is_collision(node_new, vertex):
                neighbor_indices.append(vertex_index)

        return np.array(neighbor_indices, dtype=int)

    def visualize(self, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = f"rrt* {self.dim}D, iteration {self.iter_max}"
        if self.visualizer is not None:
            self.visualizer.animation(
                self.vertices[: self.num_vertices],
                self.vertex_parents[: self.num_vertices],
                self.path,
                figure_title,
                animation=False,
                img_filename=img_filename,
            )


class IRRTStarND(RRTStarND):
    def __init__(
        self,
        start,
        goal,
        environment,
        iter_max,
        batch_size,
        plot_flag=False,
        timer=None,
        step_len=None,
        search_radius=None,
        clearance=None,
    ):
        """IRRT* planner with the same public interface as BITStar."""
        self.timer = timer
        self.env = environment
        self.iter_max = int(iter_max)
        self.batch_size = int(batch_size) if batch_size is not None else 0
        self.plot_planning_process = bool(plot_flag)

        if step_len is None:
            step_len = 1.0
        if search_radius is None:
            search_radius = 5.0
        if clearance is None:
            clearance = 0.0

        super().__init__(
            x_start=np.array(start, dtype=float),
            x_goal=np.array(goal, dtype=float),
            step_len=float(step_len),
            search_radius=float(search_radius),
            iter_max=self.iter_max,
            env=environment,
            clearance=float(clearance),
        )

        # BITStar-compatible bookkeeping
        self.start = self.to_key(start)
        self.goal = self.to_key(goal)
        self.samples = []
        self.edges = {}
        self.g_scores = {}
        self.T = 0
        self.path = []

        # IRRT* bookkeeping
        self.path_solutions = []

    def to_key(self, point, ndigits=6):
        arr = np.round(np.array(point, dtype=float), ndigits)
        return tuple(arr.tolist())

    def init(self):
        cMin, direction = self.get_distance_and_direction(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(direction)
        x_center = (self.x_start + self.x_goal) / 2.0
        return cMin, x_center, C

    def planning(self, visualize=False, refresh_interval=1):
        """Plan a path (BITStar-compatible return signature)."""
        start_checks = getattr(self.env, "collision_check_count", 0)
        init_time = time()

        iteration_costs = []
        iteration_times = []
        first_solution_iter = None
        first_solution_cost = None
        first_solution_nodes = None
        final_solution_nodes = 0

        start_goal_straightline_dist, x_center, C = self.init()
        c_best = np.inf
        x_best = None

        for k in range(self.iter_max):
            final_iter = k

            if len(self.path_solutions) > 0:
                c_best, x_best = self.find_best_path_solution()

            node_rand = self.generate_random_node(c_best, start_goal_straightline_dist, x_center, C)
            node_nearest_index = self._nearest_index(node_rand)
            node_nearest = self.vertices[node_nearest_index]
            node_new = self.new_state(node_nearest, node_rand)

            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    pass
                else:
                    node_near, neighbor_indices = self.near_neighbors(node_new, k_nearest=25)
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.num_vertices += 1

                    node_min_index = node_nearest_index
                    cost_min = self.cost(node_nearest_index) + float(np.linalg.norm(node_new - node_nearest))

                    for node_near_index in neighbor_indices:
                        node_near_index = int(node_near_index)
                        if node_near_index == node_nearest_index:
                            continue
                        node_near_i = self.vertices[node_near_index]
                        if not self.utils.is_collision(node_near_i, node_new):
                            cost_new = self.cost(node_near_index) + float(np.linalg.norm(node_new - node_near_i))
                            if cost_new < cost_min:
                                node_min_index = node_near_index
                                cost_min = cost_new

                    self.vertex_parents[node_new_index] = int(node_min_index)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                    if self.InGoalRegion(node_new):
                        self.path_solutions.append(int(node_new_index))

            cur_time = time() - init_time
            iteration_times.append(cur_time)
            iteration_costs.append(float(c_best) if np.isfinite(c_best) else np.inf)

            if first_solution_iter is None and np.isfinite(c_best) and x_best is not None:
                first_solution_iter = k + 1
                first_solution_cost = float(c_best)
                try:
                    tmp_path = self.extract_path(x_best)
                    first_solution_nodes = len(tmp_path) if tmp_path is not None else None
                except Exception:
                    first_solution_nodes = None

        if len(self.path_solutions) > 0:
            c_best, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []
            c_best = np.inf

        if visualize:
            try:
                self.visualize(x_center, c_best, start_goal_straightline_dist, C)
            except Exception:
                pass

        def _to_key_list(path_arr):
            return [self.to_key(p) for p in path_arr]

        path_keys = _to_key_list(self.path) if isinstance(self.path, np.ndarray) and len(self.path) > 0 else []
        final_solution_nodes = len(path_keys) if path_keys else 0

        samples = [self.to_key(p) for p in self.vertices[: self.num_vertices]]

        edges = {}
        for idx in range(1, self.num_vertices):
            parent_idx = int(self.vertex_parents[idx])
            child_k = self.to_key(self.vertices[idx])
            parent_k = self.to_key(self.vertices[parent_idx])
            edges[child_k] = parent_k

        n_checks = getattr(self.env, "collision_check_count", 0) - start_checks
        best_cost = float(c_best) if np.isfinite(c_best) else np.inf
        total_samples = int(self.num_vertices)
        runtime = time() - init_time

        return (
            path_keys,
            samples,
            edges,
            n_checks,
            best_cost,
            total_samples,
            runtime,
            final_iter if "final_iter" in locals() else 0,
            iteration_costs,
            iteration_times,
            first_solution_iter,
            first_solution_cost,
            first_solution_nodes,
            final_solution_nodes,
        )

    def near_neighbors(self, node_new, k_nearest: int = 20):
        """Use KNN neighborhood instead of fixed radius to cap collision checks."""
        indices = self._knn_indices(node_new, k_nearest)

        # optional: remove self if somehow included
        # (in IRRT*, node_new is not yet in vertices when called, so usually not needed)

        return self.vertices[: self.num_vertices][indices], np.array(indices, dtype=int)

    def find_best_path_solution(self):
        path_costs = []
        for goal_parent_vertex_idx in self.path_solutions:
            v = self.vertices[: self.num_vertices][goal_parent_vertex_idx]
            path_costs.append(self.cost(int(goal_parent_vertex_idx)) + self.Line(v, self.x_goal))
        best_path_idx = int(np.argmin(path_costs))
        c_best = float(path_costs[best_path_idx])
        x_best = int(self.path_solutions[best_path_idx])
        return c_best, x_best

    def generate_random_node(self, c_max, c_min, x_center, C):
        if c_max < np.inf:
            return self.SampleInformedSubset(c_max, c_min, x_center, C)
        return self.SampleFree()

    def SampleInformedSubset(self, c_max, c_min, x_center, C):
        # nD prolate hyperspheroid parameters
        under = c_max**2 - c_min**2
        eps = 1e-9 if under < 0 else 0.0
        r = np.zeros(self.dim, dtype=float)
        r[0] = c_max / 2.0
        if self.dim > 1:
            r[1:] = math.sqrt(max(0.0, under + eps)) / 2.0

        L = np.diag(r)
        while True:
            xball = self.SampleUnitBall(self.dim)
            node_rand = (C @ L @ xball) + x_center
            if self.utils.is_valid(node_rand):
                break
        return node_rand

    @staticmethod
    def SampleUnitBall(dim: int):
        # Uniform sampling in the n-ball via normal direction + radius^(1/n)
        v = np.random.normal(size=dim)
        nrm = float(np.linalg.norm(v))
        if nrm == 0.0:
            v[0] = 1.0
            nrm = 1.0
        v = v / nrm
        r = np.random.uniform(0.0, 1.0) ** (1.0 / float(dim))
        return r * v

    @staticmethod
    def RotationToWorldFrame(direction_unit: np.ndarray):
        """Return an orthonormal matrix C such that C[:,0] aligns with direction_unit.

        Uses a Householder reflection to map e1 -> direction_unit.
        """
        a = np.array(direction_unit, dtype=float)
        dim = int(a.shape[0])
        if dim == 0:
            raise ValueError("direction_unit must be non-empty")
        a_norm = float(np.linalg.norm(a))
        if a_norm == 0.0:
            return np.eye(dim)
        a = a / a_norm

        e1 = np.zeros(dim, dtype=float)
        e1[0] = 1.0
        u = e1 - a
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-12:
            return np.eye(dim)
        u = u / u_norm
        H = np.eye(dim) - 2.0 * np.outer(u, u)
        return H

    def visualize(self, x_center, c_best, start_goal_straightline_dist, C, figure_title=None, img_filename=None):
        if figure_title is None:
            figure_title = f"irrt* {self.dim}D, iteration {self.iter_max}"
        if self.visualizer is not None:
            self.visualizer.animation(
                self.vertices[: self.num_vertices],
                self.vertex_parents[: self.num_vertices],
                self.path,
                figure_title,
                x_center,
                c_best,
                start_goal_straightline_dist,
                C,
                img_filename=img_filename,
            )


class NIRRTStarECSP(IRRTStarND):
    """Neural-Informed IRRT* (NIRRT*).

    Hybrid sampling policy:
      - with prob 0.4: standard IRRT* informed (ellipsoid) sampling
      - otherwise: sample uniformly from neural-guidance states Xguide
    """

    def __init__(
        self,
        start,
        goal,
        environment,
        iter_max,
        batch_size,
        neural_wrapper=None,
        plot_flag=False,
        timer=None,
        step_len=None,
        search_radius=None,
        clearance=None,
        guidance_batch_size=256,
        guidance_batch_size_min=64,
        guidance_batch_decay=0.9,

    ):
        super().__init__(
            start=start,
            goal=goal,
            environment=environment,
            iter_max=iter_max,
            batch_size=batch_size,
            plot_flag=plot_flag,
            timer=timer,
            step_len=step_len,
            search_radius=search_radius,
            clearance=clearance,
        )

        self.neural_wrapper = neural_wrapper
        self.guidance_batch_size = int(guidance_batch_size)
        self.guidance_batch_size_min = int(guidance_batch_size_min)
        self.guidance_batch_decay = float(guidance_batch_decay)

        # last cost used to generate guidance
        self._last_guidance_cost = np.inf

        # Guidance states cache (list[np.ndarray])
        self.Xguide = []

        # NN-guidance thresholds (mirrors nibit_star_fixed defaults)
        self.free_th = 0.6
        self.free_th_min = 0.4
        self.free_th_relax = 0.90

    
    def _sample_informed_raw(self, c_max, c_min, x_center, C):
        """Informed (hyper-)ellipsoid sampling *without* collision / validity checks.
        Used to generate candidates for NN scoring.
        """
        under = c_max**2 - c_min**2
        eps = 1e-9 if under < 0 else 0.0
        r = np.zeros(self.dim, dtype=float)
        r[0] = c_max / 2.0
        if self.dim > 1:
            r[1:] = math.sqrt(max(0.0, under + eps)) / 2.0
        L = np.diag(r)
        xball = self.SampleUnitBall(self.dim)
        node_rand = (C @ L @ xball) + x_center
        return node_rand

    def sample_from_env(self, c_best, batch_size):
        """Generate guidance states Xguide via NN scoring.
        Candidates come from a mix of informed sampling (ellipsoid) and global sampling,
        then we keep the top points by P(path) among those passing an adaptive P(free) filter.
        Returns a list of numeric vectors (np.ndarray) of length dim.
        """
        # 1) Generate candidates (no collision checks)
        candidates = []

        has_solution = (c_best < np.inf) and (self.c_min is not None)

        if has_solution:
            # 100% informed (ellipsoid) sampling
            c_min = float(self.c_min)
            x_center = self.x_center
            C = self.C
            for _ in range(int(batch_size)):
                candidates.append(
                    self._sample_informed_raw(float(c_best), c_min, x_center, C)
                )
        else:
            # 100% global sampling
            for _ in range(int(batch_size)):
                candidates.append(self.SampleGlobally())

        joints_np = np.asarray(candidates, dtype=float).reshape((-1, self.dim))
        if len(joints_np) == 0:
            return []

        # 2) If no network, just return random points
        if self.neural_wrapper is None:
            if len(joints_np) > batch_size:
                idx = np.random.choice(len(joints_np), size=batch_size, replace=False)
                joints_np = joints_np[idx]
            return [np.array(p, dtype=float) for p in joints_np]

        # 3) NN scoring
        if hasattr(self.neural_wrapper, "predict_probs"):
            p_free, p_path = self.neural_wrapper.predict_probs(joints_np)
        else:
            _, p_free = self.neural_wrapper.get_free_mask(joints_np, prob_th=0.0)
            _, p_path = self.neural_wrapper.get_path_mask(joints_np, prob_th=0.0)

        p_free = np.asarray(p_free, dtype=float).reshape(-1)
        p_path = np.asarray(p_path, dtype=float).reshape(-1)

        # 4) Adaptive free filter
        free_th = float(self.free_th)
        free_mask = (p_free >= free_th)
        while (not np.any(free_mask)) and (free_th > self.free_th_min + 1e-9):
            free_th = max(self.free_th_min, free_th * self.free_th_relax)
            free_mask = (p_free >= free_th)
        self.free_th = free_th
        if not np.any(free_mask):
            free_mask = np.ones_like(free_mask, dtype=bool)

        # 5) Select top by cost-aware score (goal-directed)
        idxs = np.where(free_mask)[0]
        if len(idxs) == 0:
            return []

        K = min(batch_size, len(idxs))

        # ---- normalize distance to goal using env.bound ----
        # bound shape: (dim,2)
        lo = self.bound[:, 0]
        hi = self.bound[:, 1]
        span = np.maximum(hi - lo, 1e-9)

        goal = self.x_goal  # np.ndarray dim
        dg = (joints_np - goal[None, :]) / span[None, :]
        d_goal_norm = np.linalg.norm(dg, axis=1)  # (M,)

        alpha = 2.0  # try 1.0~4.0
        score = (p_path * p_free) * np.exp(-alpha * d_goal_norm)

        order = idxs[np.argsort(score[idxs])[::-1]]
        sel = order[:K]
        return [np.array(joints_np[i], dtype=float) for i in sel]

    def _update_guidance(self, c_best: float, k: int):
        """
        Refresh Xguide only when the improvement over last refresh is large enough:
        delta = last_cost - c_best >= max(abs_eps, rel_eps * last_cost)

        Also decay guidance_batch_size over time.
        """
        if self.neural_wrapper is None:
            self.Xguide = []
            return

        # If no solution yet: still build guidance from GLOBAL candidates,
        # but do it periodically (or when cache empty) to avoid extra NN cost.
        if not np.isfinite(c_best):
            refresh_every = 20
            if (not self.Xguide) or (k % refresh_every == 0):
                try:
                    bs = max(self.guidance_batch_size_min, int(self.guidance_batch_size))
                    self.Xguide = self.sample_from_env(c_best, bs)  # will use global sampling
                    # do NOT update _last_guidance_cost (still inf)
                    new_bs = int(round(self.guidance_batch_size * self.guidance_batch_decay))
                    self.guidance_batch_size = max(self.guidance_batch_size_min, new_bs)
                except Exception:
                    self.Xguide = []
            return

        # ---- minimal improvement thresholds (tune here) ----
        abs_eps = 1e-3      # absolute minimum improvement
        rel_eps = 5e-3      # 0.5% relative minimum improvement

        last = float(getattr(self, "_last_guidance_cost", float("inf")))

        # First time we get a finite solution, allow refresh immediately
        if not np.isfinite(last):
            do_refresh = True
        else:
            delta = last - c_best
            min_improve = max(abs_eps, rel_eps * abs(last))
            do_refresh = (delta >= min_improve)

        if not do_refresh:
            return

        try:
            # Refresh Xguide using current batch_size
            bs = max(self.guidance_batch_size_min, int(self.guidance_batch_size))
            self.Xguide = self.sample_from_env(c_best, bs)
            self._last_guidance_cost = float(c_best)

            # Decay batch_size for next refresh
            new_bs = int(round(self.guidance_batch_size * self.guidance_batch_decay))
            self.guidance_batch_size = max(self.guidance_batch_size_min, new_bs)

        except Exception:
            self.Xguide = []

    def _sample_from_guidance(self):
        if not self.Xguide:
            return None
        idx = int(np.random.randint(0, len(self.Xguide)))
        return np.array(self.Xguide[idx], dtype=float)

    def generate_random_node(self, c_max, c_min, x_center, C, k=0):
        """
        Hybrid sampler with dynamic mix:
        - before first solution:  
        - after  first solution
        """
        # Update guidance cache (uses current best cost)
        self._update_guidance(c_max, k)

        has_first_solution = (len(self.path_solutions) > 0)

        # User-requested schedule:
        # before first solution -> 
        # after  first solution -> 
        p_irrt = 0.05 if (not has_first_solution) else 0.4

        if np.random.rand() < p_irrt:
            # IRRT* informed / global (fallback)
            if c_max < np.inf:
                return self.SampleInformedSubset(c_max, c_min, x_center, C)
            return self.SampleFree()

        # Guidance sampling
        g = self._sample_from_guidance()
        if g is not None and self.utils.is_valid(g):
            return g

        # fallback if guidance empty/invalid
        if c_max < np.inf:
            return self.SampleInformedSubset(c_max, c_min, x_center, C)
        return self.SampleFree()

    def planning(self, visualize=False, refresh_interval=1):
        """Same as IRRTStarND.planning, but uses hybrid sampler."""
        start_checks = getattr(self.env, "collision_check_count", 0)
        init_time = time()

        iteration_costs = []
        iteration_times = []
        first_solution_iter = None
        first_solution_cost = None
        first_solution_nodes = None
        final_solution_nodes = 0

        start_goal_straightline_dist, x_center, C = self.init()
        c_best = np.inf
        x_best = None

        for k in range(self.iter_max):
            final_iter = k

            if len(self.path_solutions) > 0:
                c_best, x_best = self.find_best_path_solution()

            node_rand = self.generate_random_node(
                c_best, start_goal_straightline_dist, x_center, C, k=k
            )
            node_nearest_index = self._nearest_index(node_rand)
            node_nearest = self.vertices[node_nearest_index]
            node_new = self.new_state(node_nearest, node_rand)

            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    pass
                else:
                    node_near, neighbor_indices = self.near_neighbors(node_new, k_nearest=25)
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.num_vertices += 1

                    node_min_index = node_nearest_index
                    cost_min = self.cost(node_nearest_index) + float(
                        np.linalg.norm(node_new - node_nearest)
                    )

                    for node_near_index in neighbor_indices:
                        node_near_index = int(node_near_index)
                        if node_near_index == node_nearest_index:
                            continue
                        node_near_i = self.vertices[node_near_index]
                        if not self.utils.is_collision(node_near_i, node_new):
                            cost_new = self.cost(node_near_index) + float(
                                np.linalg.norm(node_new - node_near_i)
                            )
                            if cost_new < cost_min:
                                node_min_index = node_near_index
                                cost_min = cost_new

                    self.vertex_parents[node_new_index] = int(node_min_index)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                    if self.InGoalRegion(node_new):
                        self.path_solutions.append(int(node_new_index))
                        # If a new solution is found, update best and refresh guidance immediately
                        c_best_new, x_best_new = self.find_best_path_solution()
                        if np.isfinite(c_best_new) and c_best_new < c_best:
                            c_best, x_best = c_best_new, x_best_new
                            self._update_guidance(c_best, k)

            cur_time = time() - init_time
            iteration_times.append(cur_time)
            iteration_costs.append(float(c_best) if np.isfinite(c_best) else np.inf)

            if first_solution_iter is None and np.isfinite(c_best) and x_best is not None:
                first_solution_iter = k + 1
                first_solution_cost = float(c_best)
                try:
                    tmp_path = self.extract_path(x_best)
                    first_solution_nodes = len(tmp_path) if tmp_path is not None else None
                except Exception:
                    first_solution_nodes = None

        if len(self.path_solutions) > 0:
            c_best, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []
            c_best = np.inf

        if visualize:
            try:
                self.visualize(x_center, c_best, start_goal_straightline_dist, C)
            except Exception:
                pass

        def _to_key_list(path_arr):
            return [self.to_key(p) for p in path_arr]

        path_keys = (
            _to_key_list(self.path)
            if isinstance(self.path, np.ndarray) and len(self.path) > 0
            else []
        )
        final_solution_nodes = len(path_keys) if path_keys else 0

        samples = [self.to_key(p) for p in self.vertices[: self.num_vertices]]

        edges = {}
        for idx in range(1, self.num_vertices):
            parent_idx = int(self.vertex_parents[idx])
            child_k = self.to_key(self.vertices[idx])
            parent_k = self.to_key(self.vertices[parent_idx])
            edges[child_k] = parent_k

        n_checks = getattr(self.env, "collision_check_count", 0) - start_checks
        best_cost = float(c_best) if np.isfinite(c_best) else np.inf
        total_samples = int(self.num_vertices)
        runtime = time() - init_time

        return (
            path_keys,
            samples,
            edges,
            n_checks,
            best_cost,
            total_samples,
            runtime,
            final_iter if "final_iter" in locals() else 0,
            iteration_costs,
            iteration_times,
            first_solution_iter,
            first_solution_cost,
            first_solution_nodes,
            final_solution_nodes,
        )


class NIRRTStarPNGND(IRRTStarND):
    """Neural-Informed IRRT* with PNG-style point-cloud guidance (nD).

    This is an n-dimensional generalization of the uploaded 3D NIRRT*-PNG class.
    It generates a candidate point cloud (either globally or from the current
    informed subset) and queries `png_wrapper.classify_path_points(...)` to keep
    only the points predicted to lie on a likely path. During planning, the
    sampler mixes:
      - point-cloud guided samples with probability `pc_sample_rate`
      - otherwise: standard IRRT* informed/global samples

    Expected PNG wrapper API (same as the 3D version):
        path_pred, path_score = png_wrapper.classify_path_points(
            pc: np.ndarray [N,dim] float32,
            start_mask: np.ndarray [N] float32,
            goal_mask: np.ndarray [N] float32,
        )
    where `path_pred` is a binary/boolean vector (or values where nonzero means True).
    """

    def __init__(
        self,
        start,
        goal,
        environment,
        iter_max,
        batch_size,
        neural_wrapper,
        plot_flag=False,
        timer=None,
        step_len=None,
        search_radius=None,
        clearance=None,
        pc_n_points: int = 2048,
        pc_over_sample_scale: int = 4,
        pc_sample_rate: float = 0.5,
        pc_update_cost_ratio: float = 0.98,
    ):
        super().__init__(
            start=start,
            goal=goal,
            environment=environment,
            iter_max=iter_max,
            batch_size=batch_size,
            plot_flag=plot_flag,
            timer=timer,
            step_len=step_len,
            search_radius=search_radius,
            clearance=clearance,
        )
        self.png_wrapper = neural_wrapper

        self.pc_n_points = int(pc_n_points)
        self.pc_over_sample_scale = int(pc_over_sample_scale)
        self.pc_sample_rate = float(pc_sample_rate)
        self.pc_update_cost_ratio = float(pc_update_cost_ratio)

        # use step_len as the neighborhood radius (mirrors 3D version)
        self.pc_neighbor_radius = float(self.step_len)*5

        self.path_point_cloud_pred = None  # np.ndarray [M,dim] or None
        self.pose_range=self.env.pose_range
        self._pc_low = self.pose_range[:, 0]
        self._pc_high = self.pose_range[:, 1]
        self._pc_span = self._pc_high - self._pc_low
        self._pc_span[self._pc_span == 0] = 1e-6

        # optional visualizer hook (not provided in this nD file)
        # callers may set self.visualizer if they have one.
        # self.visualizer should expose: set_path_point_cloud_pred(pc)
        # and animation(...) like other visualizers.
        self.visualizer = getattr(self, "visualizer", None)

    # ---------------- Point cloud helpers ----------------
    def _normalize_pc(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc, dtype=np.float32)
        pc_norm = (pc - self._pc_low) / self._pc_span
        pc_norm = np.clip(pc_norm, 0.0, 1.0)
        return pc_norm.astype(np.float32)

    def _mask_around_points(self, pc: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
        """Return boolean mask for pc points within `radius` of any `points`."""
        if pc is None or len(pc) == 0:
            return np.zeros((0,), dtype=bool)
        pts = np.asarray(points, dtype=float).reshape((-1, self.dim))
        r2 = float(radius) ** 2
        # compute min squared distance to any anchor point
        # (N,1,dim) - (1,M,dim) -> (N,M,dim)
        dif = pc[:, None, :] - pts[None, :, :]
        d2 = np.sum(dif * dif, axis=-1)  # (N,M)
        return (np.min(d2, axis=1) <= r2)

    def _global_point_cloud(self, n_points: int) -> np.ndarray:
        lo = self.bound[:, 0] + self.clearance
        hi = self.bound[:, 1] - self.clearance
        # guard against invalid bounds after clearance
        hi = np.maximum(hi, lo + 1e-9)
        return np.random.uniform(lo[None, :], hi[None, :], size=(int(n_points), self.dim))

    def _informed_point_cloud(self, cmax: float, cmin: float, x_center: np.ndarray, C: np.ndarray, n_points: int) -> np.ndarray:
        # reuse IRRTStarND informed sampling (with validity checks) for robustness
        pc = []
        for _ in range(int(n_points)):
            pc.append(self.SampleInformedSubset(float(cmax), float(cmin), x_center, C))
        return np.asarray(pc, dtype=float).reshape((-1, self.dim))

    def update_point_cloud(self, cmax: float, cmin: float, x_center: np.ndarray, C: np.ndarray):
        """Generate point cloud and run PNG wrapper to keep predicted path points."""
        if self.pc_sample_rate <= 0.0:
            self.path_point_cloud_pred = None
            if hasattr(self.visualizer, "set_path_point_cloud_pred"):
                self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)
            return

        n_raw = int(max(1, self.pc_n_points * max(1, self.pc_over_sample_scale)))

        if np.isfinite(cmax):
            pc = self._informed_point_cloud(cmax, cmin, x_center, C, n_raw)
        else:
            pc = self._global_point_cloud(n_raw)

        start_mask = self._mask_around_points(pc, self.x_start[None, :], self.pc_neighbor_radius)
        goal_mask = self._mask_around_points(pc, self.x_goal[None, :], self.pc_neighbor_radius)
        # If no wrapper, fallback to using the raw point cloud directly
        if self.png_wrapper is None:
            self.path_point_cloud_pred = pc
        else:
            pc_norm = self._normalize_pc(pc)
            path_pred, _ = self.png_wrapper.classify_path_points(
                pc_norm,
                start_mask.astype(np.float32),
                goal_mask.astype(np.float32),
            )
            path_pred = np.asarray(path_pred).reshape((-1,))
            keep = np.nonzero(path_pred)[0]
            self.path_point_cloud_pred = pc[keep] if len(keep) > 0 else np.zeros((0, self.dim), dtype=float)

        if hasattr(self.visualizer, "set_path_point_cloud_pred"):
            self.visualizer.set_path_point_cloud_pred(self.path_point_cloud_pred)

    def init_pc(self, x_center: np.ndarray, C: np.ndarray, cmin: float):
        self.update_point_cloud(cmax=np.inf, cmin=float(cmin), x_center=x_center, C=C)

    def SamplePointCloud(self):
        pc = self.path_point_cloud_pred
        if pc is None or len(pc) == 0:
            return None
        idx = int(np.random.randint(0, len(pc)))
        return np.array(pc[idx], dtype=float)

    # ---------------- Planning override ----------------

    def generate_random_node_png(
        self,
        c_curr: float,
        c_min: float,
        x_center: np.ndarray,
        C: np.ndarray,
        c_update: float,
    ):
        # Refresh PC when best cost improves enough
        if np.isfinite(c_curr) and (c_curr < self.pc_update_cost_ratio * float(c_update)):
            self.update_point_cloud(cmax=float(c_curr), cmin=float(c_min), x_center=x_center, C=C)
            c_update = float(c_curr)

        # Sample from PC with probability pc_sample_rate
        if (self.pc_sample_rate > 0.0) and (np.random.random() < self.pc_sample_rate):
            g = self.SamplePointCloud()
            if g is not None and self.utils.is_valid(g):
                return g, c_update

        # Fallback to standard IRRT* sampler
        if np.isfinite(c_curr):
            return self.SampleInformedSubset(float(c_curr), float(c_min), x_center, C), c_update
        return self.SampleFree(), c_update

    def planning(self, visualize=False, refresh_interval=1):
        """Plan a path (same return signature as IRRTStarND.planning)."""
        start_checks = getattr(self.env, "collision_check_count", 0)
        init_time = time()

        iteration_costs = []
        iteration_times = []
        first_solution_iter = None
        first_solution_cost = None
        first_solution_nodes = None
        final_solution_nodes = 0

        start_goal_straightline_dist, x_center, C = self.init()
        c_best = np.inf
        x_best = None

        # init point cloud in global mode
        self.init_pc(x_center=x_center, C=C, cmin=float(start_goal_straightline_dist))
        c_update = np.inf

        for k in range(self.iter_max):
            final_iter = k

            if len(self.path_solutions) > 0:
                c_best, x_best = self.find_best_path_solution()

            node_rand, c_update = self.generate_random_node_png(
                c_best, start_goal_straightline_dist, x_center, C, c_update
            )

            node_nearest_index = self._nearest_index(node_rand)
            node_nearest = self.vertices[node_nearest_index]
            node_new = self.new_state(node_nearest, node_rand)

            if not self.utils.is_collision(node_nearest, node_new):
                if np.linalg.norm(node_new - node_nearest) < 1e-8:
                    pass
                else:
                    node_near, neighbor_indices = self.near_neighbors(node_new, k_nearest=25)
                    node_new_index = self.num_vertices
                    self.vertices[node_new_index] = node_new
                    self.num_vertices += 1

                    node_min_index = node_nearest_index
                    cost_min = self.cost(node_nearest_index) + float(np.linalg.norm(node_new - node_nearest))

                    for node_near_index in neighbor_indices:
                        node_near_index = int(node_near_index)
                        if node_near_index == node_nearest_index:
                            continue
                        node_near_i = self.vertices[node_near_index]
                        if not self.utils.is_collision(node_near_i, node_new):
                            cost_new = self.cost(node_near_index) + float(np.linalg.norm(node_new - node_near_i))
                            if cost_new < cost_min:
                                node_min_index = node_near_index
                                cost_min = cost_new

                    self.vertex_parents[node_new_index] = int(node_min_index)
                    self.rewire(node_new, neighbor_indices, node_new_index)

                    if self.InGoalRegion(node_new):
                        self.path_solutions.append(int(node_new_index))

            cur_time = time() - init_time
            iteration_times.append(cur_time)
            iteration_costs.append(float(c_best) if np.isfinite(c_best) else np.inf)

            if first_solution_iter is None and np.isfinite(c_best) and x_best is not None:
                first_solution_iter = k + 1
                first_solution_cost = float(c_best)
                try:
                    tmp_path = self.extract_path(x_best)
                    first_solution_nodes = len(tmp_path) if tmp_path is not None else None
                except Exception:
                    first_solution_nodes = None

        if len(self.path_solutions) > 0:
            c_best, x_best = self.find_best_path_solution()
            self.path = self.extract_path(x_best)
        else:
            self.path = []
            c_best = np.inf

        if visualize:
            try:
                self.visualize(x_center, c_best, start_goal_straightline_dist, C)
            except Exception:
                pass

        def _to_key_list(path_arr):
            return [self.to_key(p) for p in path_arr]

        path_keys = _to_key_list(self.path) if isinstance(self.path, np.ndarray) and len(self.path) > 0 else []
        final_solution_nodes = len(path_keys) if path_keys else 0

        samples = [self.to_key(p) for p in self.vertices[: self.num_vertices]]

        edges = {}
        for idx in range(1, self.num_vertices):
            parent_idx = int(self.vertex_parents[idx])
            child_k = self.to_key(self.vertices[idx])
            parent_k = self.to_key(self.vertices[parent_idx])
            edges[child_k] = parent_k

        n_checks = getattr(self.env, "collision_check_count", 0) - start_checks
        best_cost = float(c_best) if np.isfinite(c_best) else np.inf
        total_samples = int(self.num_vertices)
        runtime = time() - init_time

        return (
            path_keys,
            samples,
            edges,
            n_checks,
            best_cost,
            total_samples,
            runtime,
            final_iter if "final_iter" in locals() else 0,
            iteration_costs,
            iteration_times,
            first_solution_iter,
            first_solution_cost,
            first_solution_nodes,
            final_solution_nodes,
        )


def get_nirrtstar_png_planner(args, problem, neural_wrapper):
    """Factory for NIRRT*-PNG (nD)."""
    return NIRRTStarPNGND(
        start=problem.get("start", problem.get("x_start")),
        goal=problem.get("goal", problem.get("x_goal")),
        environment=problem.get("env", problem.get("environment")),
        iter_max=args.iter_max,
        batch_size=getattr(args, "batch_size", 200),
        neural_wrapper=neural_wrapper,
        plot_flag=getattr(args, "plot_flag", False),
        timer=problem.get("timer", None) if isinstance(problem, dict) else None,
        step_len=getattr(args, "step_len", None),
        search_radius=problem.get("search_radius", None),
        clearance=getattr(args, "clearance", None),
        pc_n_points=getattr(args, "pc_n_points", 2048),
        pc_over_sample_scale=getattr(args, "pc_over_sample_scale", 3),
        pc_sample_rate=getattr(args, "pc_sample_rate", 0.5),
        pc_update_cost_ratio=getattr(args, "pc_update_cost_ratio", 0.98),
    )


# ------------------ Factories (kept for compatibility) ------------------


def get_rrt_star_planner(args, problem, neural_wrapper=None):
    return RRTStarND(
        problem["x_start"],
        problem["x_goal"],
        args.step_len,
        problem["search_radius"],
        args.iter_max,
        problem["env"],
        args.clearance,
    )


def get_irrt_star_planner(args, problem, neural_wrapper=None):
    return IRRTStarND(
        problem["x_start"],
        problem["x_goal"],
        problem.get("env", problem.get("environment")),
        args.iter_max,
        getattr(args, "batch_size", 200),
        plot_flag=getattr(args, "plot_flag", False),
        timer=problem.get("timer", None) if isinstance(problem, dict) else None,
        step_len=getattr(args, "step_len", None),
        search_radius=problem.get("search_radius", None),
        clearance=getattr(args, "clearance", None),
    )


def get_irrtstar_planner(args, problem, neural_wrapper=None):
    planner = IRRTStarND(
        problem["start"],
        problem["goal"],
        problem["env"],
        args.iter_max,
        args.batch_size,
        plot_flag=getattr(args, "plot_flag", False),
        timer=problem.get("timer", None) if isinstance(problem, dict) else None,
        step_len=getattr(args, "step_len", None),
        search_radius=getattr(args, "search_radius", None),
        clearance=getattr(args, "clearance", None),
    )
    return planner


def get_nirrtstarECSP_planner(args, problem, neural_wrapper=None):
    """Factory for NIRRT* (hybrid IRRT* + neural guidance)."""
    planner = NIRRTStarECSP(
        problem.get("start", problem.get("x_start")),
        problem.get("goal", problem.get("x_goal")),
        problem.get("env", problem.get("environment")),
        args.iter_max,
        getattr(args, "batch_size", 200),
        neural_wrapper=neural_wrapper,
        plot_flag=getattr(args, "plot_flag", False),
        timer=problem.get("timer", None) if isinstance(problem, dict) else None,
        step_len=getattr(args, "step_len", None),
        search_radius=getattr(args, "search_radius", None),
        clearance=getattr(args, "clearance", None),
        guidance_batch_size=getattr(args, "guidance_batch_size", 100),
    )
    return planner


# Default factory (BITStar-compatible).
get_path_planner = get_irrtstar_planner
