import math
from time import time
from dataclasses import dataclass
import numpy as np

INF = float("inf")


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
        return b.reshape((2, dim)).T

    # (2,dim)
    if b.ndim == 2 and b.shape == (2, dim):
        return b.T

    # (dim,2)
    if b.ndim == 2 and b.shape == (dim, 2):
        return b

    raise ValueError(f"Bad env.bound shape {b.shape}; expected flat (2*dim,) or (dim,2). dim={dim}")


@dataclass
class _TreeState:
    # parent[i] = parent index in this tree, -1 for root/unreached
    parent: np.ndarray
    # cost-to-come from root in this tree
    cost: np.ndarray
    # open / unvisited / closed are boolean masks for indices
    open_mask: np.ndarray
    unvisited_mask: np.ndarray
    closed_mask: np.ndarray


class BiFMTStar:
    """
    Bi-directional FMT* with BITStar/IRRTStar-compatible interface.

    Environment API expected (same as your IRRTStarND):
      - env._state_fp(x: np.ndarray) -> bool     # True if free
      - env._edge_fp(a: np.ndarray, b: np.ndarray) -> bool  # True if edge free
      - env.bound = [(min,max), ...] or flat(2*dim,) or (2,dim) / (dim,2)
      - env.collision_check_count (optional int)
    """

    def __init__(
        self,
        start,
        goal,
        environment,
        iter_max,
        batch_size,
        plot_flag=False,
        timer=None,
        search_radius=None,
        clearance=0.0,
        rng_seed=None,
    ):
        self.env = environment
        self.x_start = np.array(start, dtype=float).reshape(-1)
        self.x_goal = np.array(goal, dtype=float).reshape(-1)
        if self.x_start.shape != self.x_goal.shape:
            raise ValueError("start and goal must have same dimensionality")

        self.dim = int(self.x_start.size)
        self.iter_max = int(iter_max)
        self.batch_size = int(batch_size)
        self.plot_planning_process = bool(plot_flag)
        self.timer = timer  # kept for signature compatibility (unused here)
        self.clearance = float(clearance)

        # neighbor radius (FMT* uses r_n scaling; we multiply by a user knob)
        self.search_radius = float(search_radius) if search_radius is not None else 5.0

        self.bound = _normalize_bounds(getattr(self.env, "bound", None), self.dim)

        if rng_seed is not None:
            np.random.seed(int(rng_seed))

        # outputs / bookkeeping
        self.path = []
        self.samples = []
        self.edges = {}
        self.g_scores = {}

        self._internal_collision_checks = 0

    def to_key(self, point, ndigits=6):
        arr = np.round(np.array(point, dtype=float).reshape(-1), ndigits)
        return tuple(arr.tolist())

    # ---------- env wrappers ----------
    def _state_free(self, x: np.ndarray) -> bool:
        return bool(self.env._state_fp(np.asarray(x, dtype=float)))

    def _edge_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        # count checks even if env doesn't expose collision_check_count
        self._internal_collision_checks += 1
        return bool(self.env._edge_fp(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))

    # ---------- sampling ----------
    def _sample_global(self):
        lo = self.bound[:, 0] + self.clearance
        hi = self.bound[:, 1] - self.clearance
        return np.random.uniform(lo, hi)

    def _sample_free_points(self, n: int):
        pts = []
        guard = 0
        max_guard = max(1000, 50 * n)
        while len(pts) < n and guard < max_guard:
            guard += 1
            x = self._sample_global()
            if self._state_free(x):
                pts.append(x)
        return np.asarray(pts, dtype=float)

    # ---------- core helpers ----------
    @staticmethod
    def _pairwise_sq_dists(X: np.ndarray, y: np.ndarray):
        d = X - y.reshape(1, -1)
        return np.einsum("ij,ij->i", d, d)

    def _rn(self, n_total: int):
        # Classic FMT* uses r_n ~ gamma * (log n / n)^(1/d).
        # Here we keep same shape but let search_radius be the "gamma" knob.
        n = max(int(n_total), 2)
        return self.search_radius * (math.log(n) / n) ** (1.0 / float(self.dim))

    def _near_indices(self, points: np.ndarray, center: np.ndarray, radius: float):
        # return indices within radius (excluding exact same)
        r2 = float(radius * radius)
        d2 = self._pairwise_sq_dists(points, center)
        idx = np.where((d2 <= r2) & (d2 > 0.0))[0]
        return idx

    def _init_trees(self, P: np.ndarray, idx_start: int, idx_goal: int):
        N = P.shape[0]
        # tree 0: forward from start; tree 1: backward from goal
        t0 = _TreeState(
            parent=np.full(N, -1, dtype=int),
            cost=np.full(N, INF, dtype=float),
            open_mask=np.zeros(N, dtype=bool),
            unvisited_mask=np.ones(N, dtype=bool),
            closed_mask=np.zeros(N, dtype=bool),
        )
        t1 = _TreeState(
            parent=np.full(N, -1, dtype=int),
            cost=np.full(N, INF, dtype=float),
            open_mask=np.zeros(N, dtype=bool),
            unvisited_mask=np.ones(N, dtype=bool),
            closed_mask=np.zeros(N, dtype=bool),
        )

        # roots
        t0.cost[idx_start] = 0.0
        t0.open_mask[idx_start] = True
        t0.unvisited_mask[idx_start] = False

        t1.cost[idx_goal] = 0.0
        t1.open_mask[idx_goal] = True
        t1.unvisited_mask[idx_goal] = False

        return t0, t1

    def _best_open_index(self, t: _TreeState):
        idx = np.where(t.open_mask)[0]
        if idx.size == 0:
            return None, INF
        costs = t.cost[idx]
        j = int(idx[int(np.argmin(costs))])
        return j, float(t.cost[j])

    def _extract_path_indices(self, meet_idx: int, t0: _TreeState, t1: _TreeState, idx_start: int, idx_goal: int):
        # start -> meet via t0 parents
        forward = []
        cur = meet_idx
        visited = set()
        while cur != -1 and cur not in visited:
            visited.add(cur)
            forward.append(cur)
            if cur == idx_start:
                break
            cur = int(t0.parent[cur])
        if not forward or forward[-1] != idx_start:
            return []

        forward = forward[::-1]  # start..meet

        # meet -> goal via t1 parents (because t1 root is goal)
        backward = []
        cur = meet_idx
        visited = set()
        while cur != -1 and cur not in visited:
            visited.add(cur)
            backward.append(cur)
            if cur == idx_goal:
                break
            cur = int(t1.parent[cur])
        if not backward or backward[-1] != idx_goal:
            return []

        # backward currently: meet..goal, so skip meet to avoid duplication
        path = forward + backward[1:]
        return path

    def planning(self, visualize=False, refresh_interval=1):
        """
        Return signature aligned with BITStar / IRRTStarND:

        (
          path_keys,
          samples_keys,
          edges_dict(child->parent),
          n_checks,
          best_cost,
          total_samples,
          runtime,
          final_iter,
          iteration_costs,
          iteration_times,
          first_solution_iter,
          first_solution_cost,
          first_solution_nodes,
          final_solution_nodes,
        )
        """
        start_checks_env = getattr(self.env, "collision_check_count", None)
        start_checks_env = int(start_checks_env) if start_checks_env is not None else None
        start_checks_internal = int(self._internal_collision_checks)

        t0_time = time()

        # ---- build sample set ----
        # We include start/goal explicitly + (batch_size) random free samples
        X = self._sample_free_points(self.batch_size)
        P = np.vstack([self.x_start.reshape(1, -1), self.x_goal.reshape(1, -1), X]) if X.size else np.vstack(
            [self.x_start.reshape(1, -1), self.x_goal.reshape(1, -1)]
        )

        idx_start = 0
        idx_goal = 1
        N = P.shape[0]

        rn = self._rn(N)

        t0, t1 = self._init_trees(P, idx_start, idx_goal)

        meet_idx = None
        best_total = INF

        iteration_costs = []
        iteration_times = []
        first_solution_iter = None
        first_solution_cost = None
        first_solution_nodes = None

        # choose which tree to expand: pick tree whose best-open cost is smaller
        for k in range(self.iter_max):
            final_iter = k

            z0, c0 = self._best_open_index(t0)
            z1, c1 = self._best_open_index(t1)

            # stop if both open empty
            if z0 is None and z1 is None:
                break

            # termination: once we have a meet, and both open minima exceed best_total, we're "done-ish"
            # (practical stopping rule; not a strict proof here but works well)
            if meet_idx is not None:
                lb = min(c0 if np.isfinite(c0) else INF, INF) + min(c1 if np.isfinite(c1) else INF, INF)
                if np.isfinite(lb) and lb >= best_total:
                    break

            # pick tree
            if z0 is None:
                active = 1
            elif z1 is None:
                active = 0
            else:
                active = 0 if c0 <= c1 else 1

            Ta = t0 if active == 0 else t1
            Tb = t1 if active == 0 else t0
            z = z0 if active == 0 else z1
            if z is None:
                # should not happen due to checks above
                continue

            # Expand from z in active tree
            # X_near = unvisited points within rn of z
            unv_idx = np.where(Ta.unvisited_mask)[0]
            if unv_idx.size > 0:
                # candidate unvisited points
                cand = P[unv_idx]
                d2 = self._pairwise_sq_dists(cand, P[z])
                near_mask = d2 <= (rn * rn)
                X_near = unv_idx[near_mask]
            else:
                X_near = np.array([], dtype=int)

            open_idx = np.where(Ta.open_mask)[0]
            open_pts = P[open_idx] if open_idx.size > 0 else None

            new_open = []

            for x in X_near:
                # Y_near = open neighbors within rn of x
                if open_idx.size == 0:
                    continue
                d2y = self._pairwise_sq_dists(open_pts, P[x])
                Yn_mask = d2y <= (rn * rn)
                Yn = open_idx[Yn_mask]
                if Yn.size == 0:
                    continue

                # choose y_min = argmin cost[y] + dist(y,x)
                # compute distances
                dx = P[Yn] - P[x].reshape(1, -1)
                dist = np.linalg.norm(dx, axis=1)
                vals = Ta.cost[Yn] + dist
                y_min = int(Yn[int(np.argmin(vals))])

                # collision check
                if self._edge_free(P[y_min], P[x]):
                    Ta.parent[x] = y_min
                    Ta.cost[x] = float(Ta.cost[y_min] + np.linalg.norm(P[x] - P[y_min]))
                    Ta.unvisited_mask[x] = False
                    Ta.open_mask[x] = True
                    new_open.append(int(x))

                    # meeting update (if other tree has reached x)
                    if np.isfinite(Tb.cost[x]):
                        total = float(Ta.cost[x] + Tb.cost[x])
                        if total < best_total:
                            best_total = total
                            meet_idx = int(x)

                            if first_solution_iter is None:
                                first_solution_iter = k + 1
                                first_solution_cost = best_total

            # move z from open -> closed
            Ta.open_mask[z] = False
            Ta.closed_mask[z] = True

            # record curves
            cur_t = time() - t0_time
            iteration_times.append(cur_t)
            iteration_costs.append(float(best_total) if np.isfinite(best_total) else INF)

        # ---- extract path ----
        path_idx = []
        if meet_idx is not None and np.isfinite(best_total):
            path_idx = self._extract_path_indices(meet_idx, t0, t1, idx_start, idx_goal)

        if path_idx:
            path_pts = P[path_idx]
            path_keys = [self.to_key(p) for p in path_pts]
            final_solution_nodes = len(path_keys)
            if first_solution_nodes is None:
                first_solution_nodes = final_solution_nodes
        else:
            path_keys = []
            final_solution_nodes = 0

        # ---- output samples / edges ----
        samples_keys = [self.to_key(p) for p in P]

        edges = {}
        # include both trees' parents (child -> parent)
        for i in range(N):
            if t0.parent[i] != -1:
                edges[self.to_key(P[i])] = self.to_key(P[int(t0.parent[i])])
            if t1.parent[i] != -1:
                edges[self.to_key(P[i])] = self.to_key(P[int(t1.parent[i])])

        # collision checks
        end_checks_env = getattr(self.env, "collision_check_count", None)
        end_checks_env = int(end_checks_env) if end_checks_env is not None else None

        if start_checks_env is not None and end_checks_env is not None:
            n_checks = end_checks_env - start_checks_env
        else:
            n_checks = int(self._internal_collision_checks - start_checks_internal)

        runtime = time() - t0_time
        best_cost = float(best_total) if np.isfinite(best_total) else INF
        total_samples = int(N)

        # compatibility fields
        if first_solution_cost is None and np.isfinite(best_cost):
            first_solution_cost = best_cost

        # --- Make BiFMT* single-shot for fair compare ---
        iteration_costs = [best_cost]
        iteration_times = [runtime]
        final_iter = 0

        if len(path_keys) > 0 and np.isfinite(best_cost):
            first_solution_iter = 1
            first_solution_cost = best_cost
            first_solution_nodes = final_solution_nodes
        else:
            first_solution_iter = -1
            first_solution_cost = float("inf")
            first_solution_nodes = -1

        return (
            path_keys,
            samples_keys,
            edges,
            n_checks,
            best_cost,
            total_samples,
            runtime,
            final_iter,          # <- 直接用
            iteration_costs,
            iteration_times,
            first_solution_iter,
            first_solution_cost,
            first_solution_nodes,
            final_solution_nodes,
        )

# -------- Factory (BITStar / IRRTStar style) --------

def get_bifmt_planner(args, problem, neural_wrapper=None):
    """
    Expected problem keys:
      - problem["start"] / problem["goal"]  (or x_start/x_goal)
      - problem["env"] (or environment)
      - problem.get("search_radius") optional
    Expected args:
      - args.iter_max
      - args.batch_size
      - args.plot_flag (optional)
      - args.search_radius (optional override)
      - args.clearance (optional)
    """
    start = problem.get("start", problem.get("x_start"))
    goal = problem.get("goal", problem.get("x_goal"))
    env = problem.get("env", problem.get("environment"))
    if env is None:
        raise ValueError("problem must contain env/environment")

    sr = getattr(args, "search_radius", None)
    if sr is None:
        sr = problem.get("search_radius", None)

    return BiFMTStar(
        start=start,
        goal=goal,
        environment=env,
        iter_max=getattr(args, "iter_max"),
        batch_size=getattr(args, "batch_size", 2000),
        plot_flag=getattr(args, "plot_flag", False),
        timer=problem.get("timer", None) if isinstance(problem, dict) else None,
        search_radius=sr,
        clearance=getattr(args, "clearance", 0.0),
    )


# optional: keep a default alias like other planners do
get_path_planner = get_bifmt_planner
