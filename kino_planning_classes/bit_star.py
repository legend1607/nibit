# kino_bit_star.py
import numpy as np
import math
import heapq
from time import time
import matplotlib.pyplot as plt
from environment.timer import Timer

INF = float("inf")


class KinoBITStar:
    """
    KinoBITStar: Kinematic-constrained BIT* for states [q, dq]
    - Expects env to implement:
        - dof, bound, sample_state(), sample_empty_points(), _state_fp(q)
        - kin_reachable(x_from, x_to), kin_edge_free(x_from, x_to)
      Optionally:
        - sample_physical_pair() -> (x_from, x_to) that satisfy dynamics
    """

    def __init__(self,
                 start,
                 goal,
                 environment,
                 iter_max=200,
                 batch_size=256,
                 eta=1.1,
                 alpha=1.0,
                 beta=0.3,
                 plot_flag=False,
                 timer=None):
        self.env = environment
        self.timer = timer or Timer()

        # state geometry
        self.dof = int(getattr(self.env, "dof", getattr(self.env, "config_dim", 2)))
        self.state_dim = 2 * self.dof
        try:
            self.bounds = np.array(self.env.bound).reshape((2, -1)).T
        except Exception:
            self.bounds = np.vstack([np.full(self.dof, -np.pi), np.full(self.dof, np.pi)]).T

        # start/goal keys (to_key pads velocities)
        self.start = self.to_key(start)
        self.goal = self.to_key(goal)

        # algorithm params
        self.iter_max = int(iter_max)
        self.batch_size = int(batch_size)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.plot_flag = bool(plot_flag)

        # data structures
        self.vertices = []       # list of keys
        self.edges = {}          # child_key -> parent_key
        self.g_scores = {}       # key -> g
        self.samples = []        # list of state keys
        self.vertex_queue = []   # heap of (value, key)
        self.edge_queue = []     # heap of (value, (u,v))
        self.old_vertices = set()

        # sampling/informed
        self.c_min = self.distance_q(self.start, self.goal)
        self.center_point = None
        self.C = None

        # runtime vars
        self.r = INF
        self.T = 0
        self.path = []
        self._eps = 1e-12

    # ----------------------- helpers -----------------------
    def to_key(self, state, ndigits=6):
        arr = np.array(state, dtype=float).flatten()
        if arr.size < self.state_dim:
            arr = np.hstack([arr, np.zeros(self.state_dim - arr.size)])
        elif arr.size > self.state_dim:
            arr = arr[:self.state_dim]
        arr = np.round(arr, ndigits)
        return tuple(arr.tolist())

    def state_arr(self, key):
        return np.array(key, dtype=float)

    def distance_q(self, a, b):
        a = np.array(a, dtype=float)[:self.dof]
        b = np.array(b, dtype=float)[:self.dof]
        return np.linalg.norm(a - b)

    def heuristic_cost(self, a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        pos = np.linalg.norm(a[:self.dof] - b[:self.dof])
        vel = np.linalg.norm(a[self.dof:] - b[self.dof:])
        return self.alpha * pos + self.beta * vel

    def get_g_score(self, x):
        if x == self.start:
            return 0.0
        return float(self.g_scores.get(x, INF))

    def get_point_value(self, x):
        return self.get_g_score(x) + self.heuristic_cost(x, self.goal)

    def get_edge_value(self, edge):
        a, b = edge
        return self.get_g_score(a) + self.heuristic_cost(a, b) + self.heuristic_cost(b, self.goal)

    def radius_init(self):
        from scipy import special
        n = max(1, self.state_dim)
        unit_ball_vol = np.pi ** (n / 2) / max(self._eps, special.gamma(n / 2 + 1))
        try:
            vol = np.prod(np.maximum(self.bounds[:, 1] - self.bounds[:, 0], self._eps))
        except Exception:
            vol = 1.0
        gamma = (1.0 + 1.0 / n) * vol / max(self._eps, unit_ball_vol)
        return 2 * self.eta * (gamma ** (1.0 / n))

    # ------------------- env wrappers -------------------
    def kin_reachable(self, x_from, x_to):
        if hasattr(self.env, "kin_reachable"):
            return bool(self.env.kin_reachable(x_from, x_to))
        # conservative fallback
        return self.distance_q(x_from, x_to) < 1.0

    def kin_edge_free(self, x_from, x_to):
        if hasattr(self.env, "kin_edge_free"):
            return bool(self.env.kin_edge_free(x_from, x_to))
        q1 = np.array(x_from[:self.dof], dtype=float)
        q2 = np.array(x_to[:self.dof], dtype=float)
        if hasattr(self.env, "_edge_fp"):
            return self.env._edge_fp(q1, q2)
        # coarse check
        steps = max(2, int(np.ceil(np.linalg.norm(q2 - q1) / 0.5)))
        for i in range(steps + 1):
            pt = q1 + (q2 - q1) * (i / steps)
            if hasattr(self.env, "_point_in_free_space"):
                if not self.env._point_in_free_space(pt):
                    return False
        return True

    # ------------------- sampling -------------------
    def informed_sample_init(self):
        q_start = np.array(self.start[:self.dof], dtype=float)
        q_goal = np.array(self.goal[:self.dof], dtype=float)
        center_q = (q_start + q_goal) / 2.0
        dq_center = np.zeros(self.dof)
        self.center_point = np.hstack([center_q, dq_center])

        a1 = q_goal - q_start
        if np.linalg.norm(a1) < 1e-8:
            self.C = np.eye(self.state_dim)
            return
        a1 = a1 / np.linalg.norm(a1)
        dim_q = max(2, self.dof)
        X = np.random.randn(dim_q, dim_q)
        X[:, 0] = np.hstack([a1, np.zeros(dim_q - a1.shape[0])]) if a1.shape[0] != dim_q else a1
        Q, _ = np.linalg.qr(X)
        if np.dot(Q[:, 0], a1[:dim_q]) < 0:
            Q[:, 0] *= -1
        C = np.eye(self.state_dim)
        C[:dim_q, :dim_q] = Q
        self.C = C

    def sample_unit_ball(self):
        u = np.random.normal(0, 1, self.state_dim)
        norm = np.linalg.norm(u)
        if norm < self._eps:
            return self.sample_unit_ball()
        r = np.random.random() ** (1.0 / max(1, self.state_dim))
        return r * u / norm

    def sample_from_env(self, c_best, batch_size):
        """
        Returns list of state keys (single states).
        If env provides sample_state(), use it.
        Also attempts informed ellipse sampling when possible.
        """
        samples = []
        if c_best is None or not np.isfinite(c_best) or self.C is None:
            for _ in range(batch_size):
                if hasattr(self.env, "sample_state"):
                    s = self.env.sample_state()
                else:
                    q = self.env.sample_empty_points() if hasattr(self.env, "sample_empty_points") else np.zeros(self.dof)
                    dq = np.zeros(self.dof)
                    s = np.hstack([q[:self.dof], dq])
                samples.append(self.to_key(s))
            return samples

        # informed ellipse for q-subspace + random dq
        a = float(c_best) / 2.0
        temp = max(0.0, c_best**2 - (self.c_min ** 2 if self.c_min is not None else 0.0))
        b = math.sqrt(temp) / 2.0 if temp > 0 else self._eps
        q_diag = [a] + [b] * (self.dof - 1)
        L = np.diag(q_diag + [1.0] * self.dof)

        attempts = 0
        max_attempts = max(1000, batch_size * 50)
        while len(samples) < batch_size and attempts < max_attempts:
            attempts += 1
            x_ball = self.sample_unit_ball()
            x = (self.C @ (L @ x_ball)) + (self.center_point if self.center_point is not None else np.zeros(self.state_dim))
            q_part = x[:self.dof]
            try:
                if hasattr(self.env, "_state_fp"):
                    if self.env._state_fp(q_part):
                        # assign dq if not meaningful in x
                        if np.allclose(x[self.dof:], 0):
                            if hasattr(self.env, "vmax"):
                                dq = np.random.uniform(-self.env.vmax, self.env.vmax)
                            else:
                                dq = np.zeros(self.dof)
                            x[self.dof:] = dq
                        samples.append(self.to_key(x))
                else:
                    samples.append(self.to_key(x))
            except Exception:
                # fallback uniform sample
                if hasattr(self.env, "sample_state"):
                    samples.append(self.to_key(self.env.sample_state()))
                else:
                    q = self.env.sample_empty_points() if hasattr(self.env, "sample_empty_points") else np.zeros(self.dof)
                    samples.append(self.to_key(np.hstack([q[:self.dof], np.zeros(self.dof)])))
        # pad if necessary
        while len(samples) < batch_size:
            if hasattr(self.env, "sample_state"):
                samples.append(self.to_key(self.env.sample_state()))
            else:
                q = self.env.sample_empty_points() if hasattr(self.env, "sample_empty_points") else np.zeros(self.dof)
                samples.append(self.to_key(np.hstack([q[:self.dof], np.zeros(self.dof)])))
        return samples

    def sample_physical_pairs_into_queue(self, n_pairs=16):
        """
        If env provides sample_physical_pair(), use it to generate physical edges.
        Insert both nodes into samples and push edge candidates to edge_queue.
        """
        if not hasattr(self.env, "sample_physical_pair"):
            return 0
        added = 0
        for _ in range(n_pairs):
            try:
                x_from, x_to = self.env.sample_physical_pair()
                k_from = self.to_key(x_from)
                k_to = self.to_key(x_to)
                # ensure nodes present
                if k_from not in self.samples and k_from not in self.vertices:
                    self.samples.append(k_from)
                if k_to not in self.samples and k_to not in self.vertices:
                    self.samples.append(k_to)
                # compute priority and push edge
                val = self.get_edge_value((k_from, k_to))
                heapq.heappush(self.edge_queue, (val, (k_from, k_to)))
                added += 1
            except Exception:
                continue
        return added

    # ---------------- core algorithm ----------------
    def setup_planning(self):
        if self.goal not in self.samples:
            self.samples.append(self.goal)
        self.g_scores[self.goal] = INF
        if self.start not in self.vertices:
            self.vertices.append(self.start)
        self.g_scores[self.start] = 0.0
        self.informed_sample_init()
        return self.radius_init()

    def bestVertexQueueValue(self):
        return self.vertex_queue[0][0] if self.vertex_queue else INF

    def bestEdgeQueueValue(self):
        return self.edge_queue[0][0] if self.edge_queue else INF

    def expand_vertex(self, v):
        # neighbors among samples within radius r (q-distance)
        neigh = [s for s in self.samples if self.distance_q(v, s) <= self.r]
        for nb in neigh:
            try:
                est_f = self.get_g_score(v) + self.heuristic_cost(v, nb) + self.heuristic_cost(nb, self.goal)
                if est_f < self.get_g_score(self.goal):
                    if self.kin_reachable(v, nb):
                        heapq.heappush(self.edge_queue, (self.get_edge_value((v, nb)), (v, nb)))
            except Exception:
                continue

        # neighbors among existing vertices (for potential rewiring edges)
        if v not in self.old_vertices:
            neigh_v = [u for u in self.vertices if self.distance_q(v, u) <= self.r]
            for nb in neigh_v:
                if nb not in self.edges or v != self.edges.get(nb):
                    try:
                        est_f = self.get_g_score(v) + self.heuristic_cost(v, nb) + self.heuristic_cost(nb, self.goal)
                        if est_f < self.get_g_score(self.goal):
                            est_g = self.get_g_score(v) + self.heuristic_cost(v, nb)
                            if est_g < self.get_g_score(nb):
                                if self.kin_reachable(v, nb):
                                    heapq.heappush(self.edge_queue, (self.get_edge_value((v, nb)), (v, nb)))
                    except Exception:
                        continue

    def rewire(self, v_new):
        neighbors = [v for v in self.vertices if v != v_new and self.distance_q(v, v_new) <= self.r]
        for nb in neighbors:
            try:
                if not self.kin_reachable(v_new, nb):
                    continue
                if not self.kin_edge_free(v_new, nb):
                    continue
                new_cost = self.get_g_score(v_new) + self.heuristic_cost(v_new, nb)
                if new_cost + 1e-12 < self.get_g_score(nb):
                    self.g_scores[nb] = new_cost
                    self.edges[nb] = v_new
                    # push nb into vertex queue for further expansion
                    heapq.heappush(self.vertex_queue, (self.get_point_value(nb), nb))
            except Exception:
                continue

    def get_best_path(self):
        path = []
        if self.get_g_score(self.goal) < INF:
            p = self.goal
            path.append(p)
            safety = 0
            while p != self.start and safety < 10000:
                p = self.edges.get(p, None)
                if p is None:
                    break
                path.append(p)
                safety += 1
            path.reverse()
        return path

    def planning(self, visualize=False, refresh_interval=10):
        self.setup_planning()
        init_time = time()
        iteration_costs = []
        final_iter = 0

        if visualize:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title("Kino-BIT* Path Planning")

        for k in range(self.iter_max):
            final_iter = k

            # if both queues empty -> produce new batch
            if (not self.vertex_queue) and (not self.edge_queue):
                c_best = self.get_g_score(self.goal)
                # try physical pairs first if env provides them
                if hasattr(self.env, "sample_physical_pair"):
                    added_pairs = self.sample_physical_pairs_into_queue(min( max(4, self.batch_size // 8), 64))
                else:
                    added_pairs = 0

                new_samples = self.sample_from_env(c_best, self.batch_size)
                self.samples.extend(new_samples)
                self.T += len(new_samples)

                self.old_vertices = set(self.vertices)
                self.vertex_queue = [(self.get_point_value(v), v) for v in self.vertices]
                heapq.heapify(self.vertex_queue)
                q = len(self.vertices) + len(self.samples)
                if q > 0:
                    self.r = self.radius_init() * ((math.log(max(2, q)) / max(1, q)) ** (1.0 / max(1, self.state_dim)))
                else:
                    self.r = self.radius_init()

            # expand vertices
            while self.vertex_queue and (self.bestVertexQueueValue() <= self.bestEdgeQueueValue()):
                try:
                    _, v = heapq.heappop(self.vertex_queue)
                except IndexError:
                    break
                self.expand_vertex(v)

            # process best edge
            if not self.edge_queue:
                iteration_costs.append(self.get_g_score(self.goal))
                if visualize and k % refresh_interval == 0:
                    self._plot(ax, k)
                continue

            try:
                _, (u, v) = heapq.heappop(self.edge_queue)
            except IndexError:
                iteration_costs.append(self.get_g_score(self.goal))
                continue

            feasible = False
            try:
                if self.kin_reachable(u, v) and self.kin_edge_free(u, v):
                    feasible = True
            except Exception:
                feasible = False

            if feasible:
                new_g = self.get_g_score(u) + self.heuristic_cost(u, v)
                if new_g + 1e-12 < self.get_g_score(v):
                    self.g_scores[v] = new_g
                    self.edges[v] = u
                    added_new = False
                    if v not in self.vertices:
                        self.vertices.append(v)
                        try:
                            self.samples.remove(v)
                        except ValueError:
                            pass
                        heapq.heappush(self.vertex_queue, (self.get_point_value(v), v))
                        added_new = True
                    # prune inconsistent edge entries referencing v
                    self.edge_queue = [item for item in self.edge_queue if not (item[1][1] == v and (self.get_g_score(item[1][0]) + self.heuristic_cost(item[1][0], v) >= self.get_g_score(item[1][0])))]
                    heapq.heapify(self.edge_queue)
                    if added_new:
                        try:
                            self.rewire(v)
                        except Exception:
                            pass

            self.path = self.get_best_path()
            iteration_costs.append(self.get_g_score(self.goal))

            if visualize and k % refresh_interval == 0:
                self._plot(ax, k)

        total_time = time() - init_time
        return (self.path, self.samples, self.edges,
                self.get_g_score(self.goal), self.T, total_time,
                final_iter, iteration_costs)

    # ---------------- visualization ----------------
    def _plot(self, ax, k):
        ax.clear()
        ax.set_title(f"Kino-BIT* Iter {k}")
        for rx, ry, rw, rh in getattr(self.env, "rect_obstacles", []):
            ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='gray', alpha=0.5))
        for cx, cy, r in getattr(self.env, "circle_obstacles", []):
            ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))

        # draw edges
        for child, parent in self.edges.items():
            try:
                p = np.array(parent[:self.dof])
                c = np.array(child[:self.dof])
                ax.plot([p[0], c[0]], [p[1], c[1]], c='skyblue', lw=0.6)
            except Exception:
                continue

        # draw samples
        if self.samples:
            samples_q = np.array([s[:self.dof] for s in self.samples])
            if samples_q.shape[1] >= 2:
                ax.scatter(samples_q[:, 0], samples_q[:, 1], c='lightgray', s=4)

        # draw path
        if len(self.path) > 1:
            path_q = np.array([p[:self.dof] for p in self.path])
            ax.plot(path_q[:, 0], path_q[:, 1], 'r-', lw=2)

        s = np.array(self.start[:self.dof])
        g = np.array(self.goal[:self.dof])
        ax.plot(s[0], s[1], 'go', markersize=6)
        ax.plot(g[0], g[1], 'ro', markersize=6)
        plt.pause(0.001)


# factory
def get_bit_planner(args, problem, neural_wrapper=None):
    planner = KinoBITStar(
        start=problem["start"],
        goal=problem["goal"],
        environment=problem['env'],
        iter_max=getattr(args, "iter_max", 200),
        batch_size=getattr(args, "batch_size", 256),
        eta=getattr(args, "eta", 1.1),
        alpha=getattr(args, "alpha", 1.0),
        beta=getattr(args, "beta", 0.3),
        plot_flag=getattr(args, "plot_flag", False),
    )
    return planner
