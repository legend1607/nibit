import numpy as np
from path_planning_classes.collision_check_utils import check_collision_line_circles_rectangles, points_in_circles_rectangles, points_in_range, points_validity

class Utils:
    def __init__(self, env, clearance):
        self.env = env
        self.clearance = clearance
        if len(self.env.obs_circle)>0:
            self.obs_circle = np.array(self.env.obs_circle)
        else:
            self.obs_circle = None
        if len(self.env.obs_rectangle)>0:
            self.obs_rectangle = np.array(self.env.obs_rectangle)
        else:
            self.obs_rectangle = None
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

    def is_collision(self, start, end, step=1.0):
        """
        更稳健的碰撞检测：对长线段分段采样，逐段调用 check_collision_line_circles_rectangles
        - step: 每段长度，越小越精细，越安全
        """
        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)
        dist = np.linalg.norm(end - start)

        n_steps = max(1, int(dist / step))  # 至少 1 段
        for i in range(n_steps):
            p1 = start + (i / n_steps) * (end - start)
            p2 = start + ((i + 1) / n_steps) * (end - start)
            line = np.array([p1, p2])
            if check_collision_line_circles_rectangles(
                line,
                self.obs_circle,
                self.obs_rectangle,
                self.clearance
            ):
                return True
        return False

    def is_inside_obs(self, node):
        """
        - inputs:
            - node: tuple (2,) or np (2,)
        - outputs:
            - in_obstacle: bool.
        """
        return points_in_circles_rectangles(
            (node[0], node[1]),
            self.obs_circle,
            self.obs_rectangle,
            self.clearance,
        )
    
    def is_in_range(self, node):
        """
        - inputs:
            - node: tuple (2,) or np (2,)
        - outputs:
            - in_range: bool.
        """
        return points_in_range(
            (node[0], node[1]),
            self.x_range,
            self.y_range,
            self.clearance,
        )

    def is_valid(self, node):
        """
        Check both in range and not in obstacle.
        - inputs:
            - node: tuple (2,) or np (2,)
        - outputs:
            - validity: bool.
        """
        return points_validity(
            (node[0], node[1]),
            self.obs_circle,
            self.obs_rectangle,
            self.x_range,
            self.y_range,
            obstacle_clearance=self.clearance,
            range_clearance=self.clearance,
        )
