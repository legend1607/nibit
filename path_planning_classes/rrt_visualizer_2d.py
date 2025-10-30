import os
import math
from os.path import join, exists

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as Rot


class RRTStarVisualizer:
    def __init__(self, x_start, x_goal, env):
        self.x_start, self.x_goal = x_start, x_goal
        self.env = env
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))
    
    def plot_scene_path(self, path, figure_title, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    def plot_grid(self, figure_title):
        self.fig, self.ax = plt.subplots()
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
        self.plot_start_goal()
        plt.title(figure_title)
        plt.axis("equal")

    @staticmethod
    def plot_visited(vertices, vertex_parents, animation):
        if animation:
            count = 0
            for vertex_index, vertex_parent_index in enumerate(vertex_parents):
                count += 1
                plt.plot([vertices[vertex_index, 0], vertices[vertex_parent_index, 0]],\
                         [vertices[vertex_index, 1], vertices[vertex_parent_index, 1]], "-g", lw=1)
                plt.gcf().canvas.mpl_connect('key_release_event',
                                                lambda event:
                                                [exit(0) if event.key == 'escape' else None])
                if count % 10 == 0:
                    plt.pause(0.001)
        else:
            for vertex_index, vertex_parent_index in enumerate(vertex_parents):
                plt.plot([vertices[vertex_index, 0], vertices[vertex_parent_index, 0]],\
                         [vertices[vertex_index, 1], vertices[vertex_parent_index, 1]], "-g", lw=1)

    def plot_start_goal(self):
        plt.scatter(self.x_start[0], self.x_start[1], s=30, c='r', marker='*', zorder=10)
        plt.scatter(self.x_goal[0], self.x_goal[1], s=30, c='y', marker='*', zorder=10)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot(path[:,0], path[:,1], '-r', linewidth=1, zorder=9)



class IRRTStarVisualizer(RRTStarVisualizer):
    def __init__(self, x_start, x_goal, env):
        super().__init__(x_start, x_goal, env)

    def animation(self, vertices, vertex_parents, path, figure_title,\
                  x_center, c_best, dist, theta, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation=False)
        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        if c_best ** 2 - dist ** 2<0:
            eps = 1e-6
        else:
            eps = 0
        a = math.sqrt(c_best ** 2 - dist ** 2+eps) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(px, py, linestyle='--', color='k', linewidth=1)


class NRRTStarPNGVisualizer(RRTStarVisualizer):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents, animation)
        if self.path_point_cloud_pred is not None:
            plt.scatter(self.path_point_cloud_pred[:,0], self.path_point_cloud_pred[:,1], s=2, c='C1')
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))


class NIRRTStarVisualizer(IRRTStarVisualizer):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred
        self.path_point_cloud_other = None

    def set_path_point_cloud_other(self, path_point_cloud_other):
        self.path_point_cloud_other = path_point_cloud_other

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_current_expansion(self, node):
        """在规划循环中更新当前扩展点"""
        self.current_node = node
    def set_current_node_nearest(self, node):
        """在规划循环中更新当前最近点"""
        self.current_node_nearest = node
    def set_current_expansion_new(self, node):
        """在规划循环中更新当前节点"""
        self.current_newnode = node
    def plot_current_expansion_rand(self):
        """绘制当前扩展点"""
        if self.current_node is not None:
            plt.scatter(self.current_node[0], self.current_node[1],
                        s=80, c='yellow', marker='o', label='Current Expand Node', zorder=10)
            
    def plot_current_expansion_new(self):
        """绘制当前扩展点"""
        if self.current_newnode is not None:
            plt.scatter(self.current_newnode[0], self.current_newnode[1],
                        s=80, c='purple', marker='o', label='Current Expand Node', zorder=10)
    def animation(self, vertices, vertex_parents, path, figure_title,\
                  x_center, c_best, dist, theta, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        if self.path_point_cloud_pred is not None:
            plt.scatter(self.path_point_cloud_pred[:,0], self.path_point_cloud_pred[:,1], s=2, c='C1')
        if self.path_point_cloud_other is not None:
            plt.scatter(self.path_point_cloud_other[:,0], self.path_point_cloud_other[:,1], s=2, c='C0')
        self.plot_current_expansion_rand()  # ✅ 新增当前扩展点
        self.plot_current_expansion_new()  # ✅ 新增当前扩展点
        self.plot_visited(vertices, vertex_parents, animation=False)
        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            plt.savefig(img_filename)
        plt.close()

class NIARRTStarVisualizer:
    def __init__(self, x_start, x_goal, env,
                 path_point_cloud_pred=None,
                 path_point_cloud_other=None,
                 path_score=None,
                 token_xyz=None,
                 token_kp_score=None,
                 global_trend=None):
        self.x_start, self.x_goal = x_start, x_goal
        self.env = env
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

        # 神经网络预测结果
        self.path_point_cloud_pred = path_point_cloud_pred  # segmentation 区域
        self.path_point_cloud_other = path_point_cloud_other  # 其他点云
        self.path_score = path_score  # segmentation 概率

        # 新增：记录当前扩展点与方向
        self.current_node = None
        self.pred_direction = None
        self.combined_cloud_pred = None
        self.keypoint_cloud_pred = None
        self.valid_cloud_combined = None

    # ------------------------ 动画主函数 ------------------------
    def animation(self, vertices, vertex_parents, path,
                  figure_title, x_center, c_best, dist, theta,
                  img_filename=None,
                  img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_visited(vertices, vertex_parents)
        self.plot_current_expansion_rand()  # ✅ 新增当前扩展点
        self.plot_predicted_clouds()
        self.plot_current_expansion_new()  # ✅ 新增当前扩展点
        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)
        self.plot_path(path)

        # 保存或显示
        if img_filename is None:
            plt.legend(loc="best", fontsize=8)
            plt.show()
        else:
            os.makedirs(img_folder, exist_ok=True)
            plt.legend(loc="best", fontsize=8)
            plt.savefig(img_filename, dpi=300)
        plt.close()

    # ------------------------ 可视化数据更新 ------------------------
    def set_path_point_cloud_other(self, path_point_cloud_other):
        self.path_point_cloud_other = path_point_cloud_other

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred
    def set_keypoint_cloud_pred(self, keypoint_cloud_pred):
        self.keypoint_cloud_pred = keypoint_cloud_pred
    def set_combined_cloud_pred(self, combined_cloud_pred):
        self.combined_cloud_pred = combined_cloud_pred
    def set_valid_cloud_combined(self, valid_cloud_combined):
        self.valid_cloud_combined = valid_cloud_combined

    def set_current_expansion(self, node):
        """在规划循环中更新当前扩展点"""
        self.current_node = node
    def set_current_node_nearest(self, node):
        """在规划循环中更新当前最近点"""
        self.current_node_nearest = node
    def set_current_expansion_new(self, node):
        """在规划循环中更新当前节点"""
        self.current_newnode = node
    def set_pred_direction(self, direction):
        self.pred_direction = direction

    # ------------------------ 场景绘制 ------------------------
    def plot_grid(self, figure_title):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(patches.Rectangle((ox, oy), w, h,
                                                edgecolor='black',
                                                facecolor='gray',
                                                fill=True))
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(patches.Circle((ox, oy), r,
                                             edgecolor='black',
                                             facecolor='gray',
                                             fill=True))
        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(patches.Rectangle((ox, oy), w, h,
                                                edgecolor='black',
                                                facecolor='black',
                                                fill=True))
        self.plot_start_goal()
        plt.title(figure_title)
        plt.axis("equal")

    def plot_start_goal(self):
        plt.scatter(self.x_start[0], self.x_start[1],
                    s=60, c='lime', marker='*', label='Start', zorder=10)
        plt.scatter(self.x_goal[0], self.x_goal[1],
                    s=60, c='red', marker='*', label='Goal', zorder=10)

    def plot_visited(self, vertices, vertex_parents):
        for i, p in enumerate(vertex_parents):
            if i == 0 or p < 0:
                continue
            plt.plot([vertices[i, 0], vertices[p, 0]],
                     [vertices[i, 1], vertices[p, 1]],
                     "-g", lw=0.8, alpha=0.6, zorder=2, label='_nolegend_')

    def plot_path(self, path):
        if len(path) > 0:
            plt.plot(path[:, 0], path[:, 1], '-r', lw=2.5, label='Final Path', zorder=5)

    # ------------------------ 关键新增部分 ------------------------
    def plot_current_expansion_rand(self):
        """绘制当前采样点和方向向量"""
        if self.current_node is not None:
            plt.scatter(self.current_node[0], self.current_node[1],
                        s=80, c='yellow', marker='o', label='Current Expand Node', zorder=10)
        if self.current_node is not None and self.pred_direction is not None:
            plt.arrow(self.current_node_nearest[0], self.current_node_nearest[1],
                      10 * self.pred_direction[0], 10 * self.pred_direction[1],
                      head_width=0.05, head_length=0.08,
                      fc='orange', ec='orange', lw=2, label='Predicted Direction', zorder=9)
    def plot_current_expansion_new(self):
        """绘制当前扩展点"""
        if self.current_newnode is not None:
            plt.scatter(self.current_newnode[0], self.current_newnode[1],
                        s=80, c='purple', marker='o', label='Current Expand Node', zorder=10)

    # ------------------------ 其他已有可视化 ------------------------
    def plot_predicted_clouds(self):
        if self.path_point_cloud_other is not None:
            plt.scatter(self.path_point_cloud_other[:, 0],
                        self.path_point_cloud_other[:, 1],
                        s=6, c='lightgray', label='Free Space', alpha=0.4, zorder=1)
        if self.valid_cloud_combined is not None:
            plt.scatter(self.valid_cloud_combined[:, 0],
                        self.valid_cloud_combined[:, 1],
                        s=10, c='brown', label='Predicted Keypoints&pathpoints', alpha=0.8, zorder=3)
            self.valid_cloud_combined=None
        elif self.combined_cloud_pred is not None:
            plt.scatter(self.combined_cloud_pred[:, 0],
                        self.combined_cloud_pred[:, 1],
                        s=10, c='purple', label='Predicted Keypoints&pathpoints', alpha=0.8, zorder=3)
            self.combined_cloud_pred=None
        else:
            if self.path_point_cloud_pred is not None:
                plt.scatter(self.path_point_cloud_pred[:, 0],
                            self.path_point_cloud_pred[:, 1],
                            s=10, c='orange', label='Predicted High-Value Points', alpha=0.8, zorder=3)
            if self.keypoint_cloud_pred is not None:
                plt.scatter(self.keypoint_cloud_pred[:, 0],
                            self.keypoint_cloud_pred[:, 1],
                            s=10, c='blue', label='Predicted Keypoints', alpha=0.8, zorder=3)

    def draw_ellipse(self, x_center, c_best, dist, theta):
        if c_best == np.inf:
            return
        a = np.sqrt(c_best ** 2 / 4.0)
        b = np.sqrt(c_best ** 2 / 4.0 - dist ** 2 / 4.0)
        ellipse = patches.Ellipse(x_center, 2 * a, 2 * b, np.degrees(theta),
                                  edgecolor='blue', fc='none', lw=1.2, ls='--', label='Informed Region')
        self.ax.add_patch(ellipse)

class NRRTStarGNGVisualizer(RRTStarVisualizer):
    def __init__(self, x_start, x_goal, env, path_point_cloud_pred=None, img_path_score=None):
        super().__init__(x_start, x_goal, env)
        self.path_point_cloud_pred = path_point_cloud_pred
        self.img_path_score = img_path_score

    def set_path_point_cloud_pred(self, path_point_cloud_pred):
        self.path_point_cloud_pred = path_point_cloud_pred

    def set_img_path_score(self, img_path_score):
        self.img_path_score = img_path_score

    def plot_prob_heatmap(self):
        if self.img_path_score is not None:
            self.ax.imshow(self.img_path_score, cmap='viridis', zorder=0)

    def animation(self, vertices, vertex_parents, path, figure_title, animation=False, img_filename=None, img_folder='visualization/planning_demo'):
        self.plot_grid(figure_title)
        self.plot_prob_heatmap()
        self.plot_visited(vertices, vertex_parents, animation)
        # if self.path_point_cloud_pred is not None:
        #     plt.scatter(self.path_point_cloud_pred[:,0], self.path_point_cloud_pred[:,1], s=2, c='C1')
        self.plot_path(path)
        if img_filename is None:
            plt.show()
        else:
            if not exists(img_folder):
                os.makedirs(img_folder, exist_ok=True)
            plt.savefig(join(img_folder, img_filename))