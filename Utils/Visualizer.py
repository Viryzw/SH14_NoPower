import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 中文支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 中文支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class TrajectoryVisualizer2D:
    def __init__(self,
                 labels,
                 point_colors=None,
                 line_colors=None,
                 line_styles=None,
                 markers=None,
                 trajectory_visible=None,
                 limits=((-10, 10), (-10, 10)),
                 label_fontsize=12,
                 title_fontsize=14,
                 legend_fontsize=12,
                 tick_fontsize=12,
                 enable_swept_area=False,
                 swept_area_start_time=300,
                 max_history_fov=500,
                 max_traj=2000):
        
        self.labels = labels
        self.num = len(labels)
        self.history = {label: [] for label in labels}

        self.point_colors = point_colors or ['b'] * self.num
        self.line_colors = line_colors or ['gray'] * self.num
        self.line_styles = line_styles or ['-'] * self.num
        self.markers = markers or ['o'] * self.num
        self.trajectory_visible = trajectory_visible or [True] * self.num

        self.limits = limits
        self.label_fontsize = label_fontsize
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        self.tick_fontsize = tick_fontsize

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self._setup_plot()
        
        self.swept_fovs = {label: [] for label in self.labels}

        # 可控参数
        self.enable_swept_area = enable_swept_area
        self.swept_area_start_time = swept_area_start_time
        self.MAX_HISTORY_FOV = max_history_fov
        self.MAX_TRAJ = max_traj

        self.time_text = None  # 新增：存储仿真时间文本对象

    def _setup_plot(self):
        self.ax.set_xlim(*self.limits[0])
        self.ax.set_ylim(*self.limits[1])
        self.ax.set_xlabel("X", fontsize=self.label_fontsize)
        self.ax.set_ylabel("Y", fontsize=self.label_fontsize)
        self.ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')

    def update(self, objects, step=None, title=None):
        assert len(objects) == self.num, "对象数量必须与标签数量匹配"

        self.ax.cla()
        self._setup_plot()

        if title:
            self.ax.set_title(title, fontsize=self.title_fontsize)
        elif step is not None:
            self.ax.set_title(f"Step {step}", fontsize=self.title_fontsize)
            
        # ==== 新增：显示仿真时间 ====
        # 假定每步0.5秒（如有不同请自行调整）
        if step is not None:             
            sim_time = step * 0.5  # 单位：秒
            # 移除上一次的时间文本
            if self.time_text is not None:
                try:
                    self.time_text.remove()
                except Exception:
                    pass
            # 左上角显示（坐标系用axes百分比）
            self.time_text = self.ax.text(
                0.01, 0.99,
                f"Time: {sim_time:.1f}s",
                transform=self.ax.transAxes,
                fontsize=14,
                verticalalignment='top',
                horizontalalignment='left',
                color='k',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round')
            )
        # =========================

        for label in self.labels:
            if label not in self.swept_fovs:
                self.swept_fovs[label] = []
        
        enable_history = (
            self.enable_swept_area and
            step is not None and
            (step if isinstance(step, (list, tuple)) else step) * 0.5 >= self.swept_area_start_time
        )
        
        if enable_history:
            for i, label in enumerate(self.labels):
                for fov_param in self.swept_fovs[label]:
                    self.draw_fov(*fov_param, color='lightblue', alpha=0.25)

        for i, obj in enumerate(objects):
            label = self.labels[i]
            pos = np.array(obj.position)
            self.history[label].append(pos)

            # 画轨迹（如果开启）
            if self.trajectory_visible[i]:
                traj = np.array(self.history[label])
                self.ax.plot(traj[:, 0], traj[:, 1],
                             color=self.line_colors[i],
                             linestyle=self.line_styles[i],
                             linewidth=1.2)

            # 画当前点
            self.ax.plot(pos[0], pos[1],
                         marker=self.markers[i],
                         color=self.point_colors[i],
                         markersize=8,
                         label=label)

            # 探测区域绘制
            if hasattr(obj, 'sensor_type') and hasattr(obj, 'detection_range'):
                if obj.sensor_type == 'fov' and hasattr(obj, 'fov_rad') and hasattr(obj, 'vehicle_heading'):
                    # 兼容 vehicle_heading 是角度（float）或向量
                    heading = obj.vehicle_heading
                    if isinstance(heading, (float, int)):
                        # 角度转单位向量（假设是弧度）
                        heading_vec = np.array([np.cos(heading), np.sin(heading)])
                    else:
                        heading_vec = heading
                        
                    # 只在累计区域功能打开且到达指定时间后才累计，每10步记录一次
                    current_step = step if isinstance(step, (list, tuple)) else step
                    if enable_history and (step is not None) and (current_step % 10 == 0):
                        if label not in self.swept_fovs:
                            self.swept_fovs[label] = []
                        self.swept_fovs[label].append(
                            (pos.copy(), heading_vec.copy(), obj.fov_rad, obj.detection_range)
                        )
                        if len(self.swept_fovs[label]) > self.MAX_HISTORY_FOV:
                            self.swept_fovs[label] = self.swept_fovs[label][-self.MAX_HISTORY_FOV:]

                    self.draw_fov(position=pos,
                                  heading=heading_vec,
                                  fov_rad=obj.fov_rad,
                                  detection_range=obj.detection_range)
                elif obj.sensor_type == 'circle':
                    self.draw_circle(position=pos,
                                     radius=obj.detection_range)
        # TODO：
        # self.ax.legend(fontsize=self.legend_fontsize)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_fov(self, position, heading, fov_rad, detection_range, color='orange', alpha=0.2):
        theta_deg = np.rad2deg(np.arctan2(heading[1], heading[0]))
        fov_deg = np.rad2deg(fov_rad) * 2
        wedge = patches.Wedge(center=position,
                              r=detection_range,
                              theta1=theta_deg - fov_deg / 2,
                              theta2=theta_deg + fov_deg / 2,
                              facecolor=color,
                              edgecolor=color,
                              alpha=alpha)
        self.ax.add_patch(wedge)

    def draw_circle(self, position, radius, color='cyan', alpha=0.15):
        circle = patches.Circle(position,
                                radius,
                                facecolor=color,
                                edgecolor='blue',
                                alpha=alpha)
        self.ax.add_patch(circle)

    def close(self):
        plt.ioff()
        plt.show()