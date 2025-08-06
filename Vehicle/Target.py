import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.Points import read_random_line_as_coordinates

class Target():
    def __init__(self, id, init_t, dt=0.5, max_speed=7.72):
        self.id = id
        self.type = 'target'
        self.T_entry = init_t
        self.T_detect = None
        self.T_capture = None
        self.T_detect_flag = False
        self.time1_flag = False

        self.dt = dt
        self.max_speed = max_speed  # 最大速度

        self.coords_list = read_random_line_as_coordinates("Vehicle/region_trajectories_start_modified.csv")
        self.current_index = 0

        self.position = self.coords_list[0]
        
    def update(self):
        # 如果轨迹结束，停止
        if self.current_index >= len(self.coords_list) - 1:
            return
        
        x1, y1 = self.position
        x2, y2 = self.coords_list[self.current_index + 1]

        dy = y2 - y1
        dx = x2 - x1
         
        angle = math.atan2(dy, dx)
        
        newx = x1 + self.max_speed * math.cos(angle) * self.dt
        newy = y1 + self.max_speed * math.sin(angle) * self.dt

        self.position = [newx, newy]
        
        if math.hypot(dx, dy) < 4:
            self.current_index += 1
            return

    def score_update(self, t, detect, capture):
        if detect:
            self.T_detect_flag += 1
        else:
            self.T_detect_flag = 0

        if self.T_detect_flag >= int(10 / self.dt) + 1:
            self.T_detect = t

        if capture:
            self.T_capture = t

    def score(self):
        time1 = None
        time2 = None
        if self.T_detect is not None:
            time1 = (self.T_detect - self.T_entry) * self.dt
        
        if self.T_capture is not None:
            time2 = (self.T_capture - self.T_entry) * self.dt
        return time1, time2
