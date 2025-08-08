import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Vehicle.Uav import Uav
from Vehicle.Usv import Usv
from Vehicle.Target import Target
from Utils.Visualizer import TrajectoryVisualizer2D

class Manager:
    def __init__(self):
        self.vehicles = {}  # (type,id) -> object
        self.targets = {}   # id -> object
        self.captured = set()
        self.detected = set()
        self.visualizer = None
        self.labels = []
        self.point_colors = []
        self.line_colors = []
        self.markers = []
        self.trajectory_visible = []
        self.time1 = []
        self.time2 = []
        self.enable_swept_area = False      # uav历史探测区域绘制开关
        self.swept_area_start_time = 300    # 开始绘制uav历史区域的时刻(s)
        

    def init_uavs(self, uav_list):
        for id_, pos, theta in uav_list:
            id_ = str(id_)
            obj = Uav(init_position=pos, init_theta=theta, id=id_)
            self.vehicles[("uav", id_)] = obj
            self.labels.append(f"UAV{id_}")
            self.point_colors.append("r")
            self.line_colors.append("r")
            self.markers.append("o")
            self.trajectory_visible.append(False)

    def init_usvs(self, usv_list):
        for id_, pos, theta in usv_list:
            id_ = str(id_)
            obj = Usv(init_position=pos, init_theta=theta, id=id_)
            self.vehicles[("usv", id_)] = obj
            self.labels.append(f"USV{id_}")
            self.point_colors.append("b")
            self.line_colors.append("b")
            self.markers.append("s")
            self.trajectory_visible.append(False)

    # def init_targets(self, target_list, t):
    #     for id_ in target_list:
    #         id_ = str(id_)
    #         obj = Target(id=id_, init_t=t)
    #         self.targets[id_] = obj
    #         self.labels.append(f"Target{id_}")
    #         self.point_colors.append("k")
    #         self.line_colors.append("k")
    #         self.markers.append("x")
    #         self.trajectory_visible.append(False)
            
    def add_targets(self, new_target_list, t):
        for id_ in new_target_list:
            id_ = str(id_)
            obj = Target(id=id_, init_t=t)
            self.targets[id_] = obj
            self.labels.append(f"Target{id_}")
            self.point_colors.append("k")
            self.line_colors.append("k")
            self.markers.append("x")
            self.trajectory_visible.append(False)
            
            self.visualizer.labels.append(f"Target{id_}")
            self.visualizer.point_colors.append("k")
            self.visualizer.line_colors.append("k")
            self.visualizer.markers.append("x")
            self.visualizer.trajectory_visible.append(False)
            self.visualizer.history[f"Target{id_}"] = []
        

    def init_objects(self, uavs, usvs):
        self.init_uavs(uavs)
        self.init_usvs(usvs)
        # self.init_targets(targets, t)

        self.visualizer = TrajectoryVisualizer2D(
            labels=self.labels,
            point_colors=self.point_colors,
            line_colors=self.line_colors,
            markers=self.markers,
            trajectory_visible=self.trajectory_visible,
            limits=((0, 9260), (0, 9260)),
            enable_swept_area=self.enable_swept_area,
            swept_area_start_time=self.swept_area_start_time
        )

    def sync_visualizer_targets(self):
        # 清理visualizer和Manager自身的标签属性，去除已被捕获删除的Target标签
        new_labels = []
        new_point_colors = []
        new_line_colors = []
        new_markers = []
        new_trajectory_visible = []

        new_labels_manager = []
        new_point_colors_manager = []
        new_line_colors_manager = []
        new_markers_manager = []
        new_trajectory_visible_manager = []

        for i, label in enumerate(self.labels):
            if label.startswith("Target"):
                tid = label[len("Target"):]
                if tid not in self.targets:
                    continue  # 被捕获移除的目标，跳过
            # 保留
            new_labels.append(label)
            new_point_colors.append(self.point_colors[i])
            new_line_colors.append(self.line_colors[i])
            new_markers.append(self.markers[i])
            new_trajectory_visible.append(self.trajectory_visible[i])
            new_labels_manager.append(label)
            new_point_colors_manager.append(self.point_colors[i])
            new_line_colors_manager.append(self.line_colors[i])
            new_markers_manager.append(self.markers[i])
            new_trajectory_visible_manager.append(self.trajectory_visible[i])

        # 更新visualizer属性
        self.visualizer.labels = new_labels
        self.visualizer.point_colors = new_point_colors
        self.visualizer.line_colors = new_line_colors
        self.visualizer.markers = new_markers
        self.visualizer.trajectory_visible = new_trajectory_visible

        # 更新Manager自身属性
        self.labels = new_labels_manager
        self.point_colors = new_point_colors_manager
        self.line_colors = new_line_colors_manager
        self.markers = new_markers_manager
        self.trajectory_visible = new_trajectory_visible_manager

        # 更新visualizer对象的数量
        self.visualizer.num = len(new_labels)

    def update(self, control_info, t, mode):
        # 更新车辆和目标状态
        for typ, id_, v, omega in control_info:
            id_ = str(id_)
            if typ.lower() in ("uav", "usv"):
                vehicle = self.vehicles.get((typ.lower(), id_))
                if vehicle:
                    vehicle.update(v, omega,
                                uavs=[v for (t, _), v in self.vehicles.items() if t == "uav"],
                                usvs=[v for (t, _), v in self.vehicles.items() if t == "usv"],
                                targets=list(self.targets.values()))
                else:
                    print(f"未找到车辆 {typ} {id_}")
            elif typ.lower() == "target":
                target = self.targets.get(id_)
                if target:
                    target.update(mode)

        # 计分更新
        for target in self.targets.values():
            detect = any(target.id in (d[1] if isinstance(d, tuple) else d)
                         for vehicle in self.vehicles.values()
                         if hasattr(vehicle, 'detected') and vehicle.detected
                         for d in vehicle.detected)
            
            capture = any(
                target.id == e[1]
                for vehicle in self.vehicles.values()
                for e in getattr(vehicle, 'captured', []))

            target.score_update(t=t, detect=detect, capture=capture)
            if target.T_detect is not None and target.time1_flag == False:
                time1, time2 = target.score()
                self.time1.append(time1)
                target.time1_flag = True

        # 检查捕获目标，删除
        to_remove = set()
        for (typ, id_), vehicle in self.vehicles.items():
            if hasattr(vehicle, 'captured') and vehicle.captured:
                for (target_type, tid, pos) in vehicle.captured:
                    if tid in self.targets:
                        to_remove.add(tid)
        for tid, target in self.targets.items():
            x, y = target.position
            if not (0 <= x <= 9260 and 0 <= y <= 9260):
                to_remove.add(tid)
                            
        for rid in to_remove:
            target_obj = self.targets[rid]
            time1, time2 = target_obj.score()
            if time2 is not None:
                self.time2.append(time2)
            self.targets.pop(rid)
            
        # 同步visualizer的标签和属性，防止数量不匹配导致断言错误
        self.sync_visualizer_targets()

        # 组合绘制所有对象
        all_objs = []
        for typ, id_ in sorted(self.vehicles.keys()):
            all_objs.append(self.vehicles[(typ, id_)])
        all_targets_sorted = [self.targets[k] for k in sorted(self.targets.keys())]
        all_objs.extend(all_targets_sorted)

        self.visualizer.update(all_objs, step=t, title="UAV-USV-Target 状态")

    def get_detected(self, typ, id_):
        """
        获取指定 UAV 或 USV 的探测目标列表
        """
        vehicle = self.vehicles.get((typ.lower(), str(id_)))
        if vehicle and hasattr(vehicle, 'detected'):
            return vehicle.detected
        return None
    
    def get_detected_usv(self):
        """
        获取usv检测目标列表总和（去重）
        """
        all_detected = []
        for id_ in ['1', '2', '3', '4']:
            vehicle = self.vehicles.get(("usv", str(id_)))
            if vehicle and getattr(vehicle, 'detected', None):
                for i in range(len(vehicle.detected)):
                    all_detected.extend(vehicle.detected[i][1])
        # 去重
        total_detected = list(set(all_detected))
        return total_detected
    
    def get_detected_all_timely(self):
        """
        获取检测目标列表总和（去重）
        """
        all_detected = []
        for typ in ['uav', 'usv']:
            for id_ in ['1', '2', '3', '4']:
                vehicle = self.vehicles.get((typ.lower(), str(id_)))
                if vehicle and getattr(vehicle, 'detected', None):
                    for i in range(len(vehicle.detected)):
                        all_detected.extend(vehicle.detected[i][1])
        # 去重
        total_detected = list(set(all_detected))
        return total_detected
    
    def get_detected_all(self):

        for vehicle in self.vehicles.values():
            if hasattr(vehicle, 'detected') and vehicle.detected:
                for detection in vehicle.detected:
                    if isinstance(detection, tuple) and len(detection) >= 2:
                        # detection格式: (time, target_id, position) 或 (time, [target_ids])
                        if isinstance(detection[1], list):
                            # 多个目标ID的情况
                            for target_id in detection[1]:
                                self.detected.add(str(target_id))
                        else:
                            # 单个目标ID的情况
                            self.detected.add(str(detection[1]))
        return list(self.detected)

    def get_captured(self, typ, id_):
        """
        获取指定 USV 的捕获目标列表
        """
        vehicle = self.vehicles.get((typ.lower(), str(id_)))
        if vehicle and hasattr(vehicle, 'captured'):
            return vehicle.captured
        return None
    
    # def get_captured_all_timely(self):
    #     """
    #     获取 USV 1、2、3、4 的捕获目标列表总和（去重）
    #     """
    #     all_captured = []
    #     typ = 'usv'
    #     for id_ in ['1', '2', '3', '4']:
    #         vehicle = self.vehicles.get((typ.lower(), str(id_)))
    #         if vehicle and getattr(vehicle, 'captured', None):
    #             for i in range(len(vehicle.captured)):
    #                 all_captured.extend(vehicle.captured[i][1])
    #     # 去重
    #     total_captured = list(set(all_captured))
    #     return total_captured
    
    def get_captured_all(self):

        for vehicle in self.vehicles.values():
            if hasattr(vehicle, 'captured') and vehicle.captured:
                for capture in vehicle.captured:
                    if isinstance(capture, tuple) and len(capture) >= 2:
                        if isinstance(capture[1], list):
                            # 多个目标ID的情况
                            for target_id in capture[1]:
                                self.captured.add(str(target_id))
                        else:
                            # 单个目标ID的情况
                            self.captured.add(str(capture[1]))
        return list(self.captured)

    def get_state(self, typ, id_):
        """
        获取指定 UAV 或 USV 的状态（位置 + 朝向）
        """
        vehicle = self.vehicles.get((typ.lower(), str(id_)))
        if vehicle:
            pos = vehicle.position[:]
            heading = vehicle.vehicle_heading
            return [pos, heading]
        return None