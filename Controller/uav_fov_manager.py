"""
UAV FOV 管理模块
负责处理视野覆盖、目标追踪和与USV的协调
"""
import math
import numpy as np
from Module.Sensor_Uav import Sensor


def pi_to_pi(angle):
    """将角度归一化到 [-pi, pi] 范围内"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class UAVFOVManager:
    def __init__(self, dt=0.5):
        self.dt = dt
        self.FOV_RANGE = 3000.0
        self.FOV_ANGLE = math.radians(60)
        self.MAX_FOV_INTERVAL = 103.0
        self.MAX_FOV_INTERVAL_STEPS = int(self.MAX_FOV_INTERVAL / self.dt)
        
        # 控制参数（从UAVController迁移）
        self.UAV_SEARCH_SPEED = 33.3
        self.ARRIVAL_THRESHOLD = 50.0
        
        # 复用现有的传感器模块
        self.sensor = Sensor(detection_range=self.FOV_RANGE, fov_deg=60)
        
        # 全局目标追踪状态
        self.global_targets = {}
        self.current_step = 0
        self.usv_states_ref = None

    def set_usv_states_ref(self, usv_states):
        """设置USV状态引用"""
        self.usv_states_ref = usv_states

    def is_target_in_fov(self, uav_pos, uav_heading, target_pos):
        """判断目标是否在FOV范围内 - 复用传感器逻辑"""
        rel_vec = np.array(target_pos) - np.array(uav_pos)
        dist = np.linalg.norm(rel_vec)
        
        if dist > self.FOV_RANGE or dist == 0:
            return False
            
        uav_dir = np.array([np.cos(uav_heading), np.sin(uav_heading)])
        rel_dir = rel_vec / dist
        angle = np.arccos(np.clip(np.dot(rel_dir, uav_dir), -1.0, 1.0))
        
        return angle <= self.FOV_ANGLE / 2

    def check_usv_range_exit(self, target_id, target_pos, manager):
        """检查目标是否进入USV处置范围"""
        if not self.usv_states_ref:
            return False
        
        # 检查是否已分配给USV
        for usv_id, usv_state in self.usv_states_ref.items():
            if usv_state.get("target_id") == target_id:
                return True
            
        # 检查是否在USV探测范围内
        for usv_id, usv_state in self.usv_states_ref.items():
            if usv_state["state"] in ["IDLE", "INTERCEPTING", "SEARCHING", "PURSUING", "RETURNING"]:
                usv_vehicle_state = manager.get_state('usv', usv_id)
                if usv_vehicle_state and usv_vehicle_state[0]:
                    usv_pos = usv_vehicle_state[0]
                    dist_to_usv = np.linalg.norm(np.array(target_pos) - np.array(usv_pos))
                    if dist_to_usv <= 800.0:  # USV探测范围
                        return True
        
        # 检查是否已被USV探测
        all_usv_detected = set()
        for usv_id in self.usv_states_ref.keys():
            usv_detections = manager.get_detected('usv', usv_id)
            if usv_detections:
                for detection in usv_detections:
                    all_usv_detected.add(str(detection[1]))
        
        return target_id in all_usv_detected

    def collect_current_fov_targets(self, manager):
        """收集当前FOV覆盖的目标"""
        current_fov_targets = {}
        
        for uav_id in ["1", "2"]:
            detections = manager.get_detected('uav', uav_id)
            if detections:
                uav_state = manager.get_state('uav', uav_id)
                if uav_state:
                    uav_pos, uav_heading = uav_state
                    for detection in detections:
                        target_id = str(detection[1])
                        if target_id in manager.targets:
                            target_pos = manager.targets[target_id].position
                            if self.is_target_in_fov(uav_pos, uav_heading, target_pos):
                                if target_id not in current_fov_targets:
                                    current_fov_targets[target_id] = set()
                                current_fov_targets[target_id].add(uav_id)
        
        return current_fov_targets

    def get_urgent_targets(self):
        """获取需要紧急FOV覆盖的目标"""
        urgent_targets = []
        
        for target_id, target_info in self.global_targets.items():
            if (not target_info.get("in_fov", False) and 
                target_info["assigned_uav"] is None):
                
                time_since_exit = self.current_step - target_info["last_exit_fov_time"]
                if time_since_exit >= self.MAX_FOV_INTERVAL_STEPS * 0.8:
                    urgent_targets.append((target_id, target_info, time_since_exit))
        
        return sorted(urgent_targets, key=lambda x: x[2], reverse=True)

    def update_global_targets(self, manager):
        """更新全局目标状态，包括FOV覆盖时间"""
        self.current_step += 1
        
        # 移除已被捕获的目标
        captured_targets = set(manager.get_captured_all())
        
        # 更新已知目标位置并检查是否需要移除
        for target_id in list(self.global_targets.keys()):
            if target_id in captured_targets:
                del self.global_targets[target_id]
                continue
                
            if target_id in manager.targets:
                target_pos = manager.targets[target_id].position
                self.global_targets[target_id]["position"] = target_pos
                
                if self.check_usv_range_exit(target_id, target_pos, manager):
                    del self.global_targets[target_id]
                    continue
            else:
                del self.global_targets[target_id]
                continue
        
        # 收集当前FOV覆盖目标
        current_fov_targets = self.collect_current_fov_targets(manager)
        
        # 更新FOV状态
        for target_id in self.global_targets:
            was_in_fov = self.global_targets[target_id].get("in_fov", False)
            is_in_fov = target_id in current_fov_targets
            
            if is_in_fov:
                self.global_targets[target_id]["last_exit_fov_time"] = self.current_step
                self.global_targets[target_id]["in_fov"] = True
            elif was_in_fov and not is_in_fov:
                self.global_targets[target_id]["last_exit_fov_time"] = self.current_step
                self.global_targets[target_id]["in_fov"] = False
        
        # 添加新发现的目标
        for target_id in manager.get_detected_all():
            target_id = str(target_id)
            if (target_id not in self.global_targets and 
                target_id in manager.targets and 
                target_id not in captured_targets):
                
                target_pos = manager.targets[target_id].position
                if not self.check_usv_range_exit(target_id, target_pos, manager):
                    is_in_fov = target_id in current_fov_targets
                    self.global_targets[target_id] = {
                        "last_exit_fov_time": self.current_step,
                        "position": target_pos,
                        "assigned_uav": None,
                        "priority": 0.0,
                        "in_fov": is_in_fov
                    }
        
        # 更新目标优先级
        for target_id, target_info in self.global_targets.items():
            if target_info.get("in_fov", False):
                target_info["priority"] = 0.0
            else:
                time_since_exit = self.current_step - target_info["last_exit_fov_time"]
                target_info["priority"] = time_since_exit

    def assign_target_to_uav(self, target_id, uav_id):
        """将目标分配给UAV"""
        if target_id in self.global_targets:
            self.global_targets[target_id]["assigned_uav"] = uav_id

    def release_uav_targets(self, uav_id):
        """释放UAV的所有目标分配"""
        for target_info in self.global_targets.values():
            if target_info["assigned_uav"] == uav_id:
                target_info["assigned_uav"] = None

    def release_all_assignments(self):
        """释放所有UAV的目标分配"""
        for target_info in self.global_targets.values():
            target_info["assigned_uav"] = None

    def get_planning_status(self):
        """获取规划状态"""
        return {
            "global_targets": len(self.global_targets),
            "urgent_targets": len(self.get_urgent_targets()),
            "target_details": {
                target_id: {
                    "time_since_exit": 0 if target_info.get("in_fov", False) 
                                      else self.current_step - target_info["last_exit_fov_time"],
                    "assigned_uav": target_info.get("assigned_uav"),
                    "in_fov": target_info.get("in_fov", False)
                }
                for target_id, target_info in self.global_targets.items()
            }
        }

    def evaluate_uav_assignment_cost(self, uav_id, target_pos, manager):
        """评估UAV执行FOV覆盖任务的代价"""
        vehicle_state = manager.get_state('uav', uav_id)
        if not vehicle_state:
            return float('inf')
            
        pos = np.array(vehicle_state[0])
        heading = vehicle_state[1]
        
        # 基础距离代价
        dist_cost = np.linalg.norm(pos - np.array(target_pos))
        
        # 检查是否已经在FOV范围内
        if self.is_target_in_fov(pos, heading, target_pos):
            return dist_cost * 0.1
        
        return dist_cost

    def assign_dynamic_fov_tasks(self, manager, uav_states):
        """动态分配FOV覆盖任务"""
        # 获取紧急目标
        urgent_targets = self.get_urgent_targets()
        if not urgent_targets:
            return
        
        # 预先收集已分配的UAV
        currently_assigned_uavs = set()
        for target_info in self.global_targets.values():
            if target_info.get("assigned_uav"):
                currently_assigned_uavs.add(target_info["assigned_uav"])
        
        for uav_id, uav_state in uav_states.items():
            if uav_state.get("detour_mode", False):
                currently_assigned_uavs.add(uav_id)
        
        # 分配任务
        for target_id, target_info, urgency in urgent_targets:
            if self.check_usv_range_exit(target_id, target_info["position"], manager):
                continue
                
            available_uavs = [uid for uid in ["1", "2"] 
                             if uid not in currently_assigned_uavs and 
                                not uav_states.get(uid, {}).get("detour_mode", False)]
            
            if available_uavs:
                costs = {uid: self.evaluate_uav_assignment_cost(uid, target_info["position"], manager) 
                        for uid in available_uavs}
                best_uav = min(costs.keys(), key=lambda x: costs[x])
                
                if costs[best_uav] < 15000:
                    target_info["assigned_uav"] = best_uav
                    currently_assigned_uavs.add(best_uav)
                    self.plan_detour_for_fov(best_uav, target_info["position"], manager, uav_states)

    def plan_detour_for_fov(self, uav_id, target_pos, manager, uav_states):
        """为UAV规划FOV覆盖的detour路径"""
        vehicle_state = manager.get_state('uav', uav_id)
        if not vehicle_state:
            return
        
        if uav_states[uav_id].get("detour_mode", False):
            return
            
        pos = np.array(vehicle_state[0])
        
        if self.is_target_in_fov(pos, vehicle_state[1], target_pos):
            return
            
        # 计算FOV覆盖点
        to_target = np.array(target_pos) - pos
        dist_to_target = np.linalg.norm(to_target)
        
        if dist_to_target > self.FOV_RANGE:
            coverage_point = pos + (to_target / dist_to_target) * (self.FOV_RANGE * 0.8)
        else:
            coverage_point = pos + (to_target / dist_to_target) * max(200, dist_to_target * 0.9)
        
        uav_states[uav_id]["detour_point"] = coverage_point
        uav_states[uav_id]["detour_mode"] = True

    def handle_detour_logic(self, state_info, pos, heading):
        """处理detour逻辑"""
        if not (state_info.get("detour_mode", False) and state_info.get("detour_point") is not None):
            return None
            
        detour_point = state_info["detour_point"]
        
        # 检查FOV覆盖状态
        for target_id, target_info in self.global_targets.items():
            if target_info.get("assigned_uav") == state_info.get("uav_id"):
                target_pos = target_info["position"]
                if self.is_target_in_fov(pos, heading, target_pos):
                    target_info["last_exit_fov_time"] = self.current_step
                    target_info["in_fov"] = True
                    target_info["assigned_uav"] = None
                    state_info["detour_mode"] = False
                    state_info["detour_point"] = None
                    return None
        
        # 继续朝detour点移动
        distance_to_detour = np.linalg.norm(detour_point - pos)
        
        if distance_to_detour < self.ARRIVAL_THRESHOLD:
            self.cleanup_failed_detour_tasks(state_info.get("uav_id"), state_info)
            return None
        elif distance_to_detour > self.FOV_RANGE * 3:
            self.cleanup_failed_detour_tasks(state_info.get("uav_id"), state_info)
            return None
        
        return self._move_to_waypoint(pos, heading, detour_point, self.UAV_SEARCH_SPEED)

    def cleanup_failed_detour_tasks(self, uav_id, uav_state):
        """清理失败的detour任务"""
        if not uav_id:
            return
            
        # 释放所有分配给该UAV的目标
        for target_info in self.global_targets.values():
            if target_info.get("assigned_uav") == uav_id:
                target_info["assigned_uav"] = None
        
        # 停止detour模式
        uav_state["detour_mode"] = False
        uav_state["detour_point"] = None

    def _move_to_waypoint(self, current_pos, current_heading, target_pos, speed):
        """移动到目标点"""
        target_angle = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
        return speed, 2.0 * pi_to_pi(target_angle - current_heading)
