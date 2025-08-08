import math
import numpy as np
from Controller.maybe import Maybesomewhere
from .uav_fov_manager import UAVFOVManager

def pi_to_pi(angle):
    """将角度归一化到 [-pi, pi] 范围内"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

# --- UAV 控制类 (修复版) --- #
class UAVController:
    def __init__(self, dt=0.5):
        self.dt = dt
        self.UAV_MAX_SPEED = 33.33
        self.UAV_SEARCH_SPEED = 33.3
        self.UAV_TURN_SPEED = 33.3
        self.UAV_MIN_TURN_RADIUS = 400.0
        self.ARRIVAL_THRESHOLD = 50.0
        self.HEADING_KP = 2.0
        loiter_time_sec = (2 * math.pi * self.UAV_MIN_TURN_RADIUS) / self.UAV_TURN_SPEED
        self.LOITER_TOTAL_STEPS = int(loiter_time_sec / self.dt)
        self.waypoints = {
            'P2': np.array([800.0, 5630.0]), 'P3': np.array([800.0, 8000.0]),
            'P4': np.array([4630.0, 8000.0]), 'P5': np.array([4630.0, 6680.0]),
            'TL': np.array([2580.0, 6680.0]), 'Q2': np.array([800.0, 3630.0]),
            'Q3': np.array([800.0, 1260.0]), 'Q4': np.array([4630.0, 1260.0]),
            'Q5': np.array([4630.0, 2580.0]), 'BR': np.array([6680.0, 2580.0]),
            'TR': np.array([6680.0, 6680.0]), 'BL': np.array([2580.0, 2580.0])
        }
        self.loop_points = ['BR', 'TR', 'TL', 'BL']
        self.TARGET_TRACKING_DURATION = int(10.0 / self.dt)
        self.TARGET_COOLDOWN_DURATION = int(90.0 / self.dt)
        self.MAX_USV_ENGAGEMENT_DISTANCE = 1030.0
        self.UAV_SUPPORT_INTERVAL_STEPS = int(90.0 / dt)
        
        # 初始化FOV管理器
        self.fov_manager = UAVFOVManager(dt=self.dt)

        self.uav_states = {
            "1": {"state": "INITIAL_MOVE", "path": ['P2', 'P3', 'P4', 'P5', 'TL'], "waypoint_idx": 0, "loiter_steps_left": 0, "loop_idx": 0, "tracking_timer": 0, "support_timer": 0, "target_info": None, "pre_mission_state": None, "tracked_targets_cooldown": {}, "detour_point": None, "detour_mode": False},
            "2": {"state": "INITIAL_MOVE", "path": ['Q2', 'Q3', 'Q4', 'Q5', 'BR'], "waypoint_idx": 0, "loiter_steps_left": 0, "loop_idx": 0, "tracking_timer": 0, "support_timer": 0, "target_info": None, "pre_mission_state": None, "tracked_targets_cooldown": {}, "detour_point": None, "detour_mode": False}
        }
        self.usv_states_ref = None

    def set_usv_states_ref(self, usv_states):
        self.usv_states_ref = usv_states
        self.fov_manager.set_usv_states_ref(usv_states)

    def _update_target_cooldowns(self, state_info):
        for target_id in list(state_info["tracked_targets_cooldown"].keys()):
            state_info["tracked_targets_cooldown"][target_id] -= 1
            if state_info["tracked_targets_cooldown"][target_id] <= 0:
                del state_info["tracked_targets_cooldown"][target_id]

    def _should_start_tracking(self, target, state_info, manager):
        target_id = target[1]
        
        # 🔥 优先检查：如果目标在USV探测范围内，立即拒绝追踪（防止摆头）
        if self.fov_manager.check_usv_range_exit(target_id, target[2], manager):
            return False
            
        # 检查USV是否已经探测到这个目标
        detected_usv = manager.get_detected_usv()
        if detected_usv and target_id in detected_usv: 
            return False
            
        # 检查目标是否在冷却期
        if target_id in state_info["tracked_targets_cooldown"]: 
            return False
            
        return True

    def _start_mission_interrupt(self, new_state, target, state_info):
        state_info["pre_mission_state"] = {"state": state_info["state"], "waypoint_idx": state_info["waypoint_idx"], "loiter_steps_left": state_info["loiter_steps_left"], "loop_idx": state_info["loop_idx"]}
        state_info["state"] = new_state
        state_info["target_info"] = {"id": target[1], "lkp": np.array(target[2])}
        if new_state == "TRACKING":
            state_info["tracking_timer"] = self.TARGET_TRACKING_DURATION
        elif new_state == "SUPPORTING":
            state_info["support_timer"] = self.UAV_SUPPORT_INTERVAL_STEPS

    def _end_mission_interrupt(self, state_info):
        if state_info["target_info"]:
            target_id = state_info["target_info"]["id"]
            state_info["tracked_targets_cooldown"][target_id] = self.TARGET_COOLDOWN_DURATION
        
        saved_state = state_info["pre_mission_state"]
        if saved_state:
            # 直接恢复保存的状态
            state_info.update(saved_state)
        else:
            # 如果没有保存状态，恢复到正常巡逻
            state_info["state"] = "TRAVELING" 
            state_info["loop_idx"] = 0
            state_info["waypoint_idx"] = 0
            state_info["loiter_steps_left"] = 0
        
        state_info["pre_mission_state"] = None
        state_info["target_info"] = None
        state_info["tracking_timer"] = 0
        state_info["support_timer"] = 0

    def _find_nearest_loop_point(self, current_pos):
        """🔥 智能循环点选择：根据当前位置找到最近的循环点"""
        if not current_pos or len(current_pos) < 2:
            return 0  # 默认返回第一个循环点
            
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, point_name in enumerate(self.loop_points):
            if point_name in self.waypoints:
                point_pos = self.waypoints[point_name]
                distance = np.linalg.norm(np.array(current_pos) - np.array(point_pos))
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i
        
        return nearest_idx

    def _calculate_patrol_movement(self, state_info, pos, heading):
        """计算巡逻移动 - 简化版（移除detour逻辑）"""
        current_state = state_info["state"]
        
        # LOITERING状态逻辑
        if current_state == "LOITERING":
            v, omega = self.UAV_TURN_SPEED, -self.UAV_TURN_SPEED / self.UAV_MIN_TURN_RADIUS
            state_info["loiter_steps_left"] -= 1
            if state_info["loiter_steps_left"] <= 0:
                state_info["state"] = "TRAVELING"
                state_info["loop_idx"] = (state_info["loop_idx"] + 1) % len(self.loop_points)
            return v, omega
        
        # 检查FOV detour需求（委托给FOV管理器）
        detour_result = self.fov_manager.handle_detour_logic(state_info, pos, heading)
        if detour_result is not None:
            return detour_result
        
        # 正常巡逻逻辑
        if current_state == "INITIAL_MOVE":
            target_pos = self.waypoints[state_info["path"][state_info["waypoint_idx"]]]
            v, omega = self._move_to_waypoint(pos, heading, target_pos, self.UAV_MAX_SPEED)
            if np.linalg.norm(target_pos - pos) < self.ARRIVAL_THRESHOLD:
                if state_info["waypoint_idx"] == len(state_info["path"]) - 1: 
                    state_info["state"] = "TRAVELING"
                    state_info["loop_idx"] = self.loop_points.index(state_info["path"][-1])
                else: 
                    state_info["waypoint_idx"] += 1
                    
        elif current_state == "TRAVELING":
            target_pos = self.waypoints[self.loop_points[state_info["loop_idx"]]]
            v, omega = self._move_to_waypoint(pos, heading, target_pos, self.UAV_SEARCH_SPEED)
            if np.linalg.norm(target_pos - pos) < self.ARRIVAL_THRESHOLD:
                state_info["state"] = "LOITERING"
                state_info["loiter_steps_left"] = self.LOITER_TOTAL_STEPS
        else:
            v, omega = 0, 0
            
        return v, omega

    def _move_to_waypoint(self, current_pos, current_heading, target_pos, speed):
        target_angle = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
        return speed, self.HEADING_KP * pi_to_pi(target_angle - current_heading)

    def update(self, manager):
        # 委托FOV管理给独立的管理器
        self.fov_manager.update_global_targets(manager)
        self.fov_manager.assign_dynamic_fov_tasks(manager, self.uav_states)
        
        controls, uav_controls = [["uav", "1", 0, 0], ["uav", "2", 0, 0], ["usv", "1", 0, 0], ["usv", "2", 0, 0], ["usv", "3", 0, 0], ["usv", "4", 0, 0]], {}
        
        for uav_id in ["1", "2"]:
            state_info = self.uav_states[uav_id]
            state_info["uav_id"] = uav_id  # 确保uav_id被设置
            self._update_target_cooldowns(state_info)
            
            # 🔥 优先检查：如果当前正在追击的目标进入USV范围，立即停止任务
            if state_info["state"] in ["TRACKING", "SUPPORTING"] and state_info["target_info"]:
                target_id = state_info["target_info"]["id"]
                target_pos = state_info["target_info"]["lkp"]
                if self.fov_manager.check_usv_range_exit(target_id, target_pos, manager):
                    # 目标进入USV 800米范围，立即结束当前任务
                    self._end_mission_interrupt(state_info)
                    # print(f"[强制停止] UAV {uav_id} 停止追击进入USV范围的目标 {target_id}")
            
            vehicle_state = manager.get_state('uav', uav_id)
            if not vehicle_state: continue
            pos, heading = np.array(vehicle_state[0]), vehicle_state[1]
            
            v, omega = 0, 0
            current_state = state_info["state"]

            if current_state == "TRACKING":
                state_info["tracking_timer"] -= 1
                target_id = state_info["target_info"]["id"]
                
                # 检查目标是否已被捕获，如果是则立即结束追踪
                if target_id in manager.get_captured_all():
                    self._end_mission_interrupt(state_info)
                    v, omega = self._calculate_patrol_movement(state_info, pos, heading)
                # 🔥 检查目标是否进入USV范围，如果是则立即停止追踪（目标已从全局列表移除）
                elif self.fov_manager.check_usv_range_exit(target_id, state_info["target_info"]["lkp"], manager):
                    # 目标进入USV范围，立即结束追踪任务
                    self._end_mission_interrupt(state_info)
                    v, omega = self._calculate_patrol_movement(state_info, pos, heading)
                else:
                    v, omega = self._move_to_waypoint(pos, heading, state_info["target_info"]["lkp"], self.UAV_MAX_SPEED)
                    if state_info["tracking_timer"] <= 0:
                        # 检查usv_states_ref是否已初始化
                        if self.usv_states_ref is not None:
                            idle_usv_ids = {uid for uid, info in self.usv_states_ref.items() if info["state"] == "IDLE"}
                            min_dist = float('inf')
                            if idle_usv_ids:
                                for usv_id_idle in idle_usv_ids:
                                   usv_state = manager.get_state('usv', usv_id_idle)
                                   if usv_state and usv_state[0]:
                                       usv_pos = usv_state[0]
                                       dist = np.linalg.norm(np.array(usv_pos) - state_info["target_info"]["lkp"])
                                       if dist < min_dist: min_dist = dist
                            
                            if min_dist > self.MAX_USV_ENGAGEMENT_DISTANCE:
                                state_info["state"] = "SUPPORTING"
                                state_info["support_timer"] = self.UAV_SUPPORT_INTERVAL_STEPS
                            else:
                                self._end_mission_interrupt(state_info)
                        else:
                            # 如果USV状态未初始化，直接结束追踪任务
                            self._end_mission_interrupt(state_info)

            elif current_state == "SUPPORTING":
                state_info["support_timer"] -= 1
                target_id = state_info["target_info"]["id"]
                
                # 检查目标是否已被捕获，如果是则立即结束支援
                if target_id in manager.get_captured_all():
                    self._end_mission_interrupt(state_info)
                    v, omega = self._calculate_patrol_movement(state_info, pos, heading)
                # 🔥 检查目标是否进入USV范围，如果是则立即停止支援（目标已从全局列表移除）
                elif self.fov_manager.check_usv_range_exit(target_id, state_info["target_info"]["lkp"], manager):
                    # 目标进入USV范围，立即结束支援任务
                    self._end_mission_interrupt(state_info)
                    v, omega = self._calculate_patrol_movement(state_info, pos, heading)
                else:
                    detections = manager.get_detected('uav', uav_id)
                    if detections and any(d[1] == target_id for d in detections):
                        self._end_mission_interrupt(state_info)
                        v, omega = self._calculate_patrol_movement(state_info, pos, heading)
                    elif state_info["support_timer"] <= 0:
                        v, omega = self._move_to_waypoint(pos, heading, state_info["target_info"]["lkp"], self.UAV_SEARCH_SPEED)
                    else:
                        v, omega = self._calculate_patrol_movement(state_info, pos, heading)
            else: 
                # 常规巡逻模式下，检查是否发现新目标需要追踪
                detected_targets = manager.get_detected('uav', uav_id)
                if detected_targets and self._should_start_tracking(detected_targets[0], state_info, manager):
                    self._start_mission_interrupt("TRACKING", detected_targets[0], state_info)
                    # 开始追踪后，立即计算追踪移动指令
                    v, omega = self._move_to_waypoint(pos, heading, state_info["target_info"]["lkp"], self.UAV_MAX_SPEED)
                else:
                    # 执行巡逻移动（包含动态detour逻辑）
                    v, omega = self._calculate_patrol_movement(state_info, pos, heading)

            uav_controls[uav_id] = [v, omega]

        # 🔥 取消同步检查：移除原有的同步逻辑，UAV各自独立巡逻
        for i, control_item in enumerate(controls):
            if control_item[0] == 'uav':
                if control_item[1] in uav_controls: controls[i][2:4] = uav_controls[control_item[1]]
        return controls

    def get_planning_status(self):
        """获取FOV规划状态 - 委托给FOV管理器"""
        return self.fov_manager.get_planning_status()

# --- USV 控制类 (LKP模型版) --- #
class USVController:
    def __init__(self, dt=0.5):
        self.dt = dt
        self.USV_MAX_SPEED = 10.28
        self.USV_SEARCH_SPEED = 5.0 
        self.DISPOSAL_RANGE = 100.0
        self.ARRIVAL_THRESHOLD = 50.0 
        self.HEADING_KP = 2.0
        self.AVOIDANCE_DISTANCE = 250.0
        self.AVOIDANCE_KP = 5.0
        self.SEARCH_DURATION_STEPS = int(60 / dt) 
        self.SEARCH_TURN_OMEGA = 0.2
        self.history = {}

        self.standby_points = {"1": np.array([6680.0, 6680.0]), "2": np.array([2580.0, 6680.0]), "3": np.array([2580.0, 2580.0]), "4": np.array([6680.0, 2580.0])}
        self.usv_states = {
            uid: {
                "state": "IDLE", 
                "target_id": None,
                "last_known_pos": None,
                "search_timer": 0,
            } for uid in self.standby_points.keys()
        }

    def _assign_new_tasks(self, manager, usv_positions):
        all_detected_ids = manager.get_detected_all()
        if not all_detected_ids: return

        all_captured_ids = manager.get_captured_all()
        # 允许已分配目标重新分配
        new_task_ids = [tid for tid in all_detected_ids if tid not in all_captured_ids]

        if not new_task_ids: return

        # 所有USV都参与分配（包括IDLE和INTERCEPTING）
        candidate_usv_ids = [uid for uid in self.usv_states.keys()]
        if not candidate_usv_ids: return

        maybe = Maybesomewhere()
        dt = 0.05

        # 记录已分配 USV ID，避免冲突
        assigned_usvs = set()
        assigned_targets = set()

        # 先保存所有目标对应的初始分配信息（包含best_usv_idx等）
        target_assignments = {}

        for target_id in new_task_ids:
            if target_id not in manager.targets: continue
            maybe.update_target_history(manager, self.history, max_history=10)
            pos_history = maybe.get_target_pos_history(self.history, target_id)
            target_velocity = maybe.estimate_velocity(pos_history, dt)
            # print("历史", pos_history)
            target_position = pos_history[-1] if pos_history else manager.targets[target_id].position
            trajectory = maybe.predict_trajectory(target_position, target_velocity, duration=1000, dt=1)
            usv_pos_list = [usv_positions[uid] for uid in candidate_usv_ids]
            usv_speed_list = [self.USV_MAX_SPEED for _ in candidate_usv_ids]

            best_usv_idx, intercept_time = maybe.calculate_fastest_usv(trajectory, usv_pos_list, usv_speed_list, dt=dt)
            target_assignments[target_id] = {
                'trajectory': trajectory,
                'best_usv_idx': best_usv_idx,
                'intercept_time': intercept_time
            }

        # 按时间排序的目标ID列表
        sorted_targets = sorted(target_assignments.items(), key=lambda x: x[1]['intercept_time'])

        final_assignments = []

        for target_id, info in sorted_targets:
            best_usv_idx = info['best_usv_idx']
            if best_usv_idx is None or best_usv_idx < 0 or best_usv_idx >= len(candidate_usv_ids):
                continue

            # 检查该USV是否已被占用
            if best_usv_idx in assigned_usvs:
                # 该USV已被占用，重新计算（剔除已分配USV）
                available_usvs = [uid for idx, uid in enumerate(candidate_usv_ids) if idx not in assigned_usvs]
                if not available_usvs:
                    # 无可用USV，跳过
                    continue

                # 重新计算最快USV，排除已分配USV
                available_pos = [usv_positions[uid] for uid in available_usvs]
                available_speed = [self.USV_MAX_SPEED for _ in available_usvs]

                best_usv_idx_new, intercept_time_new = maybe.calculate_fastest_usv(
                    info['trajectory'], available_pos, available_speed, dt=1)

                if best_usv_idx_new is None or best_usv_idx_new < 0 or best_usv_idx_new >= len(available_usvs):
                    continue

                # 更新 best_usv_idx 和 intercept_time 为重新计算结果
                best_usv_idx = candidate_usv_ids.index(available_usvs[best_usv_idx_new])
                intercept_time = intercept_time_new
            else:
                intercept_time = info['intercept_time']

            # 标记分配
            assigned_usvs.add(best_usv_idx)
            assigned_targets.add(target_id)

            # 计算拦截点
            intercept_step = int(intercept_time)
            trajectory = info['trajectory']
            if intercept_step >= len(trajectory):
                intercept_step = len(trajectory) - 1
            intercept_point = np.array(trajectory[intercept_step])

            final_assignments.append({
                'usv_id': candidate_usv_ids[best_usv_idx],
                'target_id': target_id,
                'lkp': intercept_point,
                'intercept_time': intercept_time
            })
            # print("预测相遇点", intercept_point)
            # print("预测相遇时间", intercept_step)

        # 先用字典收集每个目标的最佳分配                                                                        ################################################
        best_assignment_per_target = {}                                                                    ################################################
                                                                                                           ################################################
        for assignment in final_assignments:                                                               ################################################
            target_id = assignment['target_id']                                                            ################################################
            if target_id not in best_assignment_per_target:                                                ############                       #############
                best_assignment_per_target[target_id] = assignment                                         ############    防止目标脚踏两只船    #############
            else:                                                                                          ############                       #############
                # 只保留时间最短的                                                                            ################################################
                if assignment['intercept_time'] < best_assignment_per_target[target_id]['intercept_time']: ################################################
                    best_assignment_per_target[target_id] = assignment                                     ################################################
                                                                                                           ################################################
        # 只留下每个目标最快的USV分配                                                                          ################################################
        final_assignments = list(best_assignment_per_target.values())                                      ################################################

        # 最终更新 usv_states
        for assignment in final_assignments:
            usv_id = assignment['usv_id']
            target_id = assignment['target_id']
            self.usv_states[usv_id].update({
                "state": "INTERCEPTING",
                "target_id": target_id,
                "last_known_pos": assignment['lkp']
            })

    def _update_lkp_from_uav(self, manager):
        all_detected_ids = manager.get_detected_all()
        for state_info in self.usv_states.values():
            if state_info["state"] == "INTERCEPTING":
                target_id = state_info["target_id"]
                if target_id in all_detected_ids and target_id in manager.targets:
                    state_info["last_known_pos"] = np.array(manager.targets[target_id].position)

    def _calculate_avoidance_omega(self, self_id, self_pos, self_heading, usv_positions):
        total_repulsion_vec = np.array([0.0, 0.0])
        for other_id, other_pos in usv_positions.items():
            if self_id == other_id: continue
            dist_vec = self_pos - np.array(other_pos)
            dist = np.linalg.norm(dist_vec)
            if 0 < dist < self.AVOIDANCE_DISTANCE:
                strength = (self.AVOIDANCE_DISTANCE - dist) / self.AVOIDANCE_DISTANCE
                total_repulsion_vec += (dist_vec / dist) * strength
        if np.linalg.norm(total_repulsion_vec) > 0:
            avoidance_angle = math.atan2(total_repulsion_vec[1], total_repulsion_vec[0])
            return self.AVOIDANCE_KP * pi_to_pi(avoidance_angle - self_heading)
        return 0.0

    def update(self, manager):
        all_usvs = [v for (t, _), v in manager.vehicles.items() if t == "usv"]
        all_targets = list(manager.targets.values())
        for usv in all_usvs:
            if hasattr(usv, 'detect'):
                usv.detect(vehicle_position=usv.position, targets=all_targets)
        
        usv_positions = {uid: manager.get_state('usv', uid)[0] for uid in self.usv_states.keys()}
        
        self._update_lkp_from_uav(manager)
        self._assign_new_tasks(manager, usv_positions)

        usv_controls = {}
        for usv_id, state_info in self.usv_states.items():
            pos, heading = np.array(usv_positions[usv_id]), manager.get_state('usv', usv_id)[1]
            v, goal_omega = self.USV_MAX_SPEED, 0.0
            current_state = state_info["state"]

            if current_state == "IDLE":
                goal = self.standby_points[usv_id]
                if np.linalg.norm(goal - pos) < self.ARRIVAL_THRESHOLD: v = 0.0
                else: goal_omega = self.HEADING_KP * pi_to_pi(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - heading)
            
            elif current_state == "INTERCEPTING":
                target_id = state_info["target_id"]
                if target_id not in manager.targets or target_id in manager.get_captured_all():
                    state_info.update({"state": "RETURNING", "target_id": None, "last_known_pos": None})
                else:
                    goal = state_info["last_known_pos"]
                    if np.linalg.norm(goal - pos) < self.ARRIVAL_THRESHOLD:
                        state_info["state"] = "SEARCHING"
                        state_info["search_timer"] = self.SEARCH_DURATION_STEPS
                    else:
                        goal_omega = self.HEADING_KP * pi_to_pi(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - heading)
            
            elif current_state == "SEARCHING":
                v, goal_omega = self.USV_SEARCH_SPEED, self.SEARCH_TURN_OMEGA
                state_info["search_timer"] -= 1
                usv_detections = manager.get_detected('usv', usv_id)
                found = any(d[1] == state_info["target_id"] for d in usv_detections) if usv_detections else False
                if found:
                    state_info["state"] = "PURSUING"
                elif state_info["search_timer"] <= 0:
                    state_info.update({"state": "RETURNING", "target_id": None, "last_known_pos": None})

            elif current_state == "PURSUING":
                target_id = state_info["target_id"]
                if target_id not in manager.targets or target_id in manager.get_captured_all():
                    state_info.update({"state": "RETURNING", "target_id": None, "last_known_pos": None})
                else:
                    target_pos = manager.targets[target_id].position
                    if np.linalg.norm(np.array(target_pos) - pos) < self.DISPOSAL_RANGE: v = 0.0
                    else: goal_omega = self.HEADING_KP * pi_to_pi(math.atan2(target_pos[1] - pos[1], target_pos[0] - pos[0]) - heading)
            
            elif current_state == "RETURNING":
                goal = self.standby_points[usv_id]
                if np.linalg.norm(goal - pos) < self.ARRIVAL_THRESHOLD:
                    state_info["state"], v = "IDLE", 0.0
                else:
                    goal_omega = self.HEADING_KP * pi_to_pi(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - heading)

            avoidance_omega = self._calculate_avoidance_omega(usv_id, pos, heading, usv_positions)
            final_omega = goal_omega + avoidance_omega
            if abs(avoidance_omega) > 0.1: v *= 0.8
            usv_controls[usv_id] = [v, final_omega]
        
        # 清理所有USV的旧分配，只保留本轮分配，防止目标脚踏两只船 
        for usv_id, state_info in self.usv_states.items():
            if state_info["state"] == "INTERCEPTING":
                state_info["target_id"] = None
                state_info["last_known_pos"] = None

        return usv_controls
