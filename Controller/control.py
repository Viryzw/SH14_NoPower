import math
import numpy as np

def pi_to_pi(angle):
    """将角度归一化到 [-pi, pi] 范围内"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

# --- UAV 控制类 (保持不变) --- #
class UAVController:
    def __init__(self, dt=0.5):
        self.dt = dt
        self.UAV_INITIAL_MAX_SPEED = 33.33
        self.UAV_SEARCH_SPEED = 33.3
        self.UAV_TURN_SPEED = 33.3
        self.UAV_TARGET_TRACKING_SPEED = 7.72
        self.UAV_MIN_TURN_RADIUS = 100.0
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
        self.TARGET_COOLDOWN_DURATION = int(70.0 / self.dt)
        self.EXCLUDE_USV_TARGETS = True
        self.uav_states = {
            "1": {"state": "INITIAL_MOVE", "path": ['P2', 'P3', 'P4', 'P5', 'TL'], "waypoint_idx": 0, "loiter_steps_left": 0, "loop_idx": 0, "target_tracking": False, "tracking_steps_left": 0, "target_position": None, "tracking_target_id": None, "previous_state": None, "previous_waypoint_idx": 0, "previous_loop_idx": 0, "previous_loiter_steps": 0, "tracked_targets_cooldown": {}},
            "2": {"state": "INITIAL_MOVE", "path": ['Q2', 'Q3', 'Q4', 'Q5', 'BR'], "waypoint_idx": 0, "loiter_steps_left": 0, "loop_idx": 0, "target_tracking": False, "tracking_steps_left": 0, "target_position": None, "tracking_target_id": None, "previous_state": None, "previous_waypoint_idx": 0, "previous_loop_idx": 0, "previous_loiter_steps": 0, "tracked_targets_cooldown": {}}
        }

    def _update_target_cooldowns(self, state_info):
        targets_to_remove = []
        for target_id, cooldown_steps in state_info["tracked_targets_cooldown"].items():
            state_info["tracked_targets_cooldown"][target_id] -= 1
            if state_info["tracked_targets_cooldown"][target_id] <= 0:
                targets_to_remove.append(target_id)
        for target_id in targets_to_remove:
            del state_info["tracked_targets_cooldown"][target_id]

    def _should_start_tracking(self, target, state_info, manager):
        target_id = target[1]
        if manager.get_detected_usv() != []: return False, f"目标在USV范围内"
        if target_id in state_info["tracked_targets_cooldown"]:
            remaining_cooldown = state_info["tracked_targets_cooldown"][target_id]
            return False, f"仍在冷却期内（剩余 {remaining_cooldown} 步 = {remaining_cooldown * self.dt:.1f} 秒）"
        return True, "可以追踪"

    def _start_tracking(self, target, state_info, uav_id, current_state):
        target_id, target_position = target[1], target[2]
        state_info["target_tracking"] = True
        state_info["tracking_steps_left"] = self.TARGET_TRACKING_DURATION
        state_info["target_position"] = np.array(target_position)
        state_info["tracking_target_id"] = target_id
        
    def _stop_tracking(self, state_info, uav_id, reason="追踪时间结束"):
        if state_info["tracking_target_id"]:
            tracked_target_id = state_info["tracking_target_id"]
            state_info["tracked_targets_cooldown"][tracked_target_id] = self.TARGET_COOLDOWN_DURATION
        state_info["target_tracking"] = False
        state_info["target_position"] = None
        state_info["tracking_target_id"] = None

    def _handle_target_tracking_logic(self, uav_id, state_info, current_state, manager):
        detected_targets = manager.get_detected('uav', uav_id)
        if detected_targets and not state_info["target_tracking"]:
            target = detected_targets[0]
            can_track, reason = self._should_start_tracking(target, state_info, manager)
            if can_track:
                self._start_tracking(target, state_info, uav_id, current_state)
        elif state_info["target_tracking"]:
            if (self.EXCLUDE_USV_TARGETS and state_info["target_position"] is not None and manager.get_detected_usv() != None):
                self._stop_tracking(state_info, uav_id, f"正在追踪的目标 {state_info['tracking_target_id']} 进入USV范围")
            else:
                state_info["tracking_steps_left"] -= 1
                if state_info["tracking_steps_left"] <= 0:
                    self._stop_tracking(state_info, uav_id)

    def _calculate_movement_control(self, current_state, state_info, pos, heading):
        v, omega = 0, 0
        if current_state == "INITIAL_MOVE":
            target_waypoint_name = state_info["path"][state_info["waypoint_idx"]]
            target_pos = self.waypoints[target_waypoint_name]
            v, omega = self._move_to_waypoint(pos, heading, target_pos, self.UAV_INITIAL_MAX_SPEED)
            if np.linalg.norm(target_pos - pos) < self.ARRIVAL_THRESHOLD:
                if state_info["waypoint_idx"] == len(state_info["path"]) - 1: state_info["state"] = "SYNCHRONIZING"
                else: state_info["waypoint_idx"] += 1
                v, omega = 0, 0
        elif current_state == "TRAVELING":
            loop_idx = state_info["loop_idx"]
            target_waypoint_name = self.loop_points[loop_idx]
            target_pos = self.waypoints[target_waypoint_name]
            v, omega = self._move_to_waypoint(pos, heading, target_pos, self.UAV_SEARCH_SPEED)
            if np.linalg.norm(target_pos - pos) < self.ARRIVAL_THRESHOLD:
                state_info["state"] = "LOITERING"
                state_info["loiter_steps_left"] = self.LOITER_TOTAL_STEPS
                v, omega = 0, 0
        elif current_state == "LOITERING":
            v = self.UAV_TURN_SPEED
            omega = -self.UAV_TURN_SPEED / self.UAV_MIN_TURN_RADIUS
            state_info["loiter_steps_left"] -= 1
            if state_info["loiter_steps_left"] <= 0:
                state_info["state"] = "TRAVELING"
                state_info["loop_idx"] = (state_info["loop_idx"] + 1) % len(self.loop_points)
        elif current_state == "SYNCHRONIZING":
            v = self.UAV_TURN_SPEED
            omega = -self.UAV_TURN_SPEED / self.UAV_MIN_TURN_RADIUS
        return v, omega

    def _move_to_waypoint(self, current_pos, current_heading, target_pos, speed):
        target_angle = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
        heading_error = pi_to_pi(target_angle - current_heading)
        return speed, self.HEADING_KP * heading_error

    def _calculate_tracking_control(self, state_info, pos, heading):
        if not (state_info["target_tracking"] and state_info["target_position"] is not None):
            return None, None
        target_pos = state_info["target_position"]
        if np.linalg.norm(target_pos - pos) < self.ARRIVAL_THRESHOLD: return 0, 0
        else: return self._move_to_waypoint(pos, heading, target_pos, self.UAV_TARGET_TRACKING_SPEED)

    def update(self, manager):
        controls, uav_controls = [["uav", "1", 0, 0], ["uav", "2", 0, 0], ["usv", "1", 0, 0], ["usv", "2", 0, 0], ["usv", "3", 0, 0], ["usv", "4", 0, 0]], {}
        for uav_id in ["1", "2"]:
            state_info = self.uav_states[uav_id]
            current_state = state_info["state"]
            self._update_target_cooldowns(state_info)
            vehicle_state = manager.get_state('uav', uav_id)
            if not vehicle_state: continue
            pos, heading = np.array(vehicle_state[0]), vehicle_state[1]
            self._handle_target_tracking_logic(uav_id, state_info, current_state, manager)
            v, omega = self._calculate_movement_control(current_state, state_info, pos, heading)
            track_v, track_omega = self._calculate_tracking_control(state_info, pos, heading)
            if track_v is not None: v, omega = track_v, track_omega
            uav_controls[uav_id] = [v, omega]
        state1, state2 = self.uav_states["1"]["state"], self.uav_states["2"]["state"]
        tracking1, tracking2 = self.uav_states["1"]["target_tracking"], self.uav_states["2"]["target_tracking"]
        if (state1 == "SYNCHRONIZING" and state2 == "SYNCHRONIZING" and not tracking1 and not tracking2):
            self.uav_states["1"]["state"], self.uav_states["1"]["loiter_steps_left"], self.uav_states["1"]["loop_idx"] = "LOITERING", self.LOITER_TOTAL_STEPS, self.loop_points.index('TL')
            self.uav_states["2"]["state"], self.uav_states["2"]["loiter_steps_left"], self.uav_states["2"]["loop_idx"] = "LOITERING", self.LOITER_TOTAL_STEPS, self.loop_points.index('BR')
        for i, control_item in enumerate(controls):
            if control_item[0] == 'uav':
                uav_id = control_item[1]
                if uav_id in uav_controls: controls[i][2:4] = uav_controls[uav_id]
        return controls


# --- USV 控制类 (全局最优分配版) --- #
class USVController:
    def __init__(self, dt=0.5):
        self.dt = dt
        self.USV_MAX_SPEED = 10.28
        self.DISPOSAL_RANGE = 100.0
        self.ARRIVAL_THRESHOLD = 50.0 
        self.HEADING_KP = 2.0
        self.AVOIDANCE_DISTANCE = 250.0
        self.AVOIDANCE_KP = 5.0

        self.standby_points = {"1": np.array([8695.0, 8695.0]), "2": np.array([565.0, 8695.0]), "3": np.array([565.0, 565.0]), "4": np.array([8695.0, 565.0])}
        
        self.usv_states = {
            uid: {
                "state": "IDLE", # 状态简化为 IDLE, PURSUING, RETURNING
                "target_id": None,
            } for uid in self.standby_points.keys()
        }

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
        # 1. 强制刷新所有USV的传感器情报
        all_usvs = [v for (t, _), v in manager.vehicles.items() if t == "usv"]
        all_targets = list(manager.targets.values())
        for usv in all_usvs:
            if hasattr(usv, 'detect'):
                usv.detect(vehicle_position=usv.position, targets=all_targets)
        
        usv_positions = {uid: manager.get_state('usv', uid)[0] for uid in self.usv_states.keys()}
        
        # 2. 全局最优任务分配
        all_captured_ids = manager.get_captured_all()
        currently_assigned_ids = {info['target_id'] for info in self.usv_states.values() if info['target_id']}
        all_detected_ids = manager.get_detected_all()
        
        # a. 找出所有“可分配”的目标和“可行动”的USV
        unassigned_target_ids = [tid for tid in all_detected_ids if tid not in all_captured_ids and tid not in currently_assigned_ids]
        available_usv_ids = {uid for uid, info in self.usv_states.items() if info["state"] in ["IDLE", "RETURNING"]}

        # b. 如果没有可分配的任务或可行动的USV，则跳过分配
        if unassigned_target_ids and available_usv_ids:
            possible_assignments = []
            for target_id in unassigned_target_ids:
                if target_id not in manager.targets: continue
                target_pos = manager.targets[target_id].position
                for usv_id in available_usv_ids:
                    dist = np.linalg.norm(np.array(usv_positions[usv_id]) - np.array(target_pos))
                    possible_assignments.append({'usv_id': usv_id, 'target_id': target_id, 'dist': dist})
            
            # c. 按距离对所有可能的分配进行排序
            possible_assignments.sort(key=lambda x: x['dist'])
            
            assigned_usvs_this_turn = set()
            assigned_targets_this_turn = set()
            
            # d. 确认最优分配
            for assignment in possible_assignments:
                usv_id, target_id = assignment['usv_id'], assignment['target_id']
                if usv_id not in assigned_usvs_this_turn and target_id not in assigned_targets_this_turn:
                    self.usv_states[usv_id].update({"state": "PURSUING", "target_id": target_id})
                    assigned_usvs_this_turn.add(usv_id)
                    assigned_targets_this_turn.add(target_id)

        # 3. 执行最终状态机
        usv_controls = {}
        for usv_id, state_info in self.usv_states.items():
            pos, heading = np.array(usv_positions[usv_id]), manager.get_state('usv', usv_id)[1]
            v, goal_omega = self.USV_MAX_SPEED, 0.0
            current_state = state_info["state"]

            if current_state == "IDLE":
                goal = self.standby_points[usv_id]
                if np.linalg.norm(goal - pos) < self.ARRIVAL_THRESHOLD: v = 0.0
                else: goal_omega = self.HEADING_KP * pi_to_pi(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - heading)
            
            elif current_state == "PURSUING":
                target_id = state_info["target_id"]
                if target_id not in manager.targets or target_id in manager.get_captured_all():
                    state_info.update({"state": "RETURNING", "target_id": None})
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
        
        return usv_controls