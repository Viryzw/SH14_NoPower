import math
import numpy as np

def pi_to_pi(angle):
    """将角度归一化到 [-pi, pi] 范围内"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

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