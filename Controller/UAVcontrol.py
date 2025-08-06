import math
import numpy as np

def pi_to_pi(angle):
    """将角度归一化到 [-pi, pi] 范围内"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

# --- UAV 控制类 --- #
class UAVController:
    def __init__(self, dt=0.5):
        # 仿真时间步长
        self.dt = dt

        # 无人机物理参数
        self.UAV_INITIAL_MAX_SPEED = 33.33  # 初始就位阶段的最大速度 (m/s)
        self.UAV_SEARCH_SPEED = 33.3        # 搜索巡航阶段的速度 (m/s)
        self.UAV_TURN_SPEED = 33.3          # 盘旋阶段的速度 (m/s)
        self.UAV_TARGET_TRACKING_SPEED = 7.72  # 目标追踪阶段的速度 (m/s)
        self.UAV_MIN_TURN_RADIUS = 100.0

        # 核心控制参数
        self.ARRIVAL_THRESHOLD = 50.0  # 到达目标点的判断阈值（米）
        self.HEADING_KP = 2.0         # 航向控制的比例增益

        # 使用搜索速度计算盘旋一圈所需的仿真步数
        loiter_time_sec = (2 * math.pi * self.UAV_MIN_TURN_RADIUS) / self.UAV_TURN_SPEED
        self.LOITER_TOTAL_STEPS = int(loiter_time_sec / self.dt)

        # 定义所有可能使用到的航点（初始路径 + 循环路径）
        self.waypoints = {
            # UAV1 初始路径
            'P2': np.array([800.0, 5630.0]),
            'P3': np.array([800.0, 8000.0]),
            'P4': np.array([4630.0, 8000.0]),
            'P5': np.array([4630.0, 6680.0]),
            'TL': np.array([2580.0, 6680.0]),  # UAV1 进入循环点

            # UAV2 初始路径
            'Q2': np.array([800.0, 3630.0]),
            'Q3': np.array([800.0, 1260.0]),
            'Q4': np.array([4630.0, 1260.0]),
            'Q5': np.array([4630.0, 2580.0]),
            'BR': np.array([6680.0, 2580.0]),  # UAV2 进入循环点

            # 循环路径其他点
            'TR': np.array([6680.0, 6680.0]),
            'BL': np.array([2580.0, 2580.0])
        }

        # 四边形循环点顺序
        self.loop_points = ['BR', 'TR', 'TL', 'BL']

        # 目标追踪参数
        self.TARGET_TRACKING_DURATION = int(10.0 / self.dt)  # 探测10s对应的仿真步数
        self.TARGET_COOLDOWN_DURATION = int(70.0 / self.dt)  # 自定义：70秒冷却时间对应的仿真步数
        
        # 目标识别次数限制
        self.MAX_TARGET_DETECTIONS = 4  # 同一目标最多被识别4次后就不再触发驻足
        
        # USV目标排除功能开关
        self.EXCLUDE_USV_TARGETS = True
        
        # 仿真步数计数器
        self.simulation_step = 0
        
        # 目标缓存的生存时间（步数）
        self.TARGET_CACHE_LIFETIME = 6  # 缓存目标6步，即3秒，给足够时间处理一闪而过的目标

        self.uav_states = {
            "1": {
                "state": "INITIAL_MOVE",
                "path": ['P2', 'P3', 'P4', 'P5', 'TL'],  # 初始路径，终点为TL
                "waypoint_idx": 0,
                "loiter_steps_left": 0,
                "loop_idx": 0,  # 循环点索引
                
                # 目标追踪相关状态
                "target_tracking": False,
                "tracking_steps_left": 0,
                "target_position": None,
                "tracking_target_id": None,  # 记录正在追踪的目标ID
                "previous_state": None,  # 保存进入追踪前的状态
                "previous_waypoint_idx": 0,  # 保存进入追踪前的航点索引
                "previous_loop_idx": 0,  # 保存进入追踪前的循环索引
                "previous_loiter_steps": 0,  # 保存进入追踪前的盘旋步数
                
                # 目标追踪冷却记录：{target_id: 剩余冷却步数}
                "tracked_targets_cooldown": {},
                
                # 目标识别次数记录：{target_id: 识别次数}
                "target_detection_count": {},
                
                # 目标缓存机制：用于处理0.5s内一闪而过的目标
                "cached_targets": {},  # {target_id: (target_data, discovery_step)}
                "pending_tracking": None,  # 等待下一步执行的追踪目标
            },
            
            "2": {
                "state": "INITIAL_MOVE",
                "path": ['Q2', 'Q3', 'Q4', 'Q5', 'BR'],  # 初始路径，终点为BR
                "waypoint_idx": 0,
                "loiter_steps_left": 0,
                "loop_idx": 0,  # 循环点索引
                
                # 目标追踪相关状态
                "target_tracking": False,
                "tracking_steps_left": 0,
                "target_position": None,
                "tracking_target_id": None,  # 记录正在追踪的目标ID
                "previous_state": None,  # 保存进入追踪前的状态
                "previous_waypoint_idx": 0,  # 保存进入追踪前的航点索引
                "previous_loop_idx": 0,  # 保存进入追踪前的循环索引
                "previous_loiter_steps": 0,  # 保存进入追踪前的盘旋步数
                
                # 目标追踪冷却记录：{target_id: 剩余冷却步数}
                "tracked_targets_cooldown": {},
                
                # 目标识别次数记录：{target_id: 识别次数}
                "target_detection_count": {},
                
                # 目标缓存机制：用于处理0.5s内一闪而过的目标
                "cached_targets": {},  # {target_id: (target_data, discovery_step)}
                "pending_tracking": None,  # 等待下一步执行的追踪目标
            }
        }

    def _update_target_cooldowns(self, state_info):
        
        """更新目标追踪冷却时间"""
        targets_to_remove = []
        for target_id, cooldown_steps in state_info["tracked_targets_cooldown"].items():
            state_info["tracked_targets_cooldown"][target_id] = cooldown_steps - 1
            if state_info["tracked_targets_cooldown"][target_id] <= 0:
                targets_to_remove.append(target_id)
        
        # 移除冷却时间结束的目标
        for target_id in targets_to_remove:
            del state_info["tracked_targets_cooldown"][target_id]

    def _update_target_cache(self, state_info):
        """更新目标缓存，移除过期的缓存目标"""
        targets_to_remove = []
        for target_id, (target_data, discovery_step) in state_info["cached_targets"].items():
            age = self.simulation_step - discovery_step
            if age > self.TARGET_CACHE_LIFETIME:
                targets_to_remove.append(target_id)
        
        # 移除过期的缓存目标
        for target_id in targets_to_remove:
            del state_info["cached_targets"][target_id]
            print(f"[步骤 {self.simulation_step}] 目标 {target_id} 缓存过期，已移除（缓存了 {age} 步）")

    def _cache_target(self, target, state_info, uav_id):
        """缓存发现的目标"""
        target_id = target[1]
        target_position = target[2]
        state_info["cached_targets"][target_id] = (target, self.simulation_step)
        
        # 显示当前识别次数
        current_count = state_info["target_detection_count"].get(target_id, 0)
        remaining_chances = self.MAX_TARGET_DETECTIONS - current_count
        print(f"[步骤 {self.simulation_step}] UAV {uav_id} 发现目标 {target_id} (位置: {target_position})，已缓存，等待处理 (已识别{current_count}次，剩余{remaining_chances}次机会)")

    def _check_cached_targets(self, state_info, uav_id, manager):
        """检查缓存的目标，看是否可以开始追踪"""
        if state_info["target_tracking"] or state_info["pending_tracking"]:
            return  # 已在追踪或等待追踪中
        
        if len(state_info["cached_targets"]) > 0:
            print(f"[步骤 {self.simulation_step}] UAV {uav_id} 正在检查 {len(state_info['cached_targets'])} 个缓存目标")
            
        for target_id, (target_data, discovery_step) in state_info["cached_targets"].items():
            can_track, reason = self._should_start_tracking(target_data, state_info, manager)
            if can_track:
                # 设置为等待追踪，下一步开始执行
                state_info["pending_tracking"] = target_data
                print(f"[步骤 {self.simulation_step}] UAV {uav_id} 准备在下一步开始追踪缓存的目标 {target_id}")
                # 从缓存中移除该目标
                del state_info["cached_targets"][target_id]
                break
            else:
                print(f"[步骤 {self.simulation_step}] UAV {uav_id} 缓存的目标 {target_id} {reason}，暂不追踪")

    def _clear_target_cache(self, state_info):
        """清空目标缓存"""
        state_info["cached_targets"].clear()
        state_info["pending_tracking"] = None

    def _reset_target_detection_counts(self, state_info):
        """重置所有目标的识别次数（可选功能）"""
        count = len(state_info["target_detection_count"])
        state_info["target_detection_count"].clear()
        if count > 0:
            print(f"已重置 {count} 个目标的识别次数记录")

    def get_target_detection_stats(self, uav_id):
        """获取目标识别统计信息（调试用）"""
        state_info = self.uav_states[uav_id]
        return dict(state_info["target_detection_count"])

    def _should_start_tracking(self, target, state_info, manager):
        
        """判断是否应该开始追踪目标"""
        target_id = target[1]
        
        # 检查目标识别次数是否超过限制
        detection_count = state_info["target_detection_count"].get(target_id, 0)
        if detection_count >= self.MAX_TARGET_DETECTIONS:
            return False, f"目标已被识别{detection_count}次，超过限制({self.MAX_TARGET_DETECTIONS}次)"
        
        # 检查目标是否在USV范围内
        detected_usv = manager.get_detected_usv()
        if detected_usv and len(detected_usv) > 0:
            return False, f"目标在USV范围内"
        
        # 检查是否在冷却期内
        if target_id in state_info["tracked_targets_cooldown"]:
            remaining_cooldown = state_info["tracked_targets_cooldown"][target_id]
            return False, f"仍在冷却期内（剩余 {remaining_cooldown} 步 = {remaining_cooldown * self.dt:.1f} 秒）"
        
        return True, "可以追踪"

    def _start_tracking(self, target, state_info, uav_id, current_state):
        
        """开始追踪目标"""
        target_id = target[1]
        target_position = target[2]
        
        # 增加目标识别次数
        if target_id not in state_info["target_detection_count"]:
            state_info["target_detection_count"][target_id] = 0
        state_info["target_detection_count"][target_id] += 1
        
        state_info["target_tracking"] = True
        state_info["tracking_steps_left"] = self.TARGET_TRACKING_DURATION
        state_info["target_position"] = np.array(target_position)
        state_info["tracking_target_id"] = target_id
        
        current_count = state_info["target_detection_count"][target_id]
        print(f"UAV {uav_id} 开始追踪目标 {target_id}（第{current_count}次识别），位置: {target_position}，将持续 {self.TARGET_TRACKING_DURATION} 步 ({self.TARGET_TRACKING_DURATION * self.dt} 秒)")
        print(f"当前状态 {current_state} 将在后台继续运行，不会被中断")

    def _stop_tracking(self, state_info, uav_id, reason="追踪时间结束"):
        
        """停止追踪目标"""
        print(f"UAV {uav_id} {reason}，恢复原有路径")
        
        # 将刚追踪的目标加入冷却列表
        if state_info["tracking_target_id"]:
            tracked_target_id = state_info["tracking_target_id"]
            state_info["tracked_targets_cooldown"][tracked_target_id] = self.TARGET_COOLDOWN_DURATION
            print(f"目标 {tracked_target_id} 进入冷却期，{self.TARGET_COOLDOWN_DURATION} 步 ({self.TARGET_COOLDOWN_DURATION * self.dt} 秒)")
        
        # 清除追踪状态
        state_info["target_tracking"] = False
        state_info["target_position"] = None
        state_info["tracking_target_id"] = None
        
        # 清空目标缓存
        self._clear_target_cache(state_info)
        print(f"UAV {uav_id} 已清空目标缓存")

    def _handle_target_tracking_logic(self, uav_id, state_info, current_state, manager):
        """处理目标追踪逻辑"""
        detected_targets = manager.get_detected('uav', uav_id)
        
        # 更新目标缓存，移除过期目标
        self._update_target_cache(state_info)
        
        # 调试信息：显示当前检测和缓存状态
        if detected_targets:
            target_ids = [t[1] for t in detected_targets]
            print(f"[步骤 {self.simulation_step}] UAV {uav_id} 当前检测到目标: {target_ids}")
        
        cached_ids = list(state_info["cached_targets"].keys())
        if cached_ids:
            print(f"[步骤 {self.simulation_step}] UAV {uav_id} 缓存中的目标: {cached_ids}")
        
        # 1. 处理等待追踪的目标（从上一步的缓存中来的）
        if state_info["pending_tracking"] and not state_info["target_tracking"]:
            pending_target = state_info["pending_tracking"]
            pending_target_id = pending_target[1]
            can_track, reason = self._should_start_tracking(pending_target, state_info, manager)
            if can_track:
                print(f"[步骤 {self.simulation_step}] UAV {uav_id} 开始追踪等待中的目标 {pending_target_id}")
                self._start_tracking(pending_target, state_info, uav_id, current_state)
            else:
                print(f"[步骤 {self.simulation_step}] UAV {uav_id} 等待追踪的目标 {pending_target_id} 无法追踪：{reason}")
            state_info["pending_tracking"] = None
        
        # 2. 处理新发现的目标 - 优先缓存策略
        if detected_targets and not state_info["target_tracking"]:
            for target in detected_targets:
                target_id = target[1]
                
                # 检查是否已经在缓存中或冷却中
                if (target_id not in state_info["cached_targets"] and 
                    target_id not in state_info["tracked_targets_cooldown"]):
                    
                    # 检查识别次数是否超过限制
                    detection_count = state_info["target_detection_count"].get(target_id, 0)
                    if detection_count >= self.MAX_TARGET_DETECTIONS:
                        print(f"[步骤 {self.simulation_step}] UAV {uav_id} 发现目标 {target_id}，但已识别{detection_count}次，超过限制({self.MAX_TARGET_DETECTIONS}次)，不再缓存")
                        continue
                    
                    # 对于0.5s一闪而过的目标，先缓存再处理
                    self._cache_target(target, state_info, uav_id)
        
        # 3. 检查缓存的目标是否可以追踪（如果当前没有在追踪且没有等待追踪的目标）
        if not state_info["target_tracking"] and not state_info["pending_tracking"]:
            self._check_cached_targets(state_info, uav_id, manager)
                
        # 4. 追踪进行中逻辑
        if state_info["target_tracking"]:
            # 检查正在追踪的目标是否进入了USV范围
            detected_usv = manager.get_detected_usv()
            if (self.EXCLUDE_USV_TARGETS and state_info["target_position"] is not None and 
                detected_usv and len(detected_usv) > 0):
                self._stop_tracking(state_info, uav_id, f"正在追踪的目标 {state_info['tracking_target_id']} 进入USV范围")
            else:
                # 正常的时间倒计时
                state_info["tracking_steps_left"] -= 1
                if state_info["tracking_steps_left"] <= 0:
                    self._stop_tracking(state_info, uav_id)

    def _calculate_movement_control(self, current_state, state_info, pos, heading):
        """计算正常状态机的运动控制"""
        v, omega = 0, 0
        
        if current_state == "INITIAL_MOVE":
            target_waypoint_name = state_info["path"][state_info["waypoint_idx"]]
            target_pos = self.waypoints[target_waypoint_name]
            v, omega = self._move_to_waypoint(pos, heading, target_pos, self.UAV_INITIAL_MAX_SPEED)
            
            # 检查到达
            if np.linalg.norm(target_pos - pos) < self.ARRIVAL_THRESHOLD:
                if state_info["waypoint_idx"] == len(state_info["path"]) - 1:
                    state_info["state"] = "SYNCHRONIZING"
                else:
                    state_info["waypoint_idx"] += 1
                v, omega = 0, 0

        elif current_state == "TRAVELING":
            loop_idx = state_info["loop_idx"]
            target_waypoint_name = self.loop_points[loop_idx]
            target_pos = self.waypoints[target_waypoint_name]
            v, omega = self._move_to_waypoint(pos, heading, target_pos, self.UAV_SEARCH_SPEED)
            
            # 检查到达
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
        """计算朝向目标点的运动控制"""
        target_angle = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
        heading_error = pi_to_pi(target_angle - current_heading)
        return speed, self.HEADING_KP * heading_error

    def _calculate_tracking_control(self, state_info, pos, heading):
        """计算目标追踪的运动控制"""
        if not (state_info["target_tracking"] and state_info["target_position"] is not None):
            return None, None
            
        target_pos = state_info["target_position"]
        dist_to_target = np.linalg.norm(target_pos - pos)
        
        if dist_to_target < self.ARRIVAL_THRESHOLD:
            return 0, 0
        else:
            return self._move_to_waypoint(pos, heading, target_pos, self.UAV_TARGET_TRACKING_SPEED)

    def update(self, manager):
        # 增加仿真步数计数器
        self.simulation_step += 1
        
        # 只初始化UAV和USV的控制指令，不包含目标
        controls = [
            ["uav", "1", 0, 0], ["uav", "2", 0, 0],
            ["usv", "1", 0, 0], ["usv", "2", 0, 0],
            ["usv", "3", 0, 0], ["usv", "4", 0, 0],
        ]

        uav_controls = {}

        for uav_id in ["1", "2"]:
            state_info = self.uav_states[uav_id]
            current_state = state_info["state"]

            # 更新目标追踪冷却时间
            self._update_target_cooldowns(state_info)

            vehicle_state = manager.get_state('uav', uav_id)
            if not vehicle_state:
                continue

            pos = np.array(vehicle_state[0])
            heading = vehicle_state[1]

            # 处理目标追踪逻辑
            self._handle_target_tracking_logic(uav_id, state_info, current_state, manager)

            # 计算正常状态机的控制指令
            v, omega = self._calculate_movement_control(current_state, state_info, pos, heading)

            # 如果正在追踪目标，覆盖上面的控制指令
            track_v, track_omega = self._calculate_tracking_control(state_info, pos, heading)
            if track_v is not None:
                v, omega = track_v, track_omega

            uav_controls[uav_id] = [v, omega]

        # 同步启动循环逻辑（只有在非追踪状态下才进行同步）
        state1 = self.uav_states["1"]["state"]
        state2 = self.uav_states["2"]["state"]
        tracking1 = self.uav_states["1"]["target_tracking"]
        tracking2 = self.uav_states["2"]["target_tracking"]
        
        if (state1 == "SYNCHRONIZING" and state2 == "SYNCHRONIZING" and 
            not tracking1 and not tracking2):
            # UAV1从TL开始循环，UAV2从BR开始循环
            self.uav_states["1"]["state"] = "LOITERING"
            self.uav_states["1"]["loiter_steps_left"] = self.LOITER_TOTAL_STEPS
            self.uav_states["1"]["loop_idx"] = self.loop_points.index('TL')
            self.uav_states["2"]["state"] = "LOITERING"
            self.uav_states["2"]["loiter_steps_left"] = self.LOITER_TOTAL_STEPS
            self.uav_states["2"]["loop_idx"] = self.loop_points.index('BR')

        # 更新控制指令
        for i, control_item in enumerate(controls):
            if control_item[0] == 'uav':
                uav_id = control_item[1]
                if uav_id in uav_controls:
                    controls[i][2] = float(uav_controls[uav_id][0])  # 确保是数字
                    controls[i][3] = float(uav_controls[uav_id][1])  # 确保是数字

        # 验证控制矩阵不包含字符串数值
        for i, ctrl in enumerate(controls):
            if len(ctrl) >= 4:
                try:
                    # 确保第3和第4个元素（索引2和3）是数字类型
                    controls[i][2] = float(ctrl[2])  # 速度
                    controls[i][3] = float(ctrl[3])  # 角速度
                except (ValueError, TypeError):
                    # 如果转换失败，使用默认值0
                    controls[i][2] = 0.0
                    controls[i][3] = 0.0

        return controls