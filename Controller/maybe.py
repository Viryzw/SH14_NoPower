import numpy as np

class Maybesomewhere:
    def update_target_history(self,manager, history, max_history=6):
        """
        用manager.get_detected_all()和manager.targets实时更新目标历史位置
        history: {"Target1": [...], ...}
        """
        detected_ids = manager.get_detected_all()
        for tid in detected_ids:
            label = f"Target{tid}"
            if tid in manager.targets:
                pos = tuple(manager.targets[tid].position)
                if label not in history:
                    history[label] = []
                history[label].append(pos)
                # 保留最近max_history个
                if len(history[label]) > max_history:
                    history[label] = history[label][-max_history:]

    def get_target_pos_history(self,history, target_id, max_history=6):
        """
        从history字典读取指定目标的历史位置
        返回最近max_history个位置点列表
        """
        label = f"Target{target_id}"
        return history.get(label, [])[-max_history:]

    def estimate_velocity(self,pos_history, dt):
        """
        根据目标历史位置估算速度
        pos_history: [(x0, y0), (x1, y1), ...]，至少两个点
        dt: 两次观测间隔时间
        返回: (vx, vy)
        """
        if len(pos_history) < 6:
            return (0.0, 0.0)
        points = np.array(pos_history[-5:])
        t = np.arange(5) * dt
        vx = np.polyfit(t, points[:, 0], 1)[0]
        vy = np.polyfit(t, points[:, 1], 1)[0]
        # print("预测vx", vx/10)
        # print("预测vy", vy/10)
        return tuple((vx/10, vy/10))

    def predict_trajectory(self,target_position, target_velocity, duration, dt):
        """
        简单线性预测目标轨迹
        target_position: (x, y) 初始位置
        target_velocity: (vx, vy) 速度向量
        duration: 预测总时间（秒）
        dt: 步长（秒）
        返回：轨迹点列表 [(x1, y1), (x2, y2), ...]
        """
        trajectory = []
        pos = np.array(target_position)
        vel = np.array(target_velocity)
        for t in np.arange(0, duration, dt):
            next_pos = pos + vel * t
            trajectory.append(tuple(next_pos))
        return trajectory

    def calculate_fastest_usv(self,trajectory, usv_positions, usv_speeds, dt):
        """
        计算哪个USV最快追上目标，若都追不上则返回最近的USV及其所需时间
        """
        min_time = float('inf')
        best_usv = -1
        min_dist = float('inf')
        nearest_usv = -1
        nearest_time = float('inf')
        for idx, (usv_pos, usv_speed) in enumerate(zip(usv_positions, usv_speeds)):
            for step, target_pos in enumerate(trajectory):
                t = step * dt
                #dist = np.linalg.norm(np.array(usv_pos) - np.array(target_pos))
                dist = np.linalg.norm(usv_pos - np.array(target_pos))
                time_needed = dist / usv_speed
                # 记录最近距离
                if dist < min_dist:
                    min_dist = dist
                    nearest_usv = idx
                    nearest_time = time_needed
                # 能追上且时间合理
                if time_needed <= t:
                    if t < min_time:
                        min_time = t
                        best_usv = idx
                    break
        # 如果没有USV能追上，则返回最近的USV
        if best_usv == -1:
            return nearest_usv, nearest_time
        return best_usv, min_time
