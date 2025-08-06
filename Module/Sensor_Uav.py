import numpy as np

class Sensor:
    def __init__(self, detection_range, fov_deg):
        """
        初始化探测器
        :param detection_range: 探测半径（单位：米）
        :param fov_deg: 视场角，单位：度（默认60度）
        """
        self.detection_range = detection_range
        self.fov_rad = np.deg2rad(fov_deg / 2)  # 转为弧度，取一半用于对称扇形
        self.sensor_type = 'fov'  # 表示扇形探测器

    def detect(self, vehicle_position, vehicle_heading, targets):
        """
        探测目标（扇形区域）
        :param vehicle_position: List[float] 或 np.array，形如 [x, y]
        :param vehicle_heading: 单位向量或方向向量，如 [1, 0] 表示向右
        :param targets: List[Target]，目标列表，每个目标有 .position 和 .id 属性
        :return: List[Tuple[target_id, position]] 或 None，被探测到的目标编号与位置
        """
        detected = []
        vehicle_pos = np.array(vehicle_position)
        heading = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)])
        norm = np.linalg.norm(heading)
        if norm == 0:
            heading = np.array([1.0, 0.0])  # 默认朝右方向
        else:
            heading = heading / norm

        for target in targets:
            target_pos = np.array(target.position)
            rel_vec = target_pos - vehicle_pos
            dist = np.linalg.norm(rel_vec)

            if dist <= self.detection_range:
                if dist == 0:
                    angle = 0  # 距离为0，视为正对
                else:
                    rel_dir = rel_vec / dist
                    angle = np.arccos(np.clip(np.dot(rel_dir, heading), -1.0, 1.0))
                
                if angle <= self.fov_rad:
                    detected.append((target.type, target.id, target.position))

        self.detected = detected if detected else None
