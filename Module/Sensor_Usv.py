import numpy as np

class Sensor:
    def __init__(self, detection_range):
        """
        初始化探测器
        :param detection_range: 探测半径（单位：米）
        """
        self.detection_range = detection_range
        self.sensor_type = 'circle'  # 表示圆形探测器

    def detect(self, vehicle_position, targets):
        """
        探测目标
        :param vehicle_position: List[float] 或 np.array，形如 [x, y]
        :param targets: List[Target]，目标列表，每个目标有 .position 和 .id 属性
        :return: List[Tuple[target_id, position]]，被探测到的目标编号与位置
        """
        detected = []

        for target in targets:
            dist = np.linalg.norm(np.array(vehicle_position) - np.array(target.position))
            if dist <= self.detection_range:
                detected.append((target.type, target.id, target.position))

        self.detected = detected if detected else None
