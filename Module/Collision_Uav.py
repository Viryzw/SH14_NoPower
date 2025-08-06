import numpy as np
import sys

class Collision:
    def __init__(self, radius, area_size=9260):
        """
        初始化碰撞检测器
        :param radius: 判定为碰撞的最小距离（单位：米）
        """
        self.radius = radius
        self.area_size = area_size

    def update(self, self_type, self_id, self_position, others):
        """
        碰撞检测
        :param self_type: 自身类型字符串，例如 "uav"
        :param self_id: 自身编号，例如 1 或 "uav1"
        :param self_position: 自身位置，List 或 np.array，形如 [x, y]
        :param others: 其他对象列表，每个对象包含 id、type、position
        :return: None，发生碰撞则终止程序
        """
        self_pos = np.array(self_position)

        for obj in others:
            if not hasattr(obj, 'type') or not hasattr(obj, 'id') or not hasattr(obj, 'position'):
                continue
            if obj.type != "uav" or obj.id == self_id:
                continue

            other_pos = np.array(obj.position)
            dist = np.linalg.norm(self_pos - other_pos)

            if dist <= self.radius:
                print(f"{self_type}_{self_id} 与 {obj.type}_{obj.id} 相撞")
                sys.exit(1)
                
        # 边界检测：越过边界则退出仿真
        if (self_pos[0] < 0 or self_pos[0] > self.area_size or self_pos[1] < 0 or self_pos[1] > self.area_size):
            sys.exit(1)

        return None
