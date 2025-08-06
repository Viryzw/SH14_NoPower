import math

class BaseVehicle:
    def __init__(self, init_position, init_theta, min_turn_radius, max_speed, dt):
        """
        初始化对象状态。
        :param x: 初始位置 x 坐标
        :param y: 初始位置 y 坐标
        :param theta: 初始朝向角度（单位：弧度）
        :param min_turn_radius: 最小允许的转弯半径（单位：米），对应最大角速度限制
        :max_speed: 最大速度
        """
        self.position = init_position  # 位置为列表 [x, y]
        self.vehicle_heading = init_theta      # 朝向角（弧度）
        self.min_turn_radius = min_turn_radius
        self.max_speed = max_speed
        self.dt = dt
        self.path = [self.position.copy()]  # 存储轨迹
    
    def update(self, v, omega):
        """
        更新状态，受最小转弯半径限制。
        :param v: 线速度（单位：m/s）
        :param omega: 角速度（单位：rad/s）
        """
        # 限制角速度以满足最小转弯半径约束
        if abs(omega) > 1e-6:
            turn_radius = abs(v / omega)
            if turn_radius < self.min_turn_radius:
                omega = math.copysign(abs(v) / self.min_turn_radius, omega)
        else:
            omega = 0.0  # 角速度过小，视为直线运动

        if abs(v) > self.max_speed:
            v = self.max_speed
            
        # 更新朝向
        self.vehicle_heading += omega * self.dt
        self.vehicle_heading = (self.vehicle_heading + math.pi) % (2 * math.pi) - math.pi # [-pi, pi]

        # 更新位置
        dx = v * math.cos(self.vehicle_heading) * self.dt
        dy = v * math.sin(self.vehicle_heading) * self.dt
        self.position[0] += dx
        self.position[1] += dy

        # 存储轨迹
        self.path.append(self.position.copy())

    def get_position(self):
        return self.position.copy()
    
    def get_orientation(self):
        return self.vehicle_heading
    
    def get_path(self):
        return self.path
