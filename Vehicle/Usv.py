import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Vehicle.BaseVehicle import BaseVehicle
from Module.Sensor_Usv import Sensor
from Module.Collision_Usv import Collision
from Module.Capture_Usv import Capture

class Usv(BaseVehicle, Sensor):
    def __init__(self, init_position, init_theta, id, min_turn_radius=20, detection_range=800, collision_radius=100, capture_range=100, max_speed=10.28, dt=0.5):
        BaseVehicle.__init__(self, init_position=init_position, init_theta=init_theta, min_turn_radius=min_turn_radius, max_speed=max_speed, dt=dt)
        Sensor.__init__(self, detection_range=detection_range)
        Collision.__init__(self, radius=collision_radius)
        Capture.__init__(self, capture_range=capture_range)
        self.id = id
        self.type = 'usv'
        self.captured = []
    
    def update(self, v, omega, uavs, usvs, targets):
        BaseVehicle.update(self, v=v, omega=omega)
        Sensor.detect(self, vehicle_position=self.position, targets=targets)
        Collision.update(self, self_type=self.type, self_id=self.id, self_position=self.position, others=usvs)
        Capture.update(self, vehicle_position=self.position, targets=targets)
        if self.captured_single is not None:
            for i in range(len(self.captured_single)):
                self.captured.append(self.captured_single[i])