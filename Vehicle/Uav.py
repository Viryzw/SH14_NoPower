import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Vehicle.BaseVehicle import BaseVehicle
from Module.Sensor_Uav import Sensor
from Module.Collision_Uav import Collision

class Uav(BaseVehicle, Sensor):
    def __init__(self, init_position, init_theta, id=0, min_turn_radius=100, detection_range=3000, fov_deg=60, collision_radius=50, max_speed=33.33, dt=0.5):
        BaseVehicle.__init__(self, init_position=init_position, init_theta=init_theta, min_turn_radius=min_turn_radius, max_speed=max_speed, dt=dt)
        Sensor.__init__(self, detection_range=detection_range, fov_deg=fov_deg)
        Collision.__init__(self, radius=collision_radius)
        self.id = id
        self.type = 'uav'
    
    def update(self, v, omega, uavs, usvs, targets):
        BaseVehicle.update(self, v=v, omega=omega)
        Sensor.detect(self, vehicle_position=self.position, vehicle_heading=self.vehicle_heading, targets=targets)
        Collision.update(self, self_type=self.type, self_id=self.id, self_position=self.position, others=uavs)