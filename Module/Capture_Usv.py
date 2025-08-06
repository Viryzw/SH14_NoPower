import numpy as np

class Capture:
    def __init__(self, capture_range):
        self.capture_range = capture_range

    def update(self, vehicle_position, targets):
        captured = []

        for target in targets:
            dist = np.linalg.norm(np.array(vehicle_position) - np.array(target.position))
            if dist <= self.capture_range and target.T_detect is not None:
                captured.append((target.type, target.id, target.position))
        
        self.captured_single = captured if captured else None
