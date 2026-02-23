# Stub file to simulate the VR bridge interface

class VrPose:
    def __init__(self, timestamp_ns, position, rotation):
        self.timestamp_ns = timestamp_ns
        self.position = position
        self.rotation = rotation

class HapticFeedback:
    def __init__(self, timestamp_ns, force, location):
        self.timestamp_ns = timestamp_ns
        self.force = force
        self.location = location

class GazeData:
    def __init__(self, timestamp_ns, position, direction):
        self.timestamp_ns = timestamp_ns
        self.position = position
        self.direction = direction

class VrBridge:
    def __init__(self):
        pass
    
    def start_streaming(self):
        pass
    
    def stop_streaming(self):
        pass
    
    def get_pose(self):
        return VrPose(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    
    def get_haptic(self):
        return HapticFeedback(0, 0.0, "unknown")
    
    def get_gaze(self):
        return GazeData(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])