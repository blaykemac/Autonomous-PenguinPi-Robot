import numpy as np

MARKER_COV_SCALE = 3
MOTOR_COV_SCALE = 0.0001


class Marker:
    # Measurements are of landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag, covariance = (MARKER_COV_SCALE*0.1*np.eye(2))):
        self.position = position
        self.tag = tag
        self.covariance = covariance

class Drive:
    # Measurement of the robot wheel velocities
    def __init__(self, left_speed, right_speed, dt, left_cov = 1*MOTOR_COV_SCALE, right_cov = 1*MOTOR_COV_SCALE):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov