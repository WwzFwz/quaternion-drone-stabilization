import numpy as np
from scipy.spatial.transform import Rotation as R

class DroneQuaternionControl:
    def __init__(self):
        # Initial state
        self.orientation = np.array([1., 1., 1., 0.])  # quaternion [w,x,y,z]
        self.angular_velocity = np.zeros(3)  # omega [x,y,z]
        
        # PID gains
        self.Kp = np.array([4.0, 4.0, 4.0])
        self.Ki = np.array([0.1, 0.1, 0.1])
        self.Kd = np.array([1.0, 1.0, 1.0])
        
        # Error tracking
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        
        # Drone physical parameters
        self.inertia = np.array([
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.02]
        ])
        
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quaternion_error(self, q_desired):
        q_current_conj = np.array([self.orientation[0], -self.orientation[1],
                                    -self.orientation[2], -self.orientation[3]])
        return self.quaternion_multiply(q_desired, q_current_conj)
    
    def update(self, target_orientation, dt, disturbance=None):
        q_error = self.quaternion_error(target_orientation)
        
        # Convert to euler angles for PID control
        r = R.from_quat([q_error[1], q_error[2], q_error[3], q_error[0]])
        euler_error = r.as_euler('xyz')
        
        # PID Control
        self.integral_error += euler_error * dt
        derivative_error = (euler_error - self.last_error) / dt
        
        control = (self.Kp * euler_error + 
                    self.Ki * self.integral_error + 
                    self.Kd * derivative_error)
        
        # # Add disturbance if present
        if disturbance is not None:
            control += disturbance
            

    
        # Update angular velocity
        self.angular_velocity = control
        
        # Update orientation using quaternion kinematics
        omega_quat = np.array([0, *self.angular_velocity])
        q_dot = 0.5 * self.quaternion_multiply(self.orientation, omega_quat)
        self.orientation += q_dot * dt
        
        # Normalize quaternion
        self.orientation /= np.linalg.norm(self.orientation)
        
        # Store error for derivative
        self.last_error = euler_error
        
        return self.orientation, self.angular_velocity


