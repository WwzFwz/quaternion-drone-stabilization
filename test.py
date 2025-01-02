import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from quaternion_drone import DroneQuaternionControl

def run_experiment(experiment_type, duration=15.0, dt=0.01):
    drone = DroneQuaternionControl()
    time = np.arange(0, duration, dt)
    orientations = []
    angular_velocities = []

    if experiment_type == "step_response":
        # Step response test - 45-degree rotation about z-axis
        target = R.from_euler('z', 45, degrees=True).as_quat()
        target = np.array([target[3], *target[:3]])
        disturbance = None
        
    elif experiment_type == "disturbance_rejection":
        # Maintain orientation while receiving periodic disturbance
        target = np.array([1., 0., 0., 0.])
    
        
    elif experiment_type == "trajectory_tracking":
        # Follow smooth trajectory
        target = np.array([1., 0., 0., 0.])
    
    for t in time:
        if experiment_type == "disturbance_rejection" :
            # disturbance = np.array([0.5, 0.5, 0]) * np.sin(2*np.pi*t)
            # decay = np.exp(-(t-5.0)) 
            # disturbance = np.array([0.1, 0.1, 0]) * np.sin(2*np.pi*t) *decay
            # disturbance = None
            if 0.0 <= t <= 1: 
                # Single hit disturbance
                disturbance = np.array([1.0, 0.0, 0])   * np.sin(2*np.pi*t)
            else:
                disturbance = None

        else:
            disturbance = None
            
        orientation, angular_velocity = drone.update(target, dt, disturbance)
        orientations.append(orientation)
        angular_velocities.append(angular_velocity)
    
    return time, np.array(orientations), np.array(angular_velocities)



def plot_results(results, experiment):
    time, orientations, angular_velocities = results
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot quaternion components
    ax1 = fig.add_subplot(131)
    ax1.plot(time, orientations[:, 0], label='w')
    ax1.plot(time, orientations[:, 1], label='x')
    ax1.plot(time, orientations[:, 2], label='y')
    ax1.plot(time, orientations[:, 3], label='z')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Quaternion Components')
    ax1.legend()
    ax1.grid(True)
    
    # Plot angular velocities
    ax2 = fig.add_subplot(132)
    ax2.plot(time, angular_velocities[:, 0], label='ωx')
    ax2.plot(time, angular_velocities[:, 1], label='ωy')
    ax2.plot(time, angular_velocities[:, 2], label='ωz')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3D visualization
    ax3 = fig.add_subplot(133, projection='3d')
    for i in range(0, len(time), 10):
        q = orientations[i]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        rotmat = r.as_matrix()
        
        origin = np.array([0, 0, 0])
        ax3.quiver(origin[0], origin[1], origin[2],
                  rotmat[0,0], rotmat[1,0], rotmat[2,0],
                  color='r', alpha=0.5)
        ax3.quiver(origin[0], origin[1], origin[2],
                  rotmat[0,1], rotmat[1,1], rotmat[2,1],
                  color='g', alpha=0.5)
        ax3.quiver(origin[0], origin[1], origin[2],
                  rotmat[0,2], rotmat[1,2], rotmat[2,2],
                  color='b', alpha=0.5)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Drone Orientation')
    
    plt.suptitle(f'{experiment.replace("_", " ").title()}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run experiments
    experiments = ["step_response", "disturbance_rejection", "trajectory_tracking"]
    
    for exp in experiments:
        print(f"Running {exp} experiment...")
        results = run_experiment(exp)
        plot_results(results, exp)



