import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

from observationModel import *

def plot_results(x, t, dataPath):
    """
    Plots the ground truth and filtered states for position, Euler angles, velocities, and biases.

    Parameters:
        x (numpy.ndarray): Filtered states.
        t (numpy.ndarray): Time array corresponding to filtered states.
        dataPath (str): Path to the data file containing ground truth information.

    Returns:
        None
    """
    data = scipy.io.loadmat(dataPath, simplify_cells=True)
    truth = data['vicon']
    truth_t = data['time']

    plt.figure(figsize=(12, 8))
    plt.subplot(4,1,1)
    plt.plot(truth_t, truth[0,:], 'r-', label='x truth')
    plt.plot(truth_t, truth[1,:], 'b-', label='y truth')
    plt.plot(truth_t, truth[2,:], 'g-', label='z truth')
    plt.plot(t, x[0,:], 'r.', label='x filtered')
    plt.plot(t, x[1,:], 'b.', label='y filtered')
    plt.plot(t, x[2,:], 'g.', label='z filtered')
    plt.legend()
    plt.ylabel('Position (m)')
    plt.title('Position')

    plt.subplot(4,1,2)
    plt.plot(truth_t, truth[3,:], 'r-', label='Roll truth')
    plt.plot(truth_t, truth[4,:], 'b-', label='Pitch truth')
    plt.plot(truth_t, truth[5,:], 'g-', label='Yaw truth')
    plt.plot(t, x[3,:], 'r.', label='Roll filtered')
    plt.plot(t, x[4,:], 'b.', label='Pitch filtered')
    plt.plot(t, x[5,:], 'g.', label='Yaw filtered')
    plt.legend()
    plt.ylabel('Euler Angles (rad/s)')
    plt.title('Roll Pitch Yaw')

    plt.subplot(4,1,3)
    plt.plot(truth_t, truth[6,:], 'r-', label='Vx truth')
    plt.plot(truth_t, truth[7,:], 'b-', label='Vy truth')
    plt.plot(truth_t, truth[8,:], 'g-', label='Vz truth')
    plt.plot(t, x[6,:], 'r.', label='Vx filtered')
    plt.plot(t, x[7,:], 'b.', label='Vy filtered')
    plt.plot(t, x[8,:], 'g.', label='Vz filtered')
    plt.legend()
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocities')

    plt.subplot(4,1,4)
    plt.plot(t, x[9,:],  'r.', label='bgx filtered')
    plt.plot(t, x[10,:], 'b.', label='bgy filtered')
    plt.plot(t, x[11,:], 'g.', label='bgz filtered')
    plt.plot(t, x[12,:], 'r-', label='bax filtered')
    plt.plot(t, x[13,:], 'b-', label='bay filtered')
    plt.plot(t, x[14,:], 'g-', label='baz filtered')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('rad/s | m/s^2')
    plt.title('Bias')

    plt.suptitle('Truth and Filtered States for \n' + str(dataPath))
    plt.show()

    pass

def plot_results2(x, t, dataPath):
    """
    Plots the ground truth and filtered states for position, Euler angles, velocities, and biases. v2

    Parameters:
        x (numpy.ndarray): Filtered states.
        t (numpy.ndarray): Time array corresponding to filtered states.
        dataPath (str): Path to the data file containing ground truth information.

    Returns:
        None
    """
    data = scipy.io.loadmat(dataPath, simplify_cells=True)
    truth = data['vicon']
    truth_t = data['time']

   # Load in data
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)

    # Get observation values
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    for i in range(len(dataFull['data'])):
        data = dataFull['data'][i]
        success, q, p = estimate_pose(data)
        ts[i] = data['t']
        if success:
            poss[i] = np.transpose(p)
            rots[i] = np.transpose(q)
        else:
            poss[i] = np.nan
            rots[i] = np.nan
    
    # Clean observation points with no values
    obs_t   = ts[~np.isnan(poss).any(axis=1)]
    obs_pos = poss[~np.isnan(poss).any(axis=1)]
    obs_rot = rots[~np.isnan(rots).any(axis=1)]

    # plot params
    param_lw = 3
    param_ms = 5

    plt.figure(figsize=(14,6))
    plt.subplot(3,2,1)
    plt.plot(truth_t, truth[0, :], 'k-', label='truth', linewidth=param_lw)
    plt.plot(obs_t, obs_pos[:, 0], 'r.', label='observed', markersize=param_ms)
    plt.plot(t, x[0,:], 'b-', label='filtered')
    plt.title('X Position')
    plt.ylabel('X Position (m)')
    plt.grid()

    plt.subplot(3,2,3)
    plt.plot(truth_t, truth[1, :], 'k-', label='truth', linewidth=param_lw)
    plt.plot(obs_t, obs_pos[:, 1], 'r.', label='observed', markersize=param_ms)
    plt.plot(t, x[1,:], 'b-', label='filtered')
    plt.title('Y Position')
    plt.ylabel('Y Position (m)')
    plt.grid()

    plt.subplot(3,2,5)
    plt.plot(truth_t, truth[2, :], 'k-', label='truth', linewidth=param_lw)
    plt.plot(obs_t, obs_pos[:, 2], 'r.', label='observed', markersize=param_ms)
    plt.plot(t, x[2,:], 'b-', label='filtered')
    plt.title('Z Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.legend()
    plt.grid()

    
    plt.subplot(3,2,2)
    plt.plot(truth_t, np.rad2deg(truth[3, :]), 'k-', label='truth', linewidth=param_lw)
    plt.plot(obs_t, np.rad2deg(obs_rot[:, 0]), 'r.', label='observed', markersize=param_ms)
    plt.plot(t, np.rad2deg(x[3,:]), 'b-', label='filtered')
    plt.title('Roll')
    # plt.xlabel('Time (s)')
    plt.ylabel('Roll (deg)')
    plt.grid()

    plt.subplot(3,2,4)
    plt.plot(truth_t, np.rad2deg(truth[4, :]), 'k-', label='truth', linewidth=param_lw)
    plt.plot(obs_t, np.rad2deg(obs_rot[:, 1]), 'r.', label='observed', markersize=param_ms)
    plt.plot(t, np.rad2deg(x[4,:]), 'b-', label='filtered')
    plt.title('Pitch')
    # plt.xlabel('Time (s)')
    plt.ylabel('Pitch (deg)')
    plt.grid()

    plt.subplot(3,2,6)
    plt.plot(truth_t, np.rad2deg(truth[5, :]), 'k-', label='truth', linewidth=param_lw)
    plt.plot(obs_t, np.rad2deg(obs_rot[:, 2]), 'r.', label='observed', markersize=param_ms)
    plt.plot(t, np.rad2deg(x[5,:]), 'b-', label='filtered')
    plt.title('Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (deg)')
    plt.grid()

    plt.suptitle('Truth, Observation, and Filtered States for \n' + str(dataPath))
    plt.tight_layout()
    # plt.show()


def plot_results3D(x, t, dataPath):
    """
    Plots the ground truth and filtered states in 3D.

    Parameters:
        x (numpy.ndarray): Filtered states.
        t (numpy.ndarray): Time array corresponding to filtered states.
        dataPath (str): Path to the data file containing ground truth information.

    Returns:
        None
    """
    data = scipy.io.loadmat(dataPath, simplify_cells=True)
    truth = data['vicon']
    truth_t = data['time']

   # Load in data
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)

    # Get observation values
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    for i in range(len(dataFull['data'])):
        data = dataFull['data'][i]
        success, q, p = estimate_pose(data)
        ts[i] = data['t']
        if success:
            poss[i] = np.transpose(p)
            rots[i] = np.transpose(q)
        else:
            poss[i] = np.nan
            rots[i] = np.nan
    
    # Clean observation points with no values
    obs_t   = ts[~np.isnan(poss).any(axis=1)]
    obs_pos = poss[~np.isnan(poss).any(axis=1)]
    obs_rot = rots[~np.isnan(rots).any(axis=1)]

    # Plot truth data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.plot(truth[0,:], truth[1,:], zs=truth[2,:], label='truth')
    plt.plot(obs_pos[:,0], obs_pos[:,1], 'r.', zs=obs_pos[:,2], label='observation')
    plt.plot(x[0,:], x[1,:], 'r--', zs=x[2,:], label='filter')
    plt.legend()
    plt.title('Truth, Observation, and Filtered States for \n' + str(dataPath))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_aspect('equal')
    plt.show()