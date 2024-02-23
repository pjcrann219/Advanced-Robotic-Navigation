import numpy as np
from matplotlib import pyplot as plt

def compare_tracks(kalmans):
    """
    Function to compare xyz beliefs of a list of Kalman objects

    Inputs:
        kalmans: list of kalman objects
    Outputs:
        None
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure()
    title = ''
    
    for i, kalman in enumerate(reversed(kalmans)):
        title += str(kalman.data)
        plt.subplot(3,1,1)
        plt.plot(kalman.data.t, kalman.bels[:,0], color = colors[i], label=str(kalman.data))
        plt.subplot(3,1,2)
        plt.plot(kalman.data.t, kalman.bels[:,1], color = colors[i], label=str(kalman.data))
        plt.subplot(3,1,3)
        plt.plot(kalman.data.t, kalman.bels[:,2], color = colors[i], label=str(kalman.data))
        
    plt.subplot(3,1,1)
    plt.ylabel('X axis (m)')
    plt.subplot(3,1,2)
    plt.ylabel('Y axis (m)')
    plt.subplot(3,1,3)
    plt.ylabel('Z axis (m)')
    plt.xlabel('Time (s)')
    plt.suptitle('Pos Belief Comparison')
    plt.legend()
    plt.show()
    pass

def compare_vel(kalmans):
    """
    Function to compare velocity beliefs of a list of Kalman objects

    Inputs:
        kalmans: list of kalman objects
    Outputs:
        None
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure()
    title = ''
    
    for i, kalman in enumerate(reversed(kalmans)):
        title += str(kalman.data)
        plt.subplot(3,1,1)
        plt.plot(kalman.data.t, kalman.bels[:,3], color = colors[i], label=str(kalman.data))
        plt.subplot(3,1,2)
        plt.plot(kalman.data.t, kalman.bels[:,4], color = colors[i], label=str(kalman.data))
        plt.subplot(3,1,3)
        plt.plot(kalman.data.t, kalman.bels[:,5], color = colors[i], label=str(kalman.data))
        
    plt.subplot(3,1,1)
    plt.ylabel('Xd (m/s)')
    plt.subplot(3,1,2)
    plt.ylabel('Yd (m/s)')
    plt.subplot(3,1,3)
    plt.ylabel('Zd (m/s)')
    plt.xlabel('Time (s)')
    plt.suptitle('Vel Belief Comparison')
    plt.legend()
    plt.show()
    pass

def compare_tracks_3D(kalmans):
    """
    Function to compare xyz beliefs of a list of Kalman objects in 3D

    Inputs:
        kalmans: list of kalman objects
    Outputs:
        None
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis (m)')
    ax.set_ylabel('Y axis (m)')
    ax.set_zlabel('Z axis (m)')
    for i, kalman in enumerate(reversed(kalmans)):
        ax.plot(kalman.bels[:,0], kalman.bels[:,1], kalman.bels[:,2], color=colors[i], label=str(kalman.data))

    plt.title('3D tracking comparison')  
    plt.legend(loc='upper right')
    plt.show()

def estimate_R(truth, compare, vVar=0.01):
    error = compare.z - truth.bels[:,0:3]
    posVar = np.var(error, axis=0)
    velVar = vVar = np.ones([1,3])
    R = np.eye(6)
    R = np.diag(np.append(posVar, velVar))
    return R

def plot_error(truth, kalmans):
    """
    Function to compare errors between a list of Kalman objects and a truth.

    Inputs:
        truth: Kalman object used as truth
        kalmans: list of kalman objects
    Outputs:
        None
    """
    plt.figure()

    for i, kalman in enumerate(reversed(kalmans)):
        error = kalman.bels - truth.bels
        errorPos = np.linalg.norm(error[:,0:3], axis=1)
        errorVel = np.linalg.norm(error[:,3:7], axis=1)
        avgPosError = np.mean(errorPos)
        avgVelError = np.mean(errorVel)

        plt.subplot(2,1,1)
        plt.plot(kalman.data.t, errorPos,label=str(kalman.data) + '\nAverage error = ' + str(round(avgPosError, 3)))

        plt.subplot(2,1,2)
        plt.plot(kalman.data.t, errorVel,label=str(kalman.data) + '\nAverage error = ' + str(round(avgVelError, 3)))
    
    plt.subplot(2,1,1)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Position error (m)')
    plt.legend(loc='upper right')

    plt.subplot(2,1,2)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity error (m/s)')
    plt.legend(loc='upper right')

    plt.suptitle('Filtered Pos and Vel errors')
    plt.tight_layout()    
    plt.show()
    pass

def plot_z_error(truth, kalmans):
    """
    Function to plot measurements errors.

    Inputs:
        truth: Kalman object to be used as truth
        kalmans: list of kalman objects
    Outputs:
        None
    """
    plt.figure()

    for i, kalman in enumerate(reversed(kalmans)):
        if kalman.measType == 'pos':
            error = kalman.data.z - truth.bels[:,0:3]
            errorPos = np.linalg.norm(error, axis=1)
            avgerror = np.mean(errorPos)
            plt.subplot(2,1,1)
            plt.plot(kalman.data.t, errorPos,'.',label=str(kalman.data) + '\nAverage error = ' + str(round(avgerror, 3)))
        elif kalman.measType == 'vel':
            error = kalman.data.z - truth.bels[:,3:7]
            errorPos = np.linalg.norm(error, axis=1)
            avgerror = np.mean(errorPos)
            plt.subplot(2,1,2)
            plt.plot(kalman.data.t, errorPos,'.',label=str(kalman.data) + '\nAverage error = ' + str(round(avgerror, 3)))
        

    
    plt.subplot(2,1,1)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Position error (m)')
    plt.legend(loc='upper right')

    plt.subplot(2,1,2)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity error (m/s)')
    plt.legend(loc='upper right')

    plt.suptitle('Measured Pos errors')
    plt.tight_layout()    
    plt.show()
    pass
