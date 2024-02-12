import numpy as np
from matplotlib import pyplot as plt

def compare_tracks(kalmans):
    colors = ['r', 'g', 'b']
    plt.figure()
    title = ''
    
    for i, kalman in enumerate(reversed(kalmans)):
        title += str(kalman.data)
        plt.subplot(3,1,1)
        plt.plot(kalman.data.t, kalman.bels[:,0], color = colors[i])
        plt.subplot(3,1,2)
        plt.plot(kalman.data.t, kalman.bels[:,1], color = colors[i])
        plt.subplot(3,1,3)
        plt.plot(kalman.data.t, kalman.bels[:,2], color = colors[i])
        
    plt.subplot(3,1,1)
    plt.ylabel('X axis (m)')
    plt.subplot(3,1,2)
    plt.ylabel('Y axis (m)')
    plt.subplot(3,1,3)
    plt.ylabel('Z axis (m)')
    plt.xlabel('Time (s)')
    plt.suptitle(title)
    plt.show()
    pass

def compare_tracks_3D(kalmans):
    colors = ['r', 'g', 'b']
    title = ''

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis (m)')
    ax.set_ylabel('Y axis (m)')
    ax.set_zlabel('Z axis (m)')
    for i, kalman in enumerate(reversed(kalmans)):
        title += str(kalman.data)
        ax.plot(kalman.bels[:,0], kalman.bels[:,1], kalman.bels[:,2], color=colors[i], label=str(kalman.data))
        
    plt.legend()
    plt.show()

def estimate_R(truth, compare, vVar=0.01):
    error = compare.z - truth.bels[:,0:3]
    posVar = np.var(error, axis=0)
    velVar = vVar = np.ones([1,3])
    R = np.eye(6)
    R = np.diag(np.append(posVar, velVar))
    return R

def plot_error(truth, kalmans):
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
    plt.legend()

    plt.subplot(2,1,2)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity error (m/s)')
    plt.legend()

    plt.suptitle('Filtered Pos and Vel errors')
    plt.tight_layout()    
    plt.show()
    pass

def plot_z_error(truth, kalmans):
    plt.figure()

    for i, kalman in enumerate(reversed(kalmans)):
        error = kalman.data.z - truth.bels[:,0:3]
        errorPos = np.linalg.norm(error, axis=1)
        avgerror = np.mean(errorPos)
        plt.subplot(2,1,1)
        plt.plot(kalman.data.t, errorPos,label=str(kalman.data) + '\nAverage error = ' + str(round(avgerror, 3)))

    
    plt.subplot(2,1,1)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Position error (m)')
    plt.legend()

    plt.suptitle('Measured Pos errors')
    plt.tight_layout()    
    plt.show()
    pass
