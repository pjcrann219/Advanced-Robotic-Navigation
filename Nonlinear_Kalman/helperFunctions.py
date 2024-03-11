import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

def plot_results(x, t, dataPath):
    data = scipy.io.loadmat(dataPath, simplify_cells=True)
    truth = data['vicon']
    truth_t = data['time']

    plt.figure()
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

    plt.show()

    pass