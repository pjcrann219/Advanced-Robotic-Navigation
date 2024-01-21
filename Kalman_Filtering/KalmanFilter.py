import numpy as np
from matplotlib import pyplot as plt

class Data:
    def __init__(self, filename):
        self.filename = filename
        data = np.loadtxt(self.filename, delimiter=',')
        self.length = data.__len__
        self.t = data[:,0]
        self.u = data[:,1:4]
        self.z = data[:,4:7]

    def __str__(self):
        return self.filename

    def plot_action(self):
        plt.figure()
        plt.subplot(3,1,1)
        plt.title('Control Inputs: ' + str(self))
        plt.plot(self.t, self.u[:,0], 'r.', label='ux')
        plt.legend()
        plt.ylabel('X Force Component (N)')

        plt.subplot(3,1,2)
        plt.plot(self.t, self.u[:,1], 'g.', label='uy')
        plt.legend()
        plt.ylabel('Y Force Component (N)')

        plt.subplot(3,1,3)
        plt.plot(self.t, self.u[:,2], 'b.', label='uz')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Z Force Component (N)')
        plt.show()

    def plot_measurements(self):
        plt.figure()
        plt.subplot(3,1,1)
        plt.title('Measurements: ' + str(self))
        plt.plot(self.t, self.z[:,0], 'r.', label='zx')
        plt.legend()
        plt.ylabel('X Measurents (m)')

        plt.subplot(3,1,2)
        plt.plot(self.t, self.z[:,1], 'g.', label='zy')
        plt.legend()
        plt.ylabel('Y Measurents (m)')

        plt.subplot(3,1,3)
        plt.plot(self.t, self.z[:,2], 'b.', label='zz')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Z Measurents (m)')
        plt.show()


class Kalman:
    def __init__(self, data, A, B, R, Q):
        self.data = data
        
mocap = Data('kalman_filter_data_mocap.txt')
k = Kalman(mocap)