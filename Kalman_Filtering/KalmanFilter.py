import numpy as np
from matplotlib import pyplot as plt

class Data:
    def __init__(self, filename):
        self.filename = filename
        data = np.loadtxt(self.filename, delimiter=',')
        self.length = max(data.shape)
        self.t = data[:,0]
        self.u = data[:,1:4]
        self.z = data[:,4:7]
        self.dt = np.append(np.diff(self.t), 0)

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
    def __init__(self, data, R, Q):
        self.data = data
        self.A = []  # State Transition A
        self.B = []  # State Transition B
        self.C = np.eye(3,6)  # Observability Matrix C
        self.R = R  # State Transition Covariance Matrix R
        self.Q = Q  # Measurement Covariance Matrix Q
        self.P = np.eye(6) 
        self.bel = np.zeros([6,1])
        self.K = np.eye(3)
        self.i = 0

    def execute(self):
        bel = np.zeros([self.data.length, 6])
        var = np.zeros([self.data.length, 6])
        while self.i < self.data.length:
            self.step()
            bel[self.i-1, :] = self.bel.T
            var[self.i-1, :] = np.diagonal(self.P)
            pass
        return bel, var
        

    def step(self):
        self.setABdt()
        self.z = np.matrix(self.data.z[self.i,:]).T
        self.u = np.matrix(self.data.u[self.i,:]).T
        self.predict()
        self.update()
        self.i += 1

    def setABdt(self):
        _m = 0.027
        _dt = self.data.dt[self.i]
        self.A = np.eye(6) + _dt*np.eye(6, 6, 3) # n x n, 6x6
        self.B = np.vstack([np.eye(3) * (_dt**2.0 / (2.0*_m)), np.eye(3) * _dt / _m])
        pass

    def predict(self):
        self.bel = self.A @ self.bel + self.B @ self.u
        self.P = self.A @ self.P @ self.A.transpose() + self.R

    def update(self):
        self.K = self.P @ self.C.transpose() @ \
             np.linalg.inv(self.C  @ self.P @ self.C.transpose() + self.Q)
        self.bel = self.bel + self.K @ (self.z - self.C @ self.bel)
        self.P = (np.eye(6) - self.K @ self.C) @ self.P
        
R = np.eye(6) # n x n, 6x6
Q = np.eye(3) # k x k, 3x3

mocap = Data('kalman_filter_data_mocap.txt')
k = Kalman(mocap, R, Q)

bel, var = k.execute()

# plt.figure()
# plt.plot(bel, '.')
# plt.plot(mocap.z, '-')
# # plt.plot([:,:], '.')
# plt.legend({'x', 'y', 'z', 'xd', 'yd', 'zd'})
# plt.show()

colors = ['r', 'g', 'b']

plt.figure()
plt.subplot(2,1,1)
for i in range(3):
    plt.plot(mocap.t, mocap.z[:,i], '.', color=colors[i])
    plt.plot(mocap.t, bel[:,i], '-', color=colors[i])
plt.title('Posistion vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend({'zx', 'belx', 'zy','bely', 'belz', 'zz'}, loc='upper right')

plt.subplot(2,1,2)
for i in range(3):
    plt.plot(mocap.t,  bel[:,3+i], '.', color=colors[i])
plt.title('Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend({'vx', 'vy', 'vz'}, loc='upper right')
plt.show()