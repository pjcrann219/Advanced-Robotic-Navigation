import numpy as np
from matplotlib import pyplot as plt

class Data:
    """
    Class to load in and work with data from .txt file.

    Attributes:
        filename (str): file to load text from.
        length (int): data length.
        t (array): time array.
        u (array): input U array.
        z (array): measurement Z array.
        dt (array): dT array.

    Methods:
        __init__(self, filename): Load and parse data.
        __str__(self): returns filename.
        plot_action(self): plots actions vs time.
        plot_measurements(self): plots measurements vs time.
    """
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
    """
    Class to implement Kalman filter.

    Attributes:
        data (Data): Data object
        A (array): State transition matrix A
        B (array): State transition matrix B
        C (array): Observability Matrix C
        R (array): State Transition Covariance Matrix R
        Q (array): Measurement Covariance Matrix Q
        P (array): Belief covariance matrix P
        bel (array): Most recent belief
        K (array): Kalman gain
        i (int): Time step counter
        bels(array): record of beliefs
        var(array): record of P where diag(var) = P

    Methods:
        __init__(self, data, R, Q): Initiates Kalman class.
        execute(self): executes Kalman filter on time series.
        setABdt(self): sets A and B matricies and dT value based on current time step.
        predict(self): Implements predict step of Kalman Filter
        update(self): Implements update step of Kalman Filter
        plot_state(self): Plots beliefs and measurements vs time
        plot_state(self, plot_z): Plots beliefs and measurements in 3D
    """
    def __init__(self, data, R, Q, C=np.eye(3,6)):
        self.data = data
        self.A = []  # State Transition A
        self.B = []  # State Transition B
        self.C = C  # Observability Matrix C
        self.R = R  # State Transition Covariance Matrix R
        self.Q = Q  # Measurement Covariance Matrix Q
        self.P = np.eye(6)
        self.bel = np.zeros([6,1])
        self.K = np.eye(3)
        self.i = 0
        self.bels = np.zeros([self.data.length, 6]) # Record of beliefs
        self.var = np.zeros([self.data.length, 6])  # Record of variances
        self.pdet = np.zeros([self.data.length, 1])  # Record of variances

    def execute(self):
        bel = np.zeros([self.data.length, 6])
        var = np.zeros([self.data.length, 6])
        while self.i < self.data.length:
            self.setABdt()
            self.z = np.matrix(self.data.z[self.i,:]).T
            self.u = np.matrix(self.data.u[self.i,:]).T
            self.predict()
            self.update()
            self.i += 1
            self.bels[self.i-1, :] = self.bel.T
            self.var[self.i-1, :] = np.diagonal(self.P)
            self.pdet[self.i-1, :] = np.linalg.det(self.P)
        return bel, var

    def setABdt(self):
        _m = 0.027
        _dt = self.data.dt[self.i]
        self.A = np.eye(6) + _dt*np.eye(6, 6, 3) # n x n, 6x6
        self.B = np.vstack([np.eye(3) * (_dt**2.0 / (2.0*_m)), np.eye(3) * _dt / _m])

    def predict(self):
        self.bel = self.A @ self.bel + self.B @ self.u
        self.P = self.A @ self.P @ self.A.transpose() + self.R

    def update(self):
        self.K = self.P @ self.C.transpose() @ \
             np.linalg.inv(self.C  @ self.P @ self.C.transpose() + self.Q)
        self.bel = self.bel + self.K @ (self.z - self.C @ self.bel)
        self.P = (np.eye(6) - self.K @ self.C) @ self.P
        
    def plot_state(self):
        colors = ['r', 'g', 'b']
        plt.figure()
        plt.suptitle(self.data)
        plt.subplot(2,1,1)
        for i in range(3):
            plt.plot(self.data.t, self.data.z[:,i], '.', color=colors[i])
            plt.plot(self.data.t, self.bels[:,i], '-', color=colors[i])
        plt.grid()
        plt.title('Posistion vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend({'x_measurement', 'x_bel', 'y_measurement', 'y_bel', 'z_measurement', 'z_bel'},\
             loc='upper right')
        
        plt.subplot(2,1,2)
        for i in range(3):
            plt.plot(self.data.t,  self.bels[:,3+i], '.', color=colors[i])
        plt.grid()
        plt.title('Velocities')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend({'vx', 'vy', 'vz'}, loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_state_3D(self, plot_z = False):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(self.bels[:,0], self.bels[:,1], self.bels[:,2])
        ax.set_xlabel('X axis (m)')
        ax.set_ylabel('Y axis (m)')
        ax.set_zlabel('Z axis (m)')
        plt.title('Filtered Beliefs from ' + str(self.data))
        if plot_z:
            ax.plot(self.data.z[:,0], self.data.z[:,1], self.data.z[:,2], '.')
        plt.show()

    def plot_Pdet(self):
        plt.figure()
        plt.suptitle(self.data)
        plt.plot(self.data.t, self.pdet, '.')
        plt.grid()
        plt.title('Pdet vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Pdet')
        plt.show()