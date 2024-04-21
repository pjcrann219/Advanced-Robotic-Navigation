import numpy as np
from propogation import *
from earth import *
from matplotlib import pyplot as plt

class INS_GNSS:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.data = np.genfromtxt(self.dataPath, delimiter=',', skip_header=1)
        self.len = len(self.data)
        self.data_t = self.data[:,0][:,np.newaxis]
        self.data_true_lla = self.data[:,1:4]
        self.data_true_rpy = self.data[:,4:7]
        self.data_gyro_xyz = self.data[:,7:10]
        self.data_accel_xyz = self.data[:,10:13]
        self.data_z_lla = self.data[:, 13:16]
        self.data_z_VNED = self.data[:,16:19]
        self.x_fb = np.zeros([self.len, 15])
        self.x_ff = np.zeros([self.len, 12])
        self.dt = 1
        self.P = np.eye(15) * 0.1
        self.Q = np.eye(15) * 0.01
        self.R = np.eye(6) * 0.00001

    def execute_feedback(self):

        n = 15

        x_prior = np.vstack([self.data[0,1:7][:, np.newaxis], np.zeros([9,1])])
        x_prior[5,0] = np.deg2rad(195)

        P = self.P
        # for i in range(self.len):
        for i in range(1000):
            t = self.data_t[i,0]

            gyro = self.data_gyro_xyz[i, :][:, np.newaxis]
            accel = self.data_accel_xyz[i, :][:, np.newaxis]
            z_lla = self.data_z_lla[i, :]
            z_VNED = self.data_z_VNED[i, :]

            # Get sigma points X0 with wights wm, wc
            X0, wm, wc = getSigmaPoints(x_prior, P, n)
            
            # Propogate sigma points through state transition
            X1 = np.zeros(np.shape(X0))
            for j in range(2*n+1):
                thisX = X0[:,j]
                X1[:,j] = np.squeeze(propogation_model_feedback(x_prior, gyro, accel, self.dt))
            
            # Recover mean
            x = np.sum(X1 * wm, axis=1)
            # Recover variance
            diff = X1-np.vstack(x)
            P = np.zeros((n, n))
            diff2 = X1-np.vstack(X1[:,0])
            for j in range(2*n+1):
                d = diff2[:, j].reshape(-1,1)
                P += wc[j] * d @ d.T
            P += self.Q

            # Get new sigma points
            X2, wm, wc = getSigmaPoints(x, self.P, n)

            # Put sigma points through measurement model
            Z = np.zeros([6,2*n+1])
            for j in range(2*n+1):
                thisX = X2[:,j]
                Z[:,j] = measurement_model(thisX)

                # Recover mean
            z = np.sum(Z * wm, axis=1)

            # Recover variance
            S = np.zeros((6,6))

            diff2 = Z-np.vstack(Z[:,0])
            for j in range(2 * n + 1):
                d = diff2[:, j].reshape(-1,1)
                S += wc[j] * d @ d.T
            S += self.R


            diff = Z-np.vstack(z)
            # Compute cross covariance
            cxz = np.zeros([n,6])
            for j in range(2 * n + 1):
                cxz += wc[j] * np.outer(X2[:,j] - x, diff[:, j])

            # Compute Kalman Gain
            K = cxz @ np.linalg.inv(S)

            z_meas = np.vstack((z_lla, z_VNED)).reshape((6, 1))

            # Update estimate
            x = x.reshape(-1,1) + K @ (z_meas - np.vstack(z))
            print(z_meas - np.vstack(z))
            # Update variance
            P = P - K @ S @ np.transpose(K)

            x_post = x

            # x_post = propogation_model_feedback(x_prior, gyro, accel, self.dt)
            self.x_fb[i,:] = x_post[:,0]
            x_prior = x_post
            print(f"i: {i}\tt: {t}")

def getSigmaPoints(x, P, n):
    """
    Function to compute sigma points used in unscented transform
    
    Args:
        x (numpy.ndarray): State vector.
        P (numpy.ndarray): Covariance matrix.
        n (int): Dimensionality of the state vector.
        t (float): Time.
    
    Returns:
        X (numpy.ndarray): Sigma point states
        wm (numpy.ndarray): weight for mean
        wc (numpy.ndarray): weight for variance
    """

    x = x.flatten()
    # Init X, i, wm, wc
    X = np.zeros([n,2*n+1])
    wm = np.zeros(2*n+1)
    wc = np.zeros(2*n+1)

    # constants
    alpha = 0.001
    k = 1
    beta = 2
    lam = (alpha**2) * (n + k) - n

    # Set point i=0
    X[:, 0] = x
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    offset = np.linalg.cholesky((n + lam) * P)
    
    for i in range(1, n+1):
        X[:,i] = x + offset[:, i-1]
        X[:,n+i] = x - offset[:, i-1]
        wm[i] = wm[n+i] = wc[i] = wc[n+i] = 0.5 / (n + lam)
    
    return X, wm, wc

def measurement_model(x):
    z = np.array([0, 0, 0, 0, 0, 0])
    z[0:3] = x[0:3]
    z[3:6] = x[6:9]
    return z
a = INS_GNSS('trajectory_data.csv')
a.execute_feedback()
# time = 0
# true_lla = 1:4
# true rpy = 4:7
# gyro_xyz = 7:10
# accel_xyz = 10:13
# z_lla = 13:16
# z_VNED = 16:19

plt.figure()
plt.subplot(3,1,1)
plt.plot(a.data_t, a.data_true_lla[:,0], '-k', label='x')
plt.plot(a.data_t, a.x_fb[:,0], '.', label='x')
plt.title('Lat vs Time')
plt.ylabel('Lat')
plt.subplot(3,1,2)
plt.plot(a.data_t, a.data_true_lla[:,1], '-k', label='x')
plt.plot(a.data_t, a.x_fb[:,1], '.', label='x')
plt.title('Lon vs Time')
plt.ylabel('Lon')
plt.subplot(3,1,3)
plt.plot(a.data_t, a.data_true_lla[:,2], '-k', label='x')
plt.plot(a.data_t, a.x_fb[:,2], '.', label='x')
plt.title('Alt vs Time')
plt.ylabel('Alt')
plt.suptitle('LLA vs Time truth and filtered')
plt.legend()

plt.figure()
plt.subplot(3,1,1)
plt.plot(a.data_t, a.data_z_VNED[:,0], '-k', label='x')
plt.plot(a.data_t, a.x_fb[:,6], '.', label='x')
plt.title('V_N vs Time')
plt.ylabel('V_N')
plt.subplot(3,1,2)
plt.plot(a.data_t, a.data_z_VNED[:,1], '-k', label='x')
plt.plot(a.data_t, a.x_fb[:,7], '.', label='x')
plt.title('V_E vs Time')
plt.ylabel('V_E')
plt.subplot(3,1,3)
plt.plot(a.data_t, a.data_z_VNED[:,2], '-k', label='x')
plt.plot(a.data_t, a.x_fb[:,8], '.', label='x')
plt.title('V_D vs Time')
plt.ylabel('V_D')
plt.legend()
plt.suptitle('VNED vs Time measured and filtered')

plt.show()