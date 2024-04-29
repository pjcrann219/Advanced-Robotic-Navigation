import numpy as np
from earth import *
from matplotlib import pyplot as plt
from feedback_propogation_model import *
from feedforwards_propogation_model import *
from haversine import haversine

class INS_GNSS:
    def __init__(self, dataPath):
        # Initialize an instance of INS_GNSS with data from a given path.
        # Inputs:
        # - dataPath (str): Path to the data file.
        # Outputs:
        # - None
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
        self.error_fb = np.zeros([self.len, 1])
        self.error_ff = np.zeros([self.len, 1])
        self.dt = 1
        self.P = np.eye(15) * 1
        self.Q = np.eye(15) * .010
        self.R = np.eye(6) * .00010

    def execute_feedback(self, debug=True):
        # Execute the feedback architecture using the given data.
        # Inputs:
        # - debug (bool): Whether to print debug information during execution.
        # Outputs:
        # - None
        n = 15

        x_prior = np.vstack([self.data[0,1:7][:, np.newaxis], np.zeros([9,1])])
        x_prior[5,0] = 3.403

        # print(x_prior)

        P = self.P
        Q = self.Q
        R = self.R
        for i in range(self.len):
        # for i in range(2000):
            t = self.data_t[i,0]
            if debug:
                print(f"i: {i}\tt: {t}")
            P = np.eye(15)
            # P[0:3,0:3] *= 0.01
            P[3:6,3:6] *= 0.01
            # Read in gyro, accel, z_lla, z_VNED
            gyro = self.data_gyro_xyz[i, :][:, np.newaxis]
            accel = self.data_accel_xyz[i, :][:, np.newaxis]
            z_lla = self.data_z_lla[i, :]
            z_VNED = self.data_z_VNED[i, :]
            
            # Get sigma points X0 with wights wm, wc
            X0, wm, wc = getSigmaPoints(x_prior, P, n)

            # Propogate sigma points through state transition
            X1 = np.zeros(np.shape(X0))
            for j in range(2*n+1):
                thisX = X0[:,j][:, np.newaxis]
                X1[:,j] = np.squeeze(feedback_propogation_model(thisX, gyro, accel, self.dt))
            
            # Recover mean
            x = np.sum(X1 * wm, axis=1)

            # Recover variance
            P = np.zeros((n, n))
            diff2 = X1-np.vstack(X1[:,0])
            for j in range(2*n+1):
                d = diff2[:, j].reshape(-1,1)
                P += wc[j] * d @ d.T
            P += Q

            # Get new sigma points
            X2, wm, wc = getSigmaPoints(x, P, n)

            # Put sigma points through measurement model
            Z = np.zeros([6,2*n+1])
            for j in range(2*n+1):
                thisX = X2[:,j]
                Z[:,j] = measurement_model(thisX)
            
            # Recover mean
            z = np.sum(Z * wm, axis=1)

            # # Recover variance
            S = np.zeros((6,6))
            diff2 = Z-np.vstack(Z[:,0])
            for j in range(2 * n + 1):
                d = diff2[:, j].reshape(-1,1)
                S += wc[j] * d @ d.T
            S += R

            # Compute cross covariance
            diff = Z-np.vstack(z)
            
            cxz = np.zeros([n,6])
            for j in range(2 * n + 1):
                cxz += wc[j] * np.outer(X2[:,j] - x, diff[:, j])

            # Compute Kalman Gain
            # cxz[0:3,0:3] += 0.1 * np.eye(3)
            K = cxz @ np.linalg.inv(S)
            
            z_meas = np.vstack((z_lla, z_VNED)).reshape((6, 1))

            # Update estimate
            # print(x[0:3])
            # print(np.vstack((x[0:3], x[6:9])).reshape((6, 1)))
            # x_post = x.reshape(-1,1) + K @ (z_meas - np.vstack(z))
            x_post = x.reshape(-1,1) + K @ (z_meas - np.vstack((x[0:3], x[6:9])).reshape((6, 1)))

            # Trust measurement and prediction the same
            x_post = x.reshape(-1,1)
            x_post[0:3] += self.data_z_lla[i,0:3].reshape(3,1)
            x_post[0:3] /= 2
            x_post[6:9] += self.data_z_VNED[i,0:3].reshape(3,1)
            x_post[6:9] /= 2
            # Update variance
            P = P - K @ S @ np.transpose(K)

            # ## TEST
            # x_prior[3:6] = self.data_true_rpy[i, 0]
            # x_post = feedback_propogation_model(x_prior, gyro, accel, self.dt)
            if x_post[0,0] > 90:
                x_post[0,0] = 90
            if x_post[0,0] < -90:
                x_post[0,0] = -90
            self.x_fb[i,:] = x_post[:,0]
            self.error_fb[i,0] = haversine(x_post[0:2,0], self.data_true_lla[i, 0:2])
            x_prior = x_post
    def execute_feedforwards(self, debug=True):
        # Execute the feedforwards architecture using the given data.
        # Inputs:
        # - debug (bool): Whether to print debug information during execution.
        # Outputs:
        # - None
        n = 12

        x_prior = np.vstack([self.data[0,1:7][:, np.newaxis], np.zeros([6,1])])
        x_prior[5,0] = 3.403

        # print(x_prior)

        P = np.eye(12) * 1
        Q = np.eye(12) * .010
        R = np.eye(6) * .010
        for i in range(self.len):
        # for i in range(5000):
            t = self.data_t[i,0]
            if debug:
                print(f"i: {i}\tt: {t}")
            P = np.eye(12)
            # P[0:3,0:3] *= 0.01
            P[3:6,3:6] *= 0.01
            # Read in gyro, accel, z_lla, z_VNED
            gyro = self.data_gyro_xyz[i, :][:, np.newaxis]
            accel = self.data_accel_xyz[i, :][:, np.newaxis]
            z_lla = self.data_z_lla[i, :]
            z_VNED = self.data_z_VNED[i, :]
            
            # Get sigma points X0 with wights wm, wc
            # print(P)
            X0, wm, wc = getSigmaPoints(x_prior, P, n)

            # Propogate sigma points through state transition
            X1 = np.zeros(np.shape(X0))
            for j in range(2*n+1):
                thisX = X0[:,j][:, np.newaxis]
                X1[:,j] = np.squeeze(feedforwards_propogation_model(thisX, gyro, accel, self.dt))
            
            # Recover mean
            x = np.sum(X1 * wm, axis=1)

            # Recover variance
            P = np.zeros((n, n))
            diff2 = X1-np.vstack(X1[:,0])
            for j in range(2*n+1):
                d = diff2[:, j].reshape(-1,1)
                P += wc[j] * d @ d.T
            P += Q

            # Get new sigma points
            X2, wm, wc = getSigmaPoints(x, P, n)

            # Put sigma points through measurement model
            Z = np.zeros([6,2*n+1])
            for j in range(2*n+1):
                thisX = X2[:,j]
                Z[:,j] = measurement_model(thisX)
            
            # Recover mean
            z = np.sum(Z * wm, axis=1)

            # # Recover variance
            S = np.zeros((6,6))
            diff2 = Z-np.vstack(Z[:,0])
            for j in range(2 * n + 1):
                d = diff2[:, j].reshape(-1,1)
                S += wc[j] * d @ d.T
            S += R

            # Compute cross covariance
            diff = Z-np.vstack(z)
            
            cxz = np.zeros([n,6])
            for j in range(2 * n + 1):
                cxz += wc[j] * np.outer(X2[:,j] - x, diff[:, j])

            # Compute Kalman Gain
            # cxz[0:3,0:3] += 0.1 * np.eye(3)
            K = cxz @ np.linalg.inv(S)
            
            z_meas = np.vstack((z_lla, z_VNED)).reshape((6, 1))

            x_post = x.reshape(-1,1) + K @ (z_meas - np.vstack((x[0:3], x[6:9])).reshape((6, 1)))

            # Trust measurement and update the same
            x_post = x.reshape(-1,1)
            x_post[0:3] += self.data_z_lla[i,0:3].reshape(3,1)
            x_post[0:3] /= 2
            x_post[6:9] += self.data_z_VNED[i,0:3].reshape(3,1)
            x_post[6:9] /= 2


            # Update variance
            P = P - K @ S @ np.transpose(K)

            # ## TEST
            # x_prior[3:6] = self.data_true_rpy[i, 0]
            # x_post = feedback_propogation_model(x_prior, gyro, accel, self.dt)
            if x_post[0,0] > 90:
                x_post[0,0] = 90
            if x_post[0,0] < -90:
                x_post[0,0] = -90
            self.x_ff[i,:] = x_post[:,0]
            # print(self.data_true_lla[i,0:2])
            self.error_ff[i,0] = haversine(x_post[0:2,0] - x_post[9:11,0], self.data_true_lla[i,0:2])
            x_prior = x_post

    def make_fb_plots(self):
        # Create plots for the feedback architecture data.
        # Inputs:
        # - None
        # Outputs:
        # - None (plots are displayed)
        x_prior = np.vstack([self.data[0,1:7][:, np.newaxis], np.zeros([9,1])])
        
        # x_prior[5,0] = np.deg2rad(195)
        x_prior[5,0] = 3.403

        # Plot Lat and Lon
        plt.figure()
        plt.plot(self.data_true_lla[:,1], self.data_true_lla[:,0], '-k', label='True State')
        plt.plot(self.x_fb[:,1], self.x_fb[:,0], 'b.', label='Filtered State')
        plt.plot(x_prior[1,0], x_prior[0,0], 'rx')
        plt.title('Lat and Lon from Feedback Architecture')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid()
        
        # Plot Error of each state vs time
        plt.figure()
        plt.suptitle('Feedback Architecture Errors')
        plt.subplot(3,2,1)
        plt.plot(self.data_t, self.x_fb[:,0] - self.data_true_lla[:,0], label='Lat Error')
        plt.title('Latitude Error')
        plt.ylabel('Error (deg)')
        plt.subplot(3,2,3)
        plt.plot(self.data_t, self.x_fb[:,1] - self.data_true_lla[:,1], label='Lon Error')
        plt.title('Longitude Error')
        plt.ylabel('Error (deg)')
        plt.subplot(3,2,5)
        plt.plot(self.data_t, self.x_fb[:,2] - self.data_true_lla[:,2], label='Alt Error')
        plt.title('Altitude Error')
        plt.ylabel('Error (m)')
        plt.xlabel('Time (s)')
        plt.subplot(3,2,2)
        plt.plot(self.data_t, self.x_fb[:,3] - np.deg2rad(self.data_true_rpy[:,0]), label='Roll Error')
        plt.title('Roll Error')
        plt.ylabel('Error (rad)')
        plt.subplot(3,2,4)
        plt.plot(self.data_t, self.x_fb[:,4] - np.deg2rad(self.data_true_rpy[:,1]), label='Pitch Error')
        plt.title('Pitch Error')
        plt.ylabel('Error (rad)')
        plt.subplot(3,2,6)
        plt.plot(self.data_t, self.x_fb[:,5] - np.deg2rad(self.data_true_rpy[:,2]), label='Yaw Error')
        plt.title('Yaw Error')
        plt.ylabel('Error (rad)')
        plt.xlabel('Time (s)')
        plt.tight_layout()

        # Plot Haversine Error vs time
        plt.figure()
        plt.plot(self.data_t, self.error_fb, 'b', label='Haversine Error')
        plt.title('Feedback Haversine Error vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (km)')
        plt.legend()
        
        plt.show()

    def make_ff_plots(self):
        # Create plots for the feedforwards architecture data.
        # Inputs:
        # - None
        # Outputs:
        # - None (plots are displayed)

        # Plot Lat and Lon
        plt.figure()
        plt.plot(self.data_true_lla[:,1], self.data_true_lla[:,0], '-k', label='True State')
        plt.plot(self.x_ff[:,1], self.x_ff[:,0], 'b.', label='Filtered State')
        plt.title('Lat and Lon from Feedforwards Architecture')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid()
        
        # Plot Error of each state vs time
        plt.figure()
        plt.suptitle('Feedforwards Architecture Errors')
        plt.subplot(3,2,1)
        plt.plot(self.data_t, self.x_ff[:,0] - self.x_ff[:,9] - self.data_true_lla[:,0], label='Lat Error')
        plt.title('Latitude Error')
        plt.ylabel('Error (deg)')
        plt.subplot(3,2,3)
        plt.plot(self.data_t, self.x_ff[:,1] - self.x_ff[:,10] - self.data_true_lla[:,1], label='Lon Error')
        plt.title('Longitude Error')
        plt.ylabel('Error (deg)')
        plt.subplot(3,2,5)
        plt.plot(self.data_t, self.x_ff[:,2] - self.x_ff[:,11] - self.data_true_lla[:,2], label='Alt Error')
        plt.title('Altitude Error')
        plt.ylabel('Error (m)')
        plt.xlabel('Time (s)')
        plt.subplot(3,2,2)
        plt.plot(self.data_t, self.x_ff[:,3] - np.deg2rad(self.data_true_rpy[:,0]), label='Roll Error')
        plt.title('Roll Error')
        plt.ylabel('Error (rad)')
        plt.subplot(3,2,4)
        plt.plot(self.data_t, self.x_ff[:,4] - np.deg2rad(self.data_true_rpy[:,1]), label='Pitch Error')
        plt.title('Pitch Error')
        plt.ylabel('Error (rad)')
        plt.subplot(3,2,6)
        plt.plot(self.data_t, self.x_ff[:,5] - np.deg2rad(self.data_true_rpy[:,2]), label='Yaw Error')
        plt.title('Yaw Error')
        plt.ylabel('Error (rad)')
        plt.xlabel('Time (s)')
        plt.tight_layout()

        # Plot Haversine Error vs time
        plt.figure()
        plt.plot(self.data_t, self.error_ff, 'b', label='Haversine Error')
        plt.title('Feedforwards Haversine Error vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (km)')
        plt.legend()
        
        plt.show()

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
    # offset = np.sqrt((n + lam) * P)
    
    for i in range(1, n+1):
        X[:,i] = x + offset[:, i-1]
        X[:,n+i] = x - offset[:, i-1]
        wm[i] = wm[n+i] = wc[i] = wc[n+i] = 0.5 / (n + lam)
    
    return X, wm, wc

def measurement_model(x):
    # Measurement model to convert the state vector into a measurement vector.
    # Inputs:
    # - x (np.ndarray): State vector.
    # Outputs:
    # - z (np.ndarray): Measurement vector.

    z = np.array([0, 0, 0, 0, 0, 0])
    z[0:3] = x[0:3]
    z[3:6] = x[6:9]
    return z
