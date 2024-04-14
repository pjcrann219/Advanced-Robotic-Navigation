import numpy as np
from matplotlib import pyplot as plt
from observationModel import *
import time

class ParticleFilter:
    def __init__(self, dataPath, M = 1000, stop = -1, debug=False):
        """
        Initialize the ParticleFilter class.

        Parameters:
        - dataPath: Path to the data file.
        - M: Number of particles.
        - stop: Stop index for quicker testing.
        - debug: Debug flag.
        """
        # Init run info
        self.dataPath = dataPath # set dataPath
        dataFull = scipy.io.loadmat(dataPath, simplify_cells=True) # Load data
        self.data = dataFull['data']
        self.truth_x = dataFull['vicon']
        self.truth_t = dataFull['time']

        if stop == -1:
            self.stop = len(self.data)
        else:
            self.stop = stop

        self.debug = debug
        self.M = M  # set M, number of particles
        self.n = 15 # set n, number of states

        if 'data0' in self.dataPath:
            self.is0 = True
        else:
            self.is0 = False

        # self.Q = np.diag([0.01] * 3 + [0.002] * 3 + [.1] * 3 + [.001] * 6) * 500
        # self.R = np.diag([0.00558, 0.00558, 0.00558, 0.00412, 0.00412, 0.00412])
        self.Q = np.diag([0.01] * 3 + [0.002] * 3 + [.1] * 3 + [.001] * 6) * 500
        self.R = np.diag([0.00558, 0.00558, 0.00558, 0.00412, 0.00412, 0.00412])

        self.C = np.zeros([6,15]) # set observability matrix C
        self.C[0:3, 0:3] = np.eye(3)
        self.C[3:6, 3:6] = np.eye(3)

        self.X_hist_weighted_average = np.zeros((np.size(self.data), self.n))
        self.X_hist_average = np.zeros((np.size(self.data), self.n))
        self.X_hist_highest = np.zeros((np.size(self.data), self.n))
        self.t_hist = np.zeros((np.size(self.data), 1))
        self.z_hist = np.zeros((np.size(self.data), 6))
        
        # Init X's and weights
        self.X = np.zeros((self.M, self.n)) # 1000x15 X_m's
        self.X[:,0:3] = np.random.uniform(low=0.0, high=2.0, size=(M, 3))                           # [0,2] xyz
        self.X[:,3:6] = np.random.uniform(low=np.deg2rad(-20), high=np.deg2rad(20), size=(M, 3))    # [-20,20] deg rpy
        self.X[:,6:9] = np.random.uniform(low=-0.5, high=0.5, size=(M, 3))                          # [-1,1] xyz_d
        self.X[:,9:12] = np.random.uniform(low=-0.5, high=0.5, size=(M, 3))                         
        self.X[:,12:15] = np.random.uniform(low=-0.5, high=0.5, size=(M, 3))
        # self.X_hist = np.zeros((len(self.data), self.M, self.n))

        self.w = np.ones((self.M, 1)) / self.M  # 1000x1  w_m's
        # self.w_hist = np.zeros((self.M, len(self.data)))

        # Initial state
        self.x0 = np.zeros((1,self.n))  # 1x15 x0

        self.X_weighted_average = np.zeros((15))
        self.X_average = np.zeros((15))
        self.X_highest = np.zeros((15))
        self.z = np.zeros((6))

        # Loop through data
        self.t = 0
        for i, _data in enumerate(self.data):
            self.dt = self.t - _data['t'] # Set dt to t_new - t_old
            self.t = _data['t'] # set t_new
            self.i = i # for debug

            # read control input
            ua = _data['acc']
            if self.is0:
                uw = _data['drpy']
            else:
                uw = _data['omg']
            self.u = np.vstack((uw, ua)).reshape(-1, 1)

            # read measurement
            success, rvecIMU, tvecIMU = estimate_pose(_data)
            if success and i < self.stop:
                self.z = np.vstack((tvecIMU, rvecIMU)).reshape(-1, 1)

                # If we get a measurement, then step UKF
                self.stepPF()

            # Record history of X_bar, t, and z
            self.X_hist_weighted_average[i, :] = self.X_weighted_average
            self.X_hist_average[i, :] = self.X_average
            self.X_hist_highest[i, :] = self.X_highest
            self.t_hist[i] = self.t
            self.z_hist[i, :] = self.z.T


    def stepPF(self):
        """
        Perform one step of the Particle Filter.
        """
        # if self.debug == True:
        # print(f"i: {self.i}/{np.size(self.data)} t: {self.t}")

        # Propogate particles
        self.propogateXs()

        # Compute weights based on measurement
        self.computeWeights()

        # Calculate weighted average
        self.X_weighted_average = np.dot(self.w.T, self.X)[0]
        # Calculate average
        self.X_average = np.mean(self.X, axis=0)
        # Calculate highest weight
        maxIdx = np.argmax(self.w)
        self.X_highest = self.X[maxIdx,:]
        
        # Resample particles using low variance sampling
        self.lowVarianceSampling()

        # self.plotPosDist()

        # dt4 = time.time() - start_time
        # print(f"propogate: {dt1}, computeWeights: {dt2}, average: {dt3}, resample: {dt4}")

    def plotPosDist(self):
        w_range = [min(self.w), max(self.w)]
        print(w_range)
        plt.suptitle('t: ' + str(self.t))
        plt.subplot(3,1,1)
        plt.cla()
        plt.plot([self.z[0], self.z[0]], w_range,'b--')
        plt.plot([self.X_weighted_average[0], self.X_weighted_average[0]], w_range,'r--')
        plt.subplot(3,1,2)
        plt.cla()
        plt.plot([self.z[1], self.z[1]], w_range,'b--')
        plt.plot([self.X_weighted_average[1], self.X_weighted_average[1]], w_range,'r--')
        plt.subplot(3,1,3)
        plt.cla()
        plt.plot([self.z[2], self.z[2]], w_range,'b--')
        plt.plot([self.X_weighted_average[2], self.X_weighted_average[2]], w_range,'r--')
        for m in range(self.M):
            plt.subplot(3,1,1)
            plt.plot(self.X[m,0], self.w[m], '.r')
            plt.subplot(3,1,2)
            plt.plot(self.X[m,1], self.w[m], '.r')
            plt.subplot(3,1,3)
            plt.plot(self.X[m,2], self.w[m], '.r')

        plt.subplot(3,1,1)
        plt.xlim([-1,3])
        plt.subplot(3,1,2)
        plt.xlim([-1,3])
        plt.subplot(3,1,3)
        plt.xlim([-0.5,3])

        plt.pause(0.01)
        
    def plotPosDistVel(self):
        w_range = [min(self.w), max(self.w)]
        plt.suptitle('Velocity, t: ' + str(self.t))
        plt.subplot(3,1,1)
        plt.cla()
        plt.plot([self.current_truth[6], self.current_truth[6]], w_range,'b--')
        plt.plot([self.X_bar[6], self.X_bar[6]], w_range,'r--')
        plt.subplot(3,1,2)
        plt.cla()
        plt.plot([self.current_truth[7], self.current_truth[7]], w_range,'b--')
        plt.plot([self.X_bar[7], self.X_bar[7]], w_range,'r--')
        plt.subplot(3,1,3)
        plt.cla()
        plt.plot([self.current_truth[8], self.current_truth[8]], w_range,'b--')
        plt.plot([self.X_bar[8], self.X_bar[8]], w_range,'r--')
        for m in range(self.M):
            plt.subplot(3,1,1)
            plt.plot(self.X[m,6], self.w[m], '.r')
            plt.subplot(3,1,2)
            plt.plot(self.X[m,7], self.w[m], '.r')
            plt.subplot(3,1,3)
            plt.plot(self.X[m,8], self.w[m], '.r')

        plt.subplot(3,1,1)
        plt.xlim([-3,3])
        plt.subplot(3,1,2)
        plt.xlim([-3,3])
        plt.subplot(3,1,3)
        plt.xlim([-3,3])

        plt.pause(0.01)
        
    def plotPos2d(self):
        for m in range(self.M):
            plt.plot(self.X[m,0], self.X[m,1], '.r')

        plt.pause(0.1)
        pass

    def lowVarianceSampling(self):
        """
        Performs low variance sampling to select particles based on their weights.
        """
        newX = np.zeros(np.shape(self.X))
        Minv = 1/self.M
        r = np.random.rand() * Minv
        c = self.w[0]
        i = 0

        for m in range(self.M):
            U = r + (m)*Minv
            while U > c:
                i = i + 1
                c = c + self.w[i]
            newX[m,:] = self.X[i,:] 

        self.X = newX

    def computeWeights(self):
        """
        Computes new weights for each particle based on p(z|x)
        """
        for m in range(self.M):

            x = self.X[m,:].reshape((self.n,1)) # grab particle (1,15) and reshape (15,1)
            z_predicted = self.C @ x # Pass through observability matrix C

            # Compute p(z|x) assuming z follows a normal dist
            constant = 1.0 / ((2 * np.pi) ** (self.n / 2) * np.linalg.det(self.R) ** 0.5)
            diff = self.z - z_predicted
            exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.R)), diff)
            self.w[m] = constant * np.exp(exponent)

        if self.debug:
            print(f"np.sum(self.w): {np.sum(self.w)}")

        # normalize all weights
        self.w /= np.sum(self.w)


    def plotResults(self, sampling='weighted_average'):
        """
        Plots filtered states vs time according to sampling method.
        """
        if sampling == 'highest':
            x_bar = self.X_hist_highest
        elif sampling == 'average':
            x_bar = self.X_hist_average
        elif sampling == 'weighted_average':
            x_bar = self.X_hist_weighted_average
        else:
            raise ValueError('Invalid sampling method!')
        pass
    
        plt.figure()
        plt.suptitle(self.dataPath +  '\n' + sampling + ' M = ' + str(self.M))
        plt.subplot(3,1,1)
        plt.plot(self.truth_t, self.truth_x[0,:], '-k', label='truth')
        plt.plot(self.t_hist, x_bar[:,0], 'r.', label='filtered')
        plt.plot(self.t_hist, self.z_hist[:,0], 'b.', label='observation')
        plt.xlabel('Time')
        plt.ylabel('X (m)')

        plt.subplot(3,1,2)
        plt.plot(self.truth_t, self.truth_x[1,:], '-k', label='truth')
        plt.plot(self.t_hist, x_bar[:,1], 'r.', label='filtered')
        plt.plot(self.t_hist, self.z_hist[:,1], 'b.', label='observation')
        plt.xlabel('Time')
        plt.ylabel('Y (m)')

        plt.subplot(3,1,3)
        plt.plot(self.truth_t, self.truth_x[2,:], '-k', label='truth')
        plt.plot(self.t_hist, x_bar[:,2], 'r.', label='filtered')
        plt.plot(self.t_hist, self.z_hist[:,2], 'b.', label='observation')
        plt.xlabel('Time')
        plt.ylabel('Z (m)')
        plt.legend()
        plt.show()

        # plt.figure()
        # plt.suptitle(self.dataPath + '\nM = ' + str(self.M))
        # plt.subplot(3,1,1)
        # plt.plot(self.truth_t, np.rad2deg(self.truth_x[3,:]), '-k', label='truth')
        # plt.plot(self.t_hist, np.rad2deg(x_bar[:,3]), 'r.', label='filtered')
        # plt.plot(self.t_hist, np.rad2deg(self.z_hist[:,3]), 'b.', label='observation')
        # plt.xlabel('Time')
        # plt.ylabel('Roll (deg)')

        # plt.subplot(3,1,2)
        # plt.plot(self.truth_t, np.rad2deg(self.truth_x[4,:]), '-k', label='truth')
        # plt.plot(self.t_hist, np.rad2deg(x_bar[:,4]), 'r.', label='filtered')
        # plt.plot(self.t_hist, np.rad2deg(self.z_hist[:,4]), 'b.', label='observation')
        # plt.xlabel('Time')
        # plt.ylabel('Pitch (deg)')

        # plt.subplot(3,1,3)
        # plt.plot(self.truth_t, np.rad2deg(self.truth_x[5,:]), '-k', label='truth')
        # plt.plot(self.t_hist, np.rad2deg(x_bar[:,5]), 'r.', label='filtered')
        # plt.plot(self.t_hist, np.rad2deg(self.z_hist[:,5]), 'b.', label='observation')
        # plt.xlabel('Time')
        # plt.ylabel('Yaw (deg)')
        # plt.legend()

        # plt.figure()
        # plt.suptitle(self.dataPath + '\nM = ' + str(self.M))
        # plt.subplot(3,1,1)
        # plt.plot(self.truth_t, self.truth_x[6,:], '-k', label='truth')
        # plt.plot(self.t_hist, x_bar[:,6], 'r.', label='filtered')
        # plt.xlabel('Time')
        # plt.ylabel('Xd (m)')

        # plt.subplot(3,1,2)
        # plt.plot(self.truth_t, self.truth_x[7,:], '-k', label='truth')
        # plt.plot(self.t_hist, x_bar[:,7], 'r.', label='filtered')
        # plt.xlabel('Time')
        # plt.ylabel('Yd(m)')

        # plt.subplot(3,1,3)
        # plt.plot(self.truth_t, self.truth_x[8,:], '-k', label='truth')
        # plt.plot(self.t_hist, x_bar[:,8], 'r.', label='filtered')
        # plt.xlabel('Time')
        # plt.ylabel('Zd (m)')
        # plt.legend()

        # plt.figure()
        # plt.suptitle(self.dataPath + '\nM = ' + str(self.M))
        # plt.subplot(3,1,1)
        # plt.plot(self.t_hist, x_bar[:,9], 'r.', label='accel')
        # plt.plot(self.t_hist, x_bar[:,12], 'g.', label='gyro')
        # plt.xlabel('Time')
        # plt.ylabel('bias X (m/s^2, rad/s)')

        # plt.subplot(3,1,2)
        # plt.plot(self.t_hist, x_bar[:,10], 'r.', label='accel')
        # plt.plot(self.t_hist, x_bar[:,13], 'g.', label='gyro')
        # plt.xlabel('Time')
        # plt.ylabel('bias Y (m/s^2, rad/s)')

        # plt.subplot(3,1,3)
        # plt.plot(self.t_hist, x_bar[:,11], 'r.', label='accel')
        # plt.plot(self.t_hist, x_bar[:,14], 'g.', label='gyro')
        # plt.xlabel('Time')
        # plt.ylabel('bias Z (m/s^2, rad/s)')
        # plt.legend()

        plt.show()

    def plotResults3D(self, sampling='weighted_average'):
        """
        Plots filter results in 3D according to sampling method.
        """
        if sampling == 'highest':
            x_bar = self.X_hist_highest
        elif sampling == 'average':
            x_bar = self.X_hist_average
        elif sampling == 'weighted_average':
            x_bar = self.X_hist_weighted_average
        else:
            raise ValueError('Invalid sampling method!')
        pass

        fig = plt.figure()
        plt.suptitle(self.dataPath +  '\n' + sampling + ' M = ' + str(self.M))
        ax = fig.add_subplot(111, projection='3d')

        plt.plot(self.truth_x[0,:], self.truth_x[1,:], 'k-', zs=self.truth_x[2,:], label='truth')
        plt.plot(x_bar[:,0], x_bar[:,1], 'b-', zs=x_bar[:,2], label='filtered')
        plt.plot(self.z_hist[:,0], self.z_hist[:,1], 'r.', zs=self.z_hist[:,2], label='observation')
        plt.legend()

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_aspect('equal')

        plt.show()

    def getRmatGmat(self, x0):
        """
        Computes the rotation matrix (rMat) and gravity rotation matrix (gMat) based on the given state.

        Args:
            x0 (ndarray): State vector containing roll, pitch, and yaw angles.

        Returns:
            rMat, gMat: Rotation matrix (rMat) and the gravity rotation matrix (gMat).
        """
        sroll, spitch, syaw = np.sin(x0[3:6]).T[0]
        croll, cpitch, cyaw = np.cos(x0[3:6]).T[0]

        rMat= np.array([[cyaw * cpitch - sroll * syaw * spitch, -croll * syaw, cyaw * spitch + cpitch * sroll * syaw], \
                [cpitch * syaw + cyaw * sroll * spitch, croll * cyaw, syaw * spitch - cyaw * cpitch * sroll], \
                [-croll * spitch, sroll, croll * cpitch]])

        gMat = np.array([[np.cos(x0[4,0]),0, -np.cos(x0[3,0]) * np.sin(x0[4,0])], \
                        [0,             1,                  np.sin(x0[3,0])], \
                        [np.sin(x0[4,0]), 0,  np.cos(x0[3,0]) * np.cos(x0[4,0])]])

        return rMat, gMat

    def propogateXs(self):
        """
        Propagates the state particles forward in time using the process model.

        This method loops through all particles and updates their state by applying the process model
        and adding process noise.

        The process model computes the next state based on the current state (x0) and control inputs (u).

        Process model:
        - x_t+1 = x_t + (xd_t + noise) * dt
        """
        # Loop through all particles
        for m in range(self.M):

            x0 = self.X[m,:].reshape((self.n,1)) # grab particle (1,15) and reshape (15,1)
            rMat, gMat = self.getRmatGmat(x0) # get rMat and gMat
            g = np.array([[0], [0], [-9.81]]) # gravity vector

            # Build xd
            xd = np.zeros(np.shape(x0))
            xd[0:3]   = x0[6:9]
            xd[3:6]   = np.linalg.inv(gMat) @ (self.u[0:3] + x0[9:12])
            xd[6:9]   = g + rMat @ (self.u[3:6] + x0[12:15])
            xd[9:12]  = np.zeros((3,1))
            xd[12:15] = np.zeros((3,1))

            # process noise
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q).reshape(-1, 1)

            # x_t+1 = x_t + xd_t * dt
            x1 = x0 + (xd + noise) * self.dt
            self.X[m,:] = x1.reshape((1, self.n)) # Reshape x1 and set X_m

    def computeRMSE(self):
        """
        Computes the Root Mean Square Error (RMSE) between the estimated states and the ground truth states.

        This method interpolates the ground truth states to match the time points of the estimated states.
        It then calculates the errors and RMSE values for position and rotation, both weighted and unweighted averages.
        """
        # Set self.truth_z_interp and self,truth_t_interp
        self.truth_z_interp = np.zeros((np.size(self.data), 6))
        self.truth_t_interp = np.zeros((np.size(self.data), 1))
        for i, t in enumerate(self.t_hist):
            t = t[0]

            x = np.interp(t, self.truth_t.flatten(), self.truth_x[0,:])
            y = np.interp(t, self.truth_t.flatten(), self.truth_x[1,:])
            z = np.interp(t, self.truth_t.flatten(), self.truth_x[2,:])
            roll = np.interp(t, self.truth_t.flatten(), self.truth_x[3,:])
            pitch = np.interp(t, self.truth_t.flatten(), self.truth_x[4,:])
            yaw = np.interp(t, self.truth_t.flatten(), self.truth_x[5,:])
            
            self.truth_z_interp[i,:] = np.array([x, y, z, roll, pitch, yaw])
            self.truth_t_interp[i] = t

        # Compute and display errors and RMSE values by sampling method
        print(f"RMSE for {self.dataPath} with {self.M} particles")
        self.error_weighted_average = self.X_hist_weighted_average[:,0:6] - self.truth_z_interp
        self.error_pos_weighted_average = np.sqrt(self.error_weighted_average[:,0]**2 + self.error_weighted_average[:,1]**2 + self.error_weighted_average[:,2]**2)
        self.error_rot_weighted_average = np.sqrt(self.error_weighted_average[:,3]**2 + self.error_weighted_average[:,4]**2 + self.error_weighted_average[:,5]**2)
        self.RMSE_weighted_average = np.sqrt(self.error_pos_weighted_average ** 2 + self.error_rot_weighted_average **2)
        self.RMSE_pos_weighted_average = np.sqrt(np.mean(self.error_pos_weighted_average ** 2))
        self.RMSE_rot_weighted_average = np.sqrt(np.mean(self.error_rot_weighted_average ** 2))
        print("\tWeighted Average:")
        print(f"\t\tPosition: {np.round(self.RMSE_pos_weighted_average, 3)} m \tRotation: {np.round(self.RMSE_rot_weighted_average, 3)} rad")

        self.error_average = self.X_hist_average[:,0:6] - self.truth_z_interp
        self.error_pos_average = np.sqrt(self.error_average[:,0]**2 + self.error_average[:,1]**2 + self.error_average[:,2]**2)
        self.error_rot_average = np.sqrt(self.error_average[:,3]**2 + self.error_average[:,4]**2 + self.error_average[:,5]**2)
        self.RMSE_average = np.sqrt(self.error_pos_average ** 2 + self.error_rot_average **2)
        self.RMSE_pos_average = np.sqrt(np.mean(self.error_pos_average ** 2))
        self.RMSE_rot_average = np.sqrt(np.mean(self.error_rot_average ** 2))
        print("\tAverage:")
        print(f"\t\tPosition: {np.round(self.RMSE_pos_average, 3)} m \tRotation: {np.round(self.RMSE_rot_average, 3)} rad")

        self.error_highest = self.X_hist_highest[:,0:6] - self.truth_z_interp
        self.error_pos_highest = np.sqrt(self.error_highest[:,0]**2 + self.error_highest[:,1]**2 + self.error_highest[:,2]**2)
        self.error_rot_highest = np.sqrt(self.error_highest[:,3]**2 + self.error_highest[:,4]**2 + self.error_highest[:,5]**2)
        self.RMSE_highest = np.sqrt(self.error_pos_highest ** 2 + self.error_rot_highest **2)
        self.RMSE_pos_highest = np.sqrt(np.mean(self.error_pos_highest ** 2))
        self.RMSE_rot_highest = np.sqrt(np.mean(self.error_rot_highest ** 2))
        print("\tHighest:")
        print(f"\t\tPosition: {np.round(self.RMSE_pos_highest, 3)} m \tRotation: {np.round(self.RMSE_rot_highest, 3)} rad")

    def plotError(self):
        """
        Plots the total error versus time for different sampling methods.

        This method plots the RMSE values computed for weighted average, average, and highest sampling methods
        vs time.
        """
        plt.figure()
        plt.plot(self.t_hist, self.RMSE_weighted_average, label='weighted_average')
        plt.plot(self.t_hist, self.RMSE_average, label='average')
        plt.plot(self.t_hist, self.RMSE_highest, label='highest')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Total Error')
        plt.title(f"Error vs Time by sampling method\n{self.dataPath} with {self.M} particles")
        plt.show()

