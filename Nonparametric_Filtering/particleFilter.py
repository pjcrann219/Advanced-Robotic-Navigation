import numpy as np
from matplotlib import pyplot as plt

from observationModel import *


class ParticleFilter:
    def __init__(self, dataPath, M = 1000):
        # Init run info
        self.dataPath = dataPath # set dataPath
        self.data = scipy.io.loadmat(dataPath, simplify_cells=True)['data'] # Load data
        self.M = M  # set M, number of particles
        self.n = 15 # set n, number of states
        self.Q = np.eye(15) * 0.0001 # set process noise covariance
        self.C = np.zeros([6,15]) # set observability matrix C
        self.C[0:3, 0:3] = np.eye(3)
        self.C[3:6, 3:6] = np.eye(3)

        # Init X's and weights
        self.X = np.zeros((self.M, self.n)) # 1000x15 X_m's
        self.X[:,0:3] = np.random.uniform(low=0.0, high=2.0, size=(M, 3))                           # [0,2] xyz
        self.X[:,3:6] = np.random.uniform(low=np.deg2rad(-20), high=np.deg2rad(20), size=(M, 3))    # [-20,20] deg rpy
        self.X[:,6:9] = np.random.uniform(low=-1.0, high=1.0, size=(M, 3))                          # [-1,1] xyz_d
        self.X[:,9:12] = np.random.uniform(low=-0.5, high=0.5, size=(M, 3))                         
        self.X[:,12:15] = np.random.uniform(low=-0.5, high=0.5, size=(M, 3))

        self.w = np.ones((self.M, 1)) / self.M  # 1000x1  w_m's

        # Initial state
        self.x0 = np.zeros((1,self.n))  # 1x15 x0

        # Loop through data
        self.t = 0
        for i, _data in enumerate((self.data)):
            self.dt = self.t - _data['t'] # Set dt to t_new - t_old
            self.t = _data['t'] # set t_new
            self.i = i # for debug

            # read control input
            uw = _data['omg']
            ua = _data['acc']
            self.u = np.vstack((uw, ua)).reshape(-1, 1)

            # read measurement
            success, rvecIMU, tvecIMU = estimate_pose(_data)
            if success: #and i < 100:
                self.z = np.vstack((tvecIMU, rvecIMU)).reshape(-1, 1)

                # If we get a measurement, then step UKF
                self.stepUKF()

        

    def stepUKF(self):
        print(f"i: {self.i}/{np.size(self.data)} t: {self.t}")
        self.propogateXs()
        self.computeWeights()
        self.X_bar = np.dot(self.w.T, self.X)[0]
        self.lowVarianceSampling()
        print(f"self.X_bar: {self.X_bar}")
        self.plotPosDist()
        # Method 2
        pass

    def plotPosDist(self):
        w_range = [min(self.w), max(self.w)]
        print(w_range)
        plt.subplot(3,1,1)
        plt.cla()
        plt.plot([self.z[0], self.z[0]], w_range,'b--')
        plt.plot([self.X_bar[0], self.X_bar[0]], w_range,'r--')
        plt.subplot(3,1,2)
        plt.cla()
        plt.plot([self.z[1], self.z[1]], w_range,'b--')
        plt.plot([self.X_bar[1], self.X_bar[1]], w_range,'r--')
        plt.subplot(3,1,3)
        plt.cla()
        plt.plot([self.z[2], self.z[2]], w_range,'b--')
        plt.plot([self.X_bar[2], self.X_bar[2]], w_range,'r--')
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

        plt.pause(0.1)
        
    def plotPos2d(self):
        for m in range(self.M):
            plt.plot(self.X[m,0], self.X[m,1], '.r')

        plt.pause(0.1)
        pass

    def lowVarianceSampling(self):
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
        pass

    def computeWeights(self):
        for m in range(self.M):

            x = self.X[m,:].reshape((self.n,1)) # grab particle (1,15) and reshape (15,1)
            z_predicted = self.C @ x

            # TODO: include rest of gaussion dist equation?
            # likelihoods = np.exp(-0.5 * np.sum((self.z - z_predicted)**2)).reshape(1,1)
            self.w[m] = np.exp(-0.5 * np.sum((self.z - z_predicted)**2)).reshape(1,1)

        self.w /= np.sum(self.w)

    def getRmatGmat(self, x0):

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

        # Loop through all particles
        for m in range(self.M):

            x0 = self.X[m,:].reshape((self.n,1)) # grab particle (1,15) and reshape (15,1)
            # print(f"np.shape(x0[3:6]): {np.shape(x0[3:6])}")
            # print(f"x0[3:6]: {x0[3:6]}")
            rMat, gMat = self.getRmatGmat(x0) # get rMat and gMat
            # rMat = R(x0[3:6])
            # gMat = G(x0[3:6])
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
            # print(f"noise[0:3]: {noise[0:3]}")
            # x_t+1 = x_t + xd_t * dt
            x1 = x0 + (xd + noise) * self.dt
            self.X[m,:] = x1.reshape((1, self.n)) # Reshape x1 and set X_m

dataPath = 'data/studentdata1.mat'
a = ParticleFilter(dataPath, M = 250)
# a.doUKF()
