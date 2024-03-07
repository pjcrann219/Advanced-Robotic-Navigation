import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

from observationModel import *
from unscentedKalmanFilter import *

# np.set_printoptions(precision=3)
dataPath = 'data/studentdata4.mat'
np.set_printoptions(suppress=True)
R = estimate_covariance(dataPath)
Q = np.eye(15)
# print(f"np.shape(R): {np.shape(R)}")
# print(f"np.shape(Q): {np.shape(Q)}")
x, t = UKF(dataPath, 1*R, .1*Q)

data = scipy.io.loadmat(dataPath, simplify_cells=True)
truth = data['vicon']
truth_t = data['time']


plt.figure()
plt.subplot(4,1,1)
plt.plot(truth_t, truth[0,:], 'r-')
plt.plot(truth_t, truth[1,:], 'b-')
plt.plot(truth_t, truth[2,:], 'g-')
plt.plot(t, x[0,:], 'r.')
plt.plot(t, x[1,:], 'b.')
plt.plot(t, x[2,:], 'g.')

plt.title('Position')

plt.subplot(4,1,2)
plt.plot(truth_t, truth[3,:], 'r-')
plt.plot(truth_t, truth[4,:], 'b-')
plt.plot(truth_t, truth[5,:], 'g-')
plt.plot(t, x[3,:], 'r.')
plt.plot(t, x[4,:], 'b.')
plt.plot(t, x[5,:], 'g.')
plt.title('Roll Pitch Yaw')

plt.subplot(4,1,3)
plt.plot(truth_t, truth[6,:], 'r-')
plt.plot(truth_t, truth[7,:], 'b-')
plt.plot(truth_t, truth[8,:], 'g-')
plt.plot(t, x[6,:], 'r.')
plt.plot(t, x[7,:], 'b.')
plt.plot(t, x[8,:], 'g.')
plt.title('Velocities')

plt.subplot(4,1,4)
plt.plot(t, x[9,:], 'r.')
plt.plot(t, x[10,:], 'b.')
plt.plot(t, x[11,:], 'g.')
plt.plot(t, x[12,:], 'r-')
plt.plot(t, x[13,:], 'b-')
plt.plot(t, x[14,:], 'g-')
plt.title('Bias')

plt.show()
# x = np.zeros([15,1])
# x[0] = 1
# x[7] = 1
# P = np.eye(15)
# n = 15
# u = np.zeros([6,1])
# # u[5] = 9.81
# dt = 1
# z = x[0:6]

# stepUFK(x, P, u, dt, n, z)

# # X, wm, wc = getSigmaPoints(x, P, n)
# # print(f"X[0,1]: {X[0,1]}, X[0,16]: {X[0,16]}")
# # plt.figure()
# # for i in range(30):
# #     plt.plot(X[1,i], X[2,i], 'x')

# # plt.show()

# # fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)') 

# for i in range(30):
#     ax.plot3D(X[0,i], X[1,i], X[2,i], 'x')

# # ax.view_init(elev=90, azim=0)
# plt.legend()
# plt.show()

# X = np.zeros([n,2*n+1])
# for i in range(1,2*n+1):
#     X[:, i]

  
# print(X)
# print(x0)
# print(state_transition(x1, u, dt))
# print(G(np.array([[0],[0],[0]])))

# plot_observation('data/studentdata0.mat')
# plot_observation3D('data/studentdata0.mat')

# dataFull = scipy.io.loadmat('data/studentdata0.mat', simplify_cells=True)

# print(get_id_locations(48))

# 3.496, 2.636