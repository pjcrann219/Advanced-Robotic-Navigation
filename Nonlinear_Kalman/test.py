import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

from observationModel import *
from unscentedKalmanFilter import *
from helperFunctions import *

# Rs = []
# for i in range(8):
#     dataPath = f'data/studentdata{i}.mat'
#     print('Ploting observations for ' + dataPath)
#     # plot_observations(dataPath)
#     R = estimate_covariance(dataPath)
#     Rs.append(estimate_covariance(dataPath))

# Rs = np.stack(Rs, axis=0)
# Rmean = np.mean(Rs, axis=0)
# print(Rmean)
# plt.show()

R = np.array([[ 0.00941694, -0.00010709, -0.00048774, -0.00107753,  0.00178652,  0.00207781],
              [-0.00010709,  0.00554322, -0.00014671, -0.00360347,  0.00164707, -0.00058822],
              [-0.00048774, -0.00014671,  0.00176918,  0.00031834,  0.00012924,  0.00021758],
              [-0.00107753, -0.00360347,  0.00031834,  0.00428488,  0.00039246, -0.00039422],
              [ 0.00178652,  0.00164707,  0.00012924,  0.00039246,  0.00692254, -0.00127304],
              [ 0.00207781, -0.00058822,  0.00021758, -0.00039422, -0.00127304,  0.00116747]])
print(R)
R = estimate_covariance('data/studentdata1.mat')
Q = .1*np.eye(15)
x, t = UKF('data/studentdata1.mat', R, .001*Q)

plot_results(x, t, 'data/studentdata1.mat')