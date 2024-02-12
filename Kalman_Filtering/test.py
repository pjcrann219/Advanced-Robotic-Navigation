import numpy as np
from matplotlib import pyplot as plt
from kalman_classes import *
from kalman_helpers import *

R = 0.0001 * np.eye(6) # n x n, 6x6
''' Mocap                                            '''
# Load mocapData
mocapData = Data('kalman_filter_data_mocap.txt')
# Define R and Q
Q = 0.01 * np.eye(3) # k x k, 3x3
# Define Kalman Filter
kalmanMocap = Kalman(mocapData, R, Q)
# Execute filter
kalmanMocap.execute()
# Plots states
# kalmanMocap.plot_state()
# kalmanMocap.plot_state_3D()
# kalmanMocap.plot_Pdet()

''' Mocap Vel                                        '''
# Load mocapData
velData = Data('kalman_filter_data_velocity.txt')
# Define R and Q
Q = 0.01 * np.eye(3) # k x k, 3x3
C = np.eye(3,6,3)
# Define Kalman Filter
kalmanVel = Kalman(velData, R, Q, C)
# Execute filter
kalmanVel.execute()
# Plots states
# kalmanMocap.plot_state()
# # kalmanMocap.plot_state_3D()
# kalmanMocap.plot_Pdet()

# ''' Low Noise                                           '''
# Load mocapData
lowNoiseData = Data('kalman_filter_data_low_noise.txt')
# Define R and Q
Q =  0.1 * np.eye(3) # k x k, 3x3
# Define Kalman Filter
kalmanLow = Kalman(lowNoiseData, R, Q)
# Execute filter
kalmanLow.execute()
# # Plots states
# # kalmanLow.plot_state()
# # kalmanLow.plot_state_3D()
# # kalmanLow.plot_Pdet()

# ''' High Noise                                           '''
# Load mocapData
highNoiseData = Data('kalman_filter_data_high_noise.txt')
# Define R and Q
Q =  0.5 * np.eye(3) # k x k, 3x3  Measurement Covariance
# Define Kalman Filter
kalmanHigh = Kalman(highNoiseData, R, Q)
# Execute filter
kalmanHigh.execute()
# Plots states
# # kalmanHigh.plot_state()
# # kalmanHigh.plot_state_3D()
# # kalmanHigh.plot_Pdet()

# ''' Compare the 3                                           '''
# # compare_tracks([kalmanMocap, kalmanLow, kalmanHigh])
# compare_tracks_3D([kalmanMocap, kalmanLow, kalmanHigh])
# # plot_error(kalmanMocap, [kalmanLow, kalmanHigh])
plot_z_error(kalmanMocap, [kalmanLow, kalmanHigh])