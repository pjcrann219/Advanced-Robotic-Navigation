import numpy as np
from matplotlib import pyplot as plt
from kalman_classes import *
from kalman_helpers import *

# Load mocapData
mocapData = Data('kalman_filter_data_mocap.txt')
# Define R and Q
R = 0.01 * np.eye(6) # n x n, 6x6
Q = 0.01 * np.eye(3) # k x k, 3x3
# Define Kalman Filter
kalmanMocaap = Kalman(mocapData, R, Q)
# Execute filter
kalmanMocaap.execute()
# Plots states
# kalmanMocaap.plot_state()

# Load mocapData
lowNoiseData = Data('kalman_filter_data_low_noise.txt')
# Define R and Q
R =  0.01 * np.eye(6) # n x n, 6x6
Q =  0.05 * np.eye(3) # k x k, 3x3
# Define Kalman Filter
kalmanLow = Kalman(lowNoiseData, R, Q)
# Execute filter
kalmanLow.execute()
# Plots states
# kalmanLow.plot_state()

# Load mocapData
highNoiseData = Data('kalman_filter_data_high_noise.txt')
# Define R and Q
R =  0.01 * np.eye(6) # n x n, 6x6
Q =  0.20 * np.eye(3) # k x k, 3x3
# Define Kalman Filter
kalmanHigh = Kalman(highNoiseData, R, Q)
# Execute filter
kalmanHigh.execute()
# Plots states
# kalmanHigh.plot_state()

compare_tracks([kalmanMocaap, kalmanLow, kalmanHigh])
