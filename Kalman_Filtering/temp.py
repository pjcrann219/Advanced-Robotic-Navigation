import numpy as np
from matplotlib import pyplot as plt
from kalman_classes import *
from kalman_helpers import *


mocapData = Data('kalman_filter_data_mocap.txt')

plt.figure()
plt.title('dT vs T')
plt.plot(mocapData.t[0:-1], np.diff(mocapData.t), 'r.', label='ux')
plt.legend()
plt.ylabel('dT')
plt.show()