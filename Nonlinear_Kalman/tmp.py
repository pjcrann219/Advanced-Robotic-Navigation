import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from observationModel import *
from unscentedKalmanFilter import *

angles = np.array([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]) * np.pi
# angles = 3*np.pi
angles2 = capPi(angles)

plt.figure()
plt.plot(angles, '.', label='orig')
plt.plot(angles2, '.', label='cap')
plt.legend()
plt.show()