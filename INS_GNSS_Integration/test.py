import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt('trajectory_data.csv', delimiter=',', skip_header=1)
print(len(data))

# time = 0
# true_lla = 1:4
# true rpy = 4:7
# gyro_xyz = 7:10
# accel_xyz = 10:13
# z_lla = 13:16
# z_VNED = 16:19

# plt.figure()
# plt.plot(data[:,1], data[:,2], '-k', label='truth')
# plt.plot(data[:,13], data[:,14], '.b', label='z')
# plt.legend()
# plt.show()

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(data[:,0], data[:,1], '-k', label='truth')
# plt.plot(data[:,0], data[:,13], '.r', label='z')

# plt.subplot(3,1,2)
# plt.plot(data[:,0], data[:,2], '-k', label='truth')
# plt.plot(data[:,0], data[:,14], '.r', label='z')

# plt.subplot(3,1,3)
# plt.plot(data[:,0], data[:,3], '-k', label='truth')
# plt.plot(data[:,0], data[:,15], '.r', label='z')

# plt.show()

# # plt.plot(data[:,0], data[:,], '.', label='')
# # plt.plot(data[:,0], data[:,], '.', label='')

# # plt.plot(data[:,0], data[:,], '.', label='')
# # plt.plot(data[:,0], data[:,], '.', label='')