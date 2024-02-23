import numpy as np
import scipy
from matplotlib import pyplot as plt

data = scipy.io.loadmat('data/studentdata0.mat', simplify_cells=True)
print(data.keys())
for i in data.keys():
    print(i + ': ' + str(np.size(data[i])))
    # for j in data[i].keys():
    #     print()

print((data['data'][1]))

# print(data['data'][-100]['acc'])
rpy = np.vstack([d['rpy'] for d in data['data']])
drpy = np.vstack([d['drpy'] for d in data['data']])
acc = np.vstack([d['acc'] for d in data['data']])
t = [d['t'] for d in data['data']]
print(t)

plt.figure()
plt.subplot(3,1,1)
plt.plot(t, rpy, '.')
plt.subplot(3,1,2)
plt.plot(t, drpy, '.')
plt.subplot(3,1,3)
plt.plot(t, acc, '.')
plt.show()
# print(data['data'][0]['t'])
# save 