import numpy as np
from matplotlib import pyplot as plt

def compare_tracks(kalmans):
    colors = ['r', 'g', 'b']
    plt.figure()
    for i, kalman in enumerate(reversed(kalmans)):
        print(kalman.data)
        plt.subplot(3,1,1)
        plt.plot(kalman.data.t, kalman.bels[:,0], color = colors[i])
        plt.subplot(3,1,2)
        plt.plot(kalman.data.t, kalman.bels[:,1], color = colors[i])
        plt.subplot(3,1,3)
        plt.plot(kalman.data.t, kalman.bels[:,2], color = colors[i])

    plt.show()
    pass