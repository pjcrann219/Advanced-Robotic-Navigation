import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from matplotlib.animation import FuncAnimation

def animateRun(dataPath):

    def updateAnimation(i):
        data = dataFull['data'][i]
        ax.clear()
        ax.imshow(data['img'])
        ax.set_axis_off()

        xlim = [0, 350]
        ylim = [0, 275]

        plt.xlim(xlim)
        plt.ylim(ylim)

        if isinstance(data['id'], int):
            id = data['id']
            xPoints = [data['p1'][0], data['p2'][0], data['p3'][0], data['p4'][0], data['p1'][0]]
            yPoints = [data['p1'][1], data['p2'][1], data['p3'][1], data['p4'][1], data['p1'][1]]
            plt.plot(xPoints, yPoints, '-r')
            plt.text(np.mean(xPoints[0:3]), np.mean(yPoints[0:3]), str(id), color='white')
            info = "Time: " + str(round(data['t'], 3)) + "   IDs in Frame: 1"
        else:
            for i, id in enumerate(data['id']):
                # print(f"id: {id}, p1: {data['p1'][0][i]}")
                # cv2.circle((data['p1'][0][i], data['p1'][1][i]), 1, (0, 0, 255))
                xPoints = [data['p1'][0][i], data['p2'][0][i], data['p3'][0][i], data['p4'][0][i], data['p1'][0][i]]
                yPoints = [data['p1'][1][i], data['p2'][1][i], data['p3'][1][i], data['p4'][1][i], data['p1'][1][i]]
                plt.plot(xPoints, yPoints, '-r')
                plt.text(np.mean(xPoints[0:3]), np.mean(yPoints[0:3]), str(id), color='white')
                # plt.plot(data['p1'][0][i], data['p1'][1][i], 'o', color='red')
                # plt.plot(data['p2'][0][i], data['p2'][1][i], 'o', color='red')
                # plt.plot(data['p3'][0][i], data['p3'][1][i], 'o', color='red')
                # plt.plot(data['p4'][0][i], data['p4'][1][i], 'o', color='red')
        
            info = "Time: " + str(round(data['t'], 3)) + "   IDs in Frame: " + str(len(data['id']))
        plt.title(dataPath)
        plt.text(0.5, 0.95, info, transform=plt.gca().transAxes, ha='right', va='top')
        # print(data)
        # for i in range(len(data['p1'])):
        #     for j in range(len(data['p1'][i])):
        #         cv2.circle(data['p1'][i][j])

    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)
    fig, ax = plt.subplots()

    l = len(dataFull['data'])
    # anim = FuncAnimation(fig, updateAnimation, frames=np.arange(0, l, 1), interval=1)
    anim = FuncAnimation(fig, updateAnimation, frames=range(l), interval=10, repeat=False)
    
    # plt.show()
    return anim

anim = animateRun('data/studentdata6.mat')
anim.save('gifs/studentdata6.gif', writer='pillow')

anim = animateRun('data/studentdata7.mat')
anim.save('gifs/studentdata7.gif', writer='pillow')

