import cv2

# # import numpy as np
# import scipy
# # from matplotlib import pyplot as plt
# # from matplotlib.animation import FuncAnimation, PillowWriter

# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

# fig, ax = plt.subplots()

# def update(i):
#     # im_normed = np.random.rand(6, 6)
#     ax.imshow(data['data'][i]['img'])
#     # ax.imshow(im_normed)
#     ax.set_axis_off()

# data = scipy.io.loadmat('data/studentdata0.mat', simplify_cells=True)
# print(len(data['data']))
# l = len(data['data'])
# anim = FuncAnimation(fig, update, frames=np.arange(0, l, 10), interval=1)
# plt.show()

# # plt.ioff()

# # # ani = FuncAnimation(fig, plt.imshow(data['data'][0]['img']), frames=1000, interval=100)

# # for i in range(len(data['data'])):
# #     plt.imshow(data['data'][i+400]['img'])
# #     plt.show()

# # # plt.imshow(data['data'][0]['img'])
# # # plt.show()

# # # for i in data.keys():
# # #     print(i + ': ' + str(np.size(data[i])))
# # #     # for j in data[i].keys():
# # #     #     print()

# # # print((data['data'][1]))

# # # print(data['data'][-100]['acc'])
# # # rpy = np.vstack([d['rpy'] for d in data['data']])
# # # drpy = np.vstack([d['drpy'] for d in data['data']])
# # # acc = np.vstack([d['acc'] for d in data['data']])
# # # t = [d['t'] for d in data['data']]

# # # plt.figure()
# # # plt.subplot(3,1,1)
# # # plt.plot(t, rpy, '.')
# # # plt.subplot(3,1,2)
# # # plt.plot(t, drpy, '.')
# # # plt.subplot(3,1,3)
# # # plt.plot(t, acc, '.')
# # # plt.show()
# # # # print(data['data'][0]['t'])
# # # # save 