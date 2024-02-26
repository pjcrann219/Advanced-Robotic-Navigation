import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from matplotlib.animation import FuncAnimation

def estimate_pose(data):
    
    # Check if there are no features detected
    if not isinstance(data['id'], int) and len(data['id']) == 0:
        return False, False, False

    # nIDs = number of ids in image
    nIDs = np.size(data['id'])
    # imgPoints = 1d array of x,y tuples showing feature positions in 2D camera frame
    imgPoints = np.empty((nIDs*4,), dtype=object)
    # objPoints = 1d array of x,y,z tuples showing feature positions in 3D world frame
    objPoints = np.empty((nIDs*4,), dtype=object)

    # If only 1 id, set imgPoints and objPoints manually
    if np.size(data['id']) == 1:
        # Set imgPoints for single ID
        imgPoints[0] = tuple(data['p1'])
        imgPoints[1] = tuple(data['p2'])
        imgPoints[2] = tuple(data['p3'])
        imgPoints[3] = tuple(data['p4'])
        # Set objPoints for single ID
        locs = get_id_locations(data['id'])
        objPoints[0] = locs[0]
        objPoints[1] = locs[1]
        objPoints[2] = locs[2]
        objPoints[3] = locs[3]
    else:
    # Set imgPoints and objPoints for each id seen
        for i, id in enumerate(data['id']):
            # Set imgPoints
            imgPoints[i*4] = (data['p1'][0][i], data['p1'][1][i])
            imgPoints[i*4 + 1] = (data['p2'][0][i], data['p2'][1][i])
            imgPoints[i*4 + 2] = (data['p3'][0][i], data['p3'][1][i])
            imgPoints[i*4 + 3] = (data['p4'][0][i], data['p4'][1][i])

            # set objPoints
            locs = get_id_locations(id)
            objPoints[i*4] = locs[0]
            objPoints[i*4 + 1] = locs[1]
            objPoints[i*4 + 2] = locs[2]
            objPoints[i*4 + 3] = locs[3]

    # Camera Matrix (zero-indexed):
    cameraMatrix = np.array([[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]])
    # Distortion parameters (k1, k2, p1, p2, k3):
    distParams = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])
   
    # Call solvePnP to calc q, p. Return results
    success, q, p = cv2.solvePnP(np.vstack(objPoints), np.vstack(imgPoints), cameraMatrix, distParams)

    # # convert Roll Pitch Yaw to Euler Angles
    # R = rotation_matrix(q[0], q[1], q[2])
    # euler = R.reshape((3, 3)) @ q

    # Convert from camera frame to robot frame
    pOffset = np.array([[-0.04], [0.0], [-0.03]])
    yawOffset =  - np.pi / 4
    R_yaw = np.array([[np.cos(yawOffset), -np.sin(yawOffset), 0],
                  [np.sin(yawOffset), np.cos(yawOffset), 0],
                  [0, 0, 1]])

    # print(p)
    # print(pOffset)
    pFlip = p
    pFlip[0] = -pFlip[0]
    pRotated = np.dot(R_yaw, pFlip)
    pRotated= pRotated + pOffset + np.array([[-.2], [0.035], [0]])
    # pRotated[0] = -pRotated[0]

    return success, q, p

def get_id_locations(id):
# Return (x,y,0) position of 4 corners given an id
    idMap = np.array([
        [0, 12, 24, 36, 48, 60, 72, 84,  96],
        [1, 13, 25, 37, 49, 61, 73, 85,  97],
        [2, 14, 26, 38, 50, 62, 74, 86,  98],
        [3, 15, 27, 39, 51, 63, 75, 87,  99],
        [4, 16, 28, 40, 52, 64, 76, 88, 100],
        [5, 17, 29, 41, 53, 65, 77, 89, 101],
        [6, 18, 30, 42, 54, 66, 78, 90, 102],
        [7, 19, 31, 43, 55, 67, 79, 91, 103],
        [8, 20, 32, 44, 56, 68, 80, 92, 104],
        [9, 21, 33, 45, 57, 69, 81, 93, 105],
        [10, 22, 34, 46, 58, 70, 82, 94, 106],
        [11, 23, 35, 47, 59, 71, 83, 95, 107]])
    row, column = np.where(idMap == id)
    row = int(row)
    column = int(column)

    y4 = column*0.152*2
    if column >= 3:
        y4 = y4 + (0.178 - 0.152)
    if column >= 6:
        y4 = y4 + (0.178 - 0.152)
    x4 = row*0.152*2

    p4= (x4, y4, 0.0)
    p1 = tuple(np.add(p4, (0.152, 0.0  , 0.0)))
    p2 = tuple(np.add(p4, (0.152, 0.152, 0.0)))
    p3 = tuple(np.add(p4, (0.0,   0.152, 0.0)))

    locs = np.empty((4,), dtype=object)
    locs[0] = p1
    locs[1] = p2
    locs[2] = p3
    locs[3] = p4

    return locs

def rotation_matrix(pitch, roll, yaw):
    """
    Construct the rotation matrix R given Euler angles pitch, roll, and yaw.
    
    Parameters:
        pitch (float): Pitch angle in radians.
        roll (float): Roll angle in radians.
        yaw (float): Yaw angle in radians.
    Returns:
        numpy.ndarray: The rotation matrix R.
    """
    # Calculate sine and cosine values of the angles
    sp = np.sin(pitch)
    sr = np.sin(roll)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    cr = np.cos(roll)
    cy = np.cos(yaw)
    
    # Construct the rotation matrix
    R = np.array([[cy*sp - sr*sy*sp, -cr*cy, cy*sp + cp*sr*sy],
                  [cp*sy + cy*sr*sp, cr*cy, sy*sp - cy*cp*sr],
                  [-cr*sp, sr, cr*cp]])
    
    return R

def plot_observation(dataPath):
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    for i in range(len(dataFull['data'])):
        data = dataFull['data'][i]
        success, q, p = estimate_pose(data)
        ts[i] = data['t']
        if success:
            poss[i] = np.transpose(p)
            rots[i] = np.transpose(q)

    plt.figure()
    plt.subplot(3,2,1)
    plt.title('x pos')
    plt.ylabel('m')
    plt.grid()
    plt.plot(ts, poss[:,0], '.')
    plt.subplot(3,2,3)
    plt.title('y pos')
    plt.ylabel('m')
    plt.grid()
    plt.plot(ts, poss[:,1], '.')
    plt.subplot(3,2,5)
    plt.title('z pos')
    plt.ylabel('m')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.plot(ts, poss[:,2], '.')

    plt.subplot(3,2,2)
    plt.title('yaw')
    plt.ylabel('rad')
    plt.grid()
    plt.plot(ts, rots[:,0], '.')
    plt.subplot(3,2,4)
    plt.title('pitch')
    plt.ylabel('rad')
    plt.grid()
    plt.plot(ts, rots[:,1], '.')
    plt.subplot(3,2,6)
    plt.title('roll')
    plt.ylabel('rad')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.plot(ts, rots[:,2], '.')

    plt.suptitle(dataPath.replace('data/', '').replace('.mat', ''))
    plt.tight_layout()
    plt.show()

def plot_observation3D(dataPath):
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    for i in range(len(dataFull['data'])):
        data = dataFull['data'][i]
        success, q, p = estimate_pose(data)
        ts[i] = data['t']
        if success:
            # print(data[im])
            poss[i] = np.transpose(p)
            rots[i] = np.transpose(q)
        else:
            poss[i] = np.nan
            rots[i] = np.nan

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)') 
    ax.plot3D(poss[:,0], poss[:,1], poss[:,2], '.-', label='Observations')
    ax.view_init(elev=90, azim=0)
    plt.title(dataPath.replace('data/', '').replace('.mat', ''))
    plt.legend()
    plt.show()

def estimate_covariance(dataPath):
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)

    # Get observation values
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    for i in range(len(dataFull['data'])):
        data = dataFull['data'][i]
        success, q, p = estimate_pose(data)
        ts[i] = data['t']
        if success:
            poss[i] = np.transpose(p)
            rots[i] = np.transpose(q)
        else:
            poss[i] = np.nan
            rots[i] = np.nan
    
    # Clean observation points with no values
    ts   = ts[~np.isnan(poss).any(axis=1)]
    poss = poss[~np.isnan(poss).any(axis=1)]
    rots = rots[~np.isnan(rots).any(axis=1)]

    # R = 
    # rots2 = 
    
    # Get truth (mocap) points
    truth_poss = np.transpose(dataFull['vicon'][0:3, :])
    truth_rots = np.transpose(dataFull['vicon'][3:6, :])
    truth_rots2 = np.transpose(dataFull['vicon'][3:6, :])
    truth_ts = np.transpose(dataFull['time'])

    for i in range(len(truth_rots)):
        i_rots = truth_rots[i,:]
        R = rotation_matrix(i_rots[0], i_rots[1], i_rots[2])
        truth_rots2[i,:] = R @ np.transpose(i_rots)
    # euler = R.reshape((3, 3)) @ q

    plt.figure()
    labels = ['X', 'Y', 'Z']
    colors = ['red', 'blue', 'green']

    for i in range(3):
        plt.plot(truth_ts, truth_poss[:,i], '-', label='truth'+str(labels[i]), color = colors[i])
        plt.plot(ts, poss[:,i], '.' , label=str(labels[i]), color = colors[i])
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)') 
    ax.plot3D(truth_poss[:,0], truth_poss[:,1], truth_poss[:,2], '.r', label='Truth')
    ax.plot3D(poss[:,0], poss[:,1], poss[:,2], '.g', label='Observations')
    ax.view_init(elev=90, azim=0)
    plt.title(dataPath.replace('data/', '').replace('.mat', ''))
    plt.legend()
    plt.show()

    # print(np.shape(rots))
    # print(np.shape(truth_rots))

    # plt.figure()
    # colors = ['red', 'green', 'blue']
    # plt.plot(truth_ts, truth_rots, '-', label=['truth Pitch', 'truth Yaw', 'truth Roll'])
    # plt.plot(ts, rots, '.' , label=['Pitch', 'Yaw', 'Roll'])
    # plt.legend()
    # plt.show()

    # print(np.shape(ts))
    # print(np.shape(truth_ts))
    # print(np.shape(np.vstack(truth_ts)))

    # plt.figure()
    # plt.plot(ts, poss)
    # plt.show()


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

        for i, id in enumerate(data['id']):
            # print(f"id: {id}, p1: {data['p1'][0][i]}")
            # cv2.circle((data['p1'][0][i], data['p1'][1][i]), 1, (0, 0, 255))
            xPoints = [data['p1'][0][i], data['p2'][0][i], data['p3'][0][i], data['p4'][0][i], data['p1'][0][i]]
            yPoints = [data['p1'][1][i], data['p2'][1][i], data['p3'][1][i], data['p4'][1][i], data['p1'][1][i]]
            plt.plot(xPoints, yPoints, '-r', color='red')
            plt.text(np.mean(xPoints[0:3]), np.mean(yPoints[0:3]), str(id), color='white')
            # plt.plot(data['p1'][0][i], data['p1'][1][i], 'o', color='red')
            # plt.plot(data['p2'][0][i], data['p2'][1][i], 'o', color='red')
            # plt.plot(data['p3'][0][i], data['p3'][1][i], 'o', color='red')
            # plt.plot(data['p4'][0][i], data['p4'][1][i], 'o', color='red')

        # print(data)
        # for i in range(len(data['p1'])):
        #     for j in range(len(data['p1'][i])):
        #         cv2.circle(data['p1'][i][j])

    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)
    fig, ax = plt.subplots()

    l = len(dataFull['data'])
    # anim = FuncAnimation(fig, updateAnimation, frames=np.arange(0, l, 1), interval=1)
    frames=range(10)
    anim = FuncAnimation(fig, updateAnimation, frames=range(l), interval=20, repeat=False)
    plt.title(dataPath)
    plt.show()
    return anim

# anim = animateRun('data/studentdata0.mat')
# anim.save('gifs/studentdata0.gif', writer='pillow')

estimate_covariance('data/studentdata0.mat')
# plot_observation('data/studentdata0.mat')
# plot_observation3D('data/studentdata0.mat')

# dataFull = scipy.io.loadmat('data/studentdata0.mat', simplify_cells=True)

# print(get_id_locations(48))

# 3.496, 2.636