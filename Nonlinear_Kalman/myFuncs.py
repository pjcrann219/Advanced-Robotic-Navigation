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
   
    # Call solvePnP to calc rvec and tvec.
    success, rvec, tvec = cv2.solvePnP(np.vstack(objPoints), np.vstack(imgPoints), cameraMatrix, distParams)

    # Convert from camera frame to object frame
    pOffset = np.array([[-0.04], [0.0], [-0.03]])
    rMat = cv2.Rodrigues(rvec)[0]
    tvec2 = -np.transpose(rMat) @ (tvec - pOffset)

    rvec2 = cv2.Rodrigues(-np.transpose(rMat))[0]
    # eulers = np.radians(cv2.decomposeProjectionMatrix(np.hstack([R, tvec]))[-1])
    # eulers[2] = eulers[2] + np.pi / 4
    # print(np.linalg.inv(R))
    pmat = np.hstack([rMat, tvec])
    roll, pitch, yaw = cv2.decomposeProjectionMatrix(pmat)[-1]
    yaw += 45
    rvec3 = np.array([roll, pitch, yaw])
    # rvec[rvec < -np.pi/2] += np.pi
    # rvec[rvec >  np.pi/2] -= np.pi
    # print(f"np.shape(eulers){np.shape(eulers)}")
    # print(f"rvec2: {rvec2}")
    # np.radians(np.array([roll, pitch, yaw + 45]))
    # rvec[:,0][rvec[:,0] < -np.pi/2] += np.pi
    # rvec[:,0][rvec[:,0] >  np.pi/2] -= np.pi
    # print(np.shape(rvec))
    # rvec[1][rvec[1] < -np.pi/4] += np.pi/2
    # rvec[1][rvec[1] >  np.pi/4] -= np.pi/2

    rvec4 = decompR(rMat)
    rvec4[1] -= np.pi
    rvec4[2] -= np.pi - np.pi/4


    return success, rvec4, tvec2

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

def decompR(R):
    roll = np.arcsin(R[2,1])
    croll = np.cos(roll)
    pitch = np.arccos(R[2,2] / croll)
    yaw = np.arccos(R[1,1] / croll)

    return np.array([roll, pitch, yaw]).T

def R(pitch, roll=-1, yaw=-1):
    if roll == -1:
        roll = pitch[1]
        yaw = pitch[2]
        pitch= pitch[0]

    spitch = np.sin(pitch)
    sroll = np.sin(roll)
    syaw = np.sin(yaw)
    cpitch = np.cos(pitch)
    croll = np.cos(roll)
    cyaw = np.cos(yaw)
    
    R = np.array([[cyaw * cpitch - sroll * syaw * spitch, -croll * syaw, cyaw * spitch + cpitch * sroll * syaw], \
            [cpitch * syaw + cyaw * sroll * spitch, croll * cyaw, syaw * spitch - cyaw * cpitch * sroll], \
            [-croll * spitch, sroll, croll * cpitch]])

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
    # Load in data
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)

    # Get observation values
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    # estimate pose
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
    
    # Get truth (mocap) points
    truth_poss = np.transpose(dataFull['vicon'][0:3, :])
    truth_rots = np.transpose(dataFull['vicon'][3:6, :])
    truth_rots2 = np.transpose(dataFull['vicon'][3:6, :])
    truth_ts = np.transpose(dataFull['time'])

    # Interpolate truth values
    truth_poss_int =  np.zeros(np.shape(poss))
    truth_rots_int =  np.zeros(np.shape(rots))
    for i, t in enumerate(ts):
        truth_poss_int[i, :] = interpTruthData(truth_poss, truth_ts, t)
        truth_rots_int[i, :] = interpTruthData(truth_rots, truth_ts, t)

    # Calculate error covariance
    x = np.hstack((poss, rots))
    truth_x_int = np.hstack((truth_poss_int, truth_rots_int))
    v = (truth_x_int - x)
    n = len(ts)
    cov = np.dot(v.T, v)/(n-1)

    # print(f"cov): {cov}")
    # cov2 = np.cov(v, rowvar=False)

    # # Plot truth and estimated pos
    plt.figure()
    plt.subplot(2,1,1)
    labelxyz = ['X', 'Y', 'Z']
    labelrpy = ['roll', 'pitch', 'yaw']
    colors = ['red', 'blue', 'green']
    for i in range(3):
        plt.plot(truth_ts, truth_poss[:,i], '-', label='truth'+str(labelxyz[i]), color = colors[i])
        plt.plot(ts, poss[:,i], '.' , label=str(labelxyz[i]), color = colors[i])
        # plt.plot(ts, truth_poss_int[:,i], '.', label='truth'+str(labels[i]), color = colors[i])
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(truth_ts, np.degrees(truth_rots[:,i]), '-', label='truth'+str(labelrpy[i]), color = colors[i])
        plt.plot(ts, np.degrees(rots[:,i]), '.' , label=str(labelrpy[i]), color = colors[i])
        # plt.plot(ts, truth_poss_int[:,i], '.', label='truth'+str(labels[i]), color = colors[i])
    plt.legend()
    plt.show()

    # Return cov matrix
    return cov

def interpTruthData(truthX, truthT, t):
    # truthT = np.vstack(truthT)
    # print(f"shape(truthX[:,0]):{np.shape(truthX[:,0])}, shape(truthT):{np.shape(truthT)}, t:{(t)}")
    # print(f"t = {t}")
    truthT = truthT.flatten()
    x = np.interp(t, truthT, truthX[:,0])
    y = np.interp(t, truthT, truthX[:,1])
    z = np.interp(t, truthT, truthX[:,2])

    return np.array([x, y, z])

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
    anim = FuncAnimation(fig, updateAnimation, frames=range(l), interval=20, repeat=False)
    plt.title(dataPath)
    plt.show()
    return anim

def UKF(data, R, Q):
    
    X0, wm, wc = getSigmaPoints(x, P, n)
    X1 = np.zeros(np.shape(X0))
    for i, thisX in enumerate(X):
        X1[:,i] = state_transition(thisX, u, dt)

    x = np.sum(X1 * wm, axis=1)


    pass

def state_transition(x0, u, dt):

    x1 = np.zeros(np.shape(x0))
    g = np.array([[0], [0], [-9.81]])

    # P1 = P0 + P0d * dt
    x1[0:3] = x0[0:3] + x0[6:9] * dt
    # q1 = q0 + Inv(G(q)) * Uw * dt
    x1[3:6] = x0[3:6] + np.linalg.inv(G(x0[6:9])) @ u[0:3] * dt
    # p1d = p0d + (g + R(q)*ua) * dt
    x1[6:9] = x0[6:9] + (g + R(x0[3:6]) @ u[3:6]) * dt
    # bg1 = bg0
    x1[9:12] = x0[9:12]
    # ba1 = ba0
    x1[12:15] = x0[12:15]

    return x1

def measurement_model(x):
    c = np.zeros([6,15])
    c[0:3, 0:3] = np.eye(3)
    c[3:6, 3:6] = np.eye(3)
    z = c @ x
    return z

def getSigmaPoints(x, P, n):
    x = x.flatten()
    # Init X, i, wm, wc
    X = np.zeros([n,2*n+1])
    wm = np.zeros(2*n+1)
    wc = np.zeros(2*n+1)

    # constants
    alpha = 0.001
    k = 1
    beta = 2
    lam = alpha**2 * (n + k) - n

    # Set point i=0
    X[:, 0] = x
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) * (1 - alpha**2 + beta)

    # Set i=1:2n
    offset = np.linalg.cholesky((n + lam) * P)
    for i in range(1, n+1):
        X[:,i] = x + offset[:, i-1]
        X[:,n+i] = x - offset[:, i-1]
        wm[i] = wm[n+i] = wc[i] = wc[n+i] = 0.5 / (n + lam)
    
    return X, wm, wc


def G(pitch, roll=999, yaw=999):
    if roll == 999:
        roll = pitch[1][0]
        yaw = pitch[2][0]
        pitch= pitch[0][0]

    # print(f'pitch: {pitch}, roll: {roll}, yaw: {yaw}')

    gMat = np.array([[np.cos(pitch),0, -np.cos(roll) * np.sin(pitch)], \
                    [0,             1, np.sin(roll)                 ], \
                    [np.sin(pitch), 0, np.cos(roll) * np.cos(pitch) ]])
    return gMat

def R(pitch, roll=999, yaw=999):
    if roll == 999:
        roll = pitch[1][0]
        yaw = pitch[2][0]
        pitch= pitch[0][0]

    spitch = np.sin(pitch)
    sroll = np.sin(roll)
    syaw = np.sin(yaw)
    cpitch = np.cos(pitch)
    croll = np.cos(roll)
    cyaw = np.cos(yaw)
    
    rMat= np.array([[cyaw * cpitch - sroll * syaw * spitch, -croll * syaw, cyaw * spitch + cpitch * sroll * syaw], \
            [cpitch * syaw + cyaw * sroll * spitch, croll * cyaw, syaw * spitch - cyaw * cpitch * sroll], \
            [-croll * spitch, sroll, croll * cpitch]])

    return rMat
## Make and save animation
# anim = animateRun('data/studentdata0.mat')
# anim.save('gifs/studentdata0.gif', writer='pillow')

## Get covariance matrix R
R = estimate_covariance('data/studentdata0.mat')
# Q = np.eye(15) * .005

x = np.ones([15,1])
P = np.eye(15)
n = 15
# for i in range(1,2*n+1):
    # print(i)
# X, wm, wc = getSigmaPoints(x, P, n)
# print(f"X[0,1]: {X[0,1]}, X[0,16]: {X[0,16]}")
# plt.figure()
# for i in range(30):
#     plt.plot(X[1,i], X[2,i], 'x')

# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)') 

# for i in range(30):
#     ax.plot3D(X[0,i], X[1,i], X[2,i], 'x')

# ax.view_init(elev=90, azim=0)
plt.legend()
plt.show()

# X = np.zeros([n,2*n+1])
# for i in range(1,2*n+1):
#     X[:, i]

  
# print(X)
# print(x0)
# print(state_transition(x1, u, dt))
# print(G(np.array([[0],[0],[0]])))

# plot_observation('data/studentdata0.mat')
# plot_observation3D('data/studentdata0.mat')

# dataFull = scipy.io.loadmat('data/studentdata0.mat', simplify_cells=True)

# print(get_id_locations(48))

# 3.496, 2.636