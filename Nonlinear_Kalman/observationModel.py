import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from matplotlib.animation import FuncAnimation

def estimate_pose(data):
    """
    Estimates the pose (position and orientation) of the drone relative to the world coordinate system,
    given the detected features in the camera frame.

    Parameters:
    - data (dict): A dictionary containing feature data including 'id', 'p1', 'p2', 'p3', and 'p4'.
                   'id' is the ID of the detected feature.
                   'p1', 'p2', 'p3', and 'p4' are lists containing the 2D pixel coordinates of the feature corners.

    Returns:
    - success (bool): Indicates whether the pose estimation was successful.
    - rvecIMU (numpy.ndarray): A 3x1 numpy array representing the rotation vector of the drone IMU in the world frame.
    - tvecIMU (numpy.ndarray): A 3x1 numpy array representing the translation vector of the drone IMU in the world frame.
    """

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
   
    # Call solvePnP to calc rvec and tvec. These are from camera to world
    success, rvec, tvec = cv2.solvePnP(np.vstack(objPoints), np.vstack(imgPoints), cameraMatrix, distParams)

    # Build Transformation matrix from camera to world
    Tcam_world = np.eye(4)
    Tcam_world[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    Tcam_world[0:3,3] = np.squeeze(tvec)
    Tworld_cam = invT(Tcam_world)

    r2 = np.sqrt(2)
    Timu_cam = np.array([   [ 1/r2, -1/r2,  0.0,  0.04],\
                            [-1/r2, -1/r2,  0.0,  0.0 ],\
                            [  0.0,   0.0, -1.0, -0.03],\
                            [  0.0,   0.0,  0.0,  1.0 ]])

    Tcam_imu = invT(Timu_cam)

    Tworld_imu = Tworld_cam @ Tcam_imu
    rvecIMU = decompR(Tworld_imu[0:3,0:3])
    tvecIMU = Tworld_imu[0:3, 3]
    rvecIMU[-1] *= -1
    return success, rvecIMU, tvecIMU

    # Tworld_imu2 = invT(Tworld_cam @ Tcam_imu)
    # rvec_imu2 = decompR(Tworld_imu2[0:3, 0:3])
    # tvec_imu2 = Tworld_imu2[0:3, 3]

    # print(f"rvecIMU: {rvecIMU}")
    # print(f"rvec_imu2: {rvec_imu2}")
    # print(f"tvecIMU: {tvecIMU}")
    # print(f"tvec_imu2: {tvec_imu2}")
    # rvec_imu2[-1] *= -1
    # return success, rvec_imu2, tvec_imu2

def invT(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    RT = R.T
    Tinv = np.eye(4)
    Tinv[0:3,0:3] = RT
    Tinv[0:3, 3] = -RT @ p

    return Tinv

def get_id_locations(id):
    """
    Calculates the 3D positions of the four corners of QR code given its ID.

    Parameters:
    - id (int): The ID we want positions of.

    Returns:
    - locs (numpy.ndarray): A 1D numpy array containing tuples representing the (x, y, z) positions
                            of the four corners of the QR code in the world coordinate system.
    """
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
    p1 = tuple(np.add(p4, (0.152, 0.0  , 0)))
    p2 = tuple(np.add(p4, (0.152, 0.152, 0)))
    p3 = tuple(np.add(p4, (0.0,   0.152, 0)))

    locs = np.empty((4,), dtype=object)
    locs[0] = p1
    locs[1] = p2
    locs[2] = p3
    locs[3] = p4

    return locs

def decompR(R):
    """
    Decomposes the rotation matrix R into Euler angles (roll, pitch, yaw).

    Parameters:
    - R (numpy.ndarray): A 3x3 numpy array representing the rotation matrix.

    Returns:
    - euler_angles (numpy.ndarray): A 1D numpy array containing the Euler angles (roll, pitch, yaw)
                                    in radians corresponding to the given rotation matrix R.
    """
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    yaw = np.arctan2(-R[1,0], R[0,0])
    roll = np.arctan2(R[2,1], R[2,2])

    return np.array([roll, pitch, yaw]).T

def estimate_covariance(dataPath):
    # Load in data
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)

    # Get observation values
    poss = np.zeros((len(dataFull['data']), 3))
    rots = np.zeros((len(dataFull['data']), 3))
    ts = np.zeros(len(dataFull['data']))

    # Data.rpy
    datarpy = np.zeros([len(dataFull['data']), 3])
    for i in range(len(dataFull['data'])):
        datarpy[i, :] = dataFull['data'][i]['rpy']

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

    # # # Plot truth and estimated pos
    # plt.figure()
    # plt.subplot(2,1,1)
    # labelxyz = ['X', 'Y', 'Z']
    # labelrpy = ['roll', 'pitch', 'yaw']
    # colors = ['red', 'blue', 'green']
    # for i in range(3):
    #     plt.plot(truth_ts, truth_poss[:,i], '-', label='truth'+str(labelxyz[i]), color = colors[i])
    #     plt.plot(ts, poss[:,i], '.' , label=str(labelxyz[i]), color = colors[i])
    # plt.subplot(2,1,2)
    # for i in range(3):
    #     plt.plot(truth_ts, np.degrees(truth_rots[:,i]), '-', label='truth'+str(labelrpy[i]), color = colors[i])
    #     plt.plot(ts, np.degrees(rots[:,i]), '.' , label=str(labelrpy[i]), color = colors[i])
    # plt.legend()
    # plt.show()

    # Return cov matrix
    return cov

def interpTruthData(truthX, truthT, t):
    """
    Interpolates truth data to estimate the position at a given time instant t.

    Parameters:
    - truthX (numpy.ndarray): An array containing the truth data points. Each row represents
                              a different time point, and each column represents a different
                              dimension (e.g., x, y, z coordinates).
    - truthT (numpy.ndarray): An array containing the time points corresponding to the truth data.
    - t (float): The time instant at which to estimate the position.

    Returns:
    - interpTruth (numpy.ndarray): A 1D numpy array containing the estimated states at time t.
    """
    truthT = truthT.flatten()
    x = np.interp(t, truthT, truthX[:,0])
    y = np.interp(t, truthT, truthX[:,1])
    z = np.interp(t, truthT, truthX[:,2])
    interpTruth = np.array([x, y, z])
    return interpTruth


# data_paths = [""] * 8
# for i in range(8):
#     data_paths[i] = 'data/studentdata' + str(i) + '.mat'

# for i, path in enumerate(data_paths):
#     estimate_covariance(path)
