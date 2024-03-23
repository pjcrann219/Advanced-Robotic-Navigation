import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from observationModel import *

def UKF(dataPath, R, Q):
    """
    Top level function to perform Unscented Kalman Filter

    Parameters:
        dataPath (str): Path to the data file.

    Returns:
        xs (np.ndarray): history of filtered states
        ts (np.ndarray): time stamps
    """
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)
    if 'data0' in dataPath:
        is0 = True
    else:
        is0 = False
        
    data = dataFull['data']
    n = 15

    x = np.zeros([15,1])
    P = np.eye(15)

    xs = np.zeros([15, len(data)])
    ts = np.zeros([len(data), 1])

    zold = x[0:6]
    
    uw = np.zeros(3)
    ua = np.zeros(3)
    ua[2] = -9.81

    last_t = 0
    for i in range(len(data)):

        t = data[i]['t']
        dt = t - last_t
        success, rvecIMU, tvecIMU = estimate_pose(data[i])
        if success:
            z = np.concatenate((tvecIMU, rvecIMU.T)).reshape(-1,1)
            if is0:
                uw = data[i]['drpy']
            else:
                uw = data[i]['omg']

            ua = data[i]['acc']
            u = np.concatenate((uw, ua)).reshape(-1,1)
            x, P = stepUKF(x, P, u, dt, n, z, t, R, Q)

        xs[:,i] = np.squeeze(x)
        ts[i] = t
        last_t = t
    
    return xs, ts

def stepUKF(x, P, u, dt, n, measurement, t, R = np.eye(6), Q = np.eye(15)):
    """
    Function to perform a timestep update of UKF

    Parameters:
        x (numpy.ndarray): State vector.
        P (numpy.ndarray): Covariance matrix.
        u (numpy.ndarray): Control input.
        dt (float): delta time.
        n (int): Dimensionality of the state vector.
        measurement (numpy.ndarray): Measurement vector.
        t (float): Time. (used for debugging)
        R (numpy.ndarray, optional): Measurement noise covariance matrix. Defaults to identity matrix.
        Q (numpy.ndarray, optional): Process noise covariance matrix. Defaults to identity matrix.
    
    Returns:
        tuple: Updated state vector and covariance matrix.
    """

    # Get sigma points X0 with wights wm, wc
    X0, wm, wc = getSigmaPoints(x, P, n, t)

    # Propogate sigma points through state transition
    X1 = np.zeros(np.shape(X0))
    for i in range(2*n+1):
        thisX = X0[:,i]
        X1[:,i] = state_transition(thisX, u, dt)

    # Recover mean
    x = np.sum(X1 * wm, axis=1)
    # Recover variance
    diff = X1-np.vstack(x)
    P = np.zeros((15, 15))
    diff2 = X1-np.vstack(X1[:,0])
    for i in range(2 * n + 1):
        d = diff2[:, i].reshape(-1,1)
        P += wc[i] * d @ d.T
    P += Q

    # Get new sigma points
    X2, wm, wc = getSigmaPoints(x, P, n, t)

    # Put sigma points through measurement model
    Z = np.zeros([6,2*n+1])
    for i in range(2*n+1):
        thisX = X2[:,i]
        Z[:,i] = measurement_model(thisX)
    
    # Recover mean
    z = np.sum(Z * wm, axis=1)

    # Recover variance
    S = np.zeros((6,6))

    diff2 = Z-np.vstack(Z[:,0])
    for i in range(2 * n + 1):
        d = diff2[:, i].reshape(-1,1)
        S += wc[i] * d @ d.T
    S += R

    diff = Z-np.vstack(z)
    # Compute cross covariance
    cxz = np.zeros([15,6])
    for i in range(2 * n + 1):
        cxz += wc[i] * np.outer(X2[:,i] - x, diff[:, i])

    # Compute Kalman Gain
    K = cxz @ np.linalg.inv(S)

    # Update estimate
    x = x.reshape(-1,1) + K @ (measurement - np.vstack(z))

    # Update variance
    P = P - K @ S @ np.transpose(K)

    return x, P

def is_positive_semidefinite(matrix):
    # Used to check positive semidefiniteness of a matrix durring debug

    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(matrix)
    
    # Check if all eigenvalues are non-negative
    if np.all(eigenvalues >= 0):
        print("PASES positive semidefinite test")
        return True
    else:
        print("FAILS positive semidefinite test")
        return False
    

def state_transition(x0, u, dt):
    """
    Function to perform state transition of the system.
    
    Args:
    x0 (numpy.ndarray): State vector at time t.
    u (numpy.ndarray): Control input vector.
    dt (float): Time step.
    
    Returns:
    numpy.ndarray: State vector at time t+1.
    """

    # Calc rotation matrix R and gravity matrix G
    rMat = R(x0[3:6])
    gMat = G(x0[3:6])

    # Convert to 2d array for easier linalg
    x0 = x0.reshape((15,1))

    # Initialize xd and x1
    xd = np.zeros(np.shape(x0))
    x1 = np.zeros(np.shape(x0))

    # init g
    g = np.array([[0], [0], [-9.81]])

    # Build xd
    xd[0:3]   = x0[6:9]
    xd[3:6]   = np.linalg.inv(gMat) @ (u[0:3] + x0[9:12])
    xd[6:9]   = g + rMat @ (u[3:6] + x0[12:15])
    xd[9:12]  = np.zeros((3,1))
    xd[12:15] = np.zeros((3,1))

    # x_t+1 = x_t + xd_t * dt
    x1 = x0 + xd * dt

    # Return a 1d array
    return np.squeeze(x1)

def measurement_model(x):
    """
    Function to compute measurements based on current state
    
    Args:
        x (numpy.ndarray): State vector.
    
    Returns:
        numpy.ndarray: measurement
    """
    c = np.zeros([6,15])
    c[0:3, 0:3] = np.eye(3)
    c[3:6, 3:6] = np.eye(3)
    z = c @ x
    return z

def getSigmaPoints(x, P, n, t):
    """
    Function to compute sigma points used in unscented transform
    
    Args:
        x (numpy.ndarray): State vector.
        P (numpy.ndarray): Covariance matrix.
        n (int): Dimensionality of the state vector.
        t (float): Time.
    
    Returns:
        X (numpy.ndarray): Sigma point states
        wm (numpy.ndarray): weight for mean
        wc (numpy.ndarray): weight for variance
    """

    x = x.flatten()
    # Init X, i, wm, wc
    X = np.zeros([n,2*n+1])
    wm = np.zeros(2*n+1)
    wc = np.zeros(2*n+1)

    # constants
    alpha = 0.001
    k = 1
    beta = 2
    lam = (alpha**2) * (n + k) - n

    # Set point i=0
    X[:, 0] = x
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    offset = np.linalg.cholesky((n + lam) * P)

    if np.imag(offset).any():
        print(f"Negative offset at t: {t}")
    
    for i in range(1, n+1):
        X[:,i] = x + offset[:, i-1]
        X[:,n+i] = x - offset[:, i-1]
        wm[i] = wm[n+i] = wc[i] = wc[n+i] = 0.5 / (n + lam)
    
    return X, wm, wc


def G(rpy):
    """
    Construct the gravity matrix from roll and pitch angles.

    Parameters:
        rpy (array-like): Array containing roll pitch yaw angles in radians.

    Returns:
        gMat (numpy.ndarray): 3x3 matrix representing the gravity vector.
    """
    roll, pitch, _ = rpy

    gMat = np.array([[np.cos(pitch),0, -np.cos(roll) * np.sin(pitch)], \
                    [0,             1,                  np.sin(roll)], \
                    [np.sin(pitch), 0,  np.cos(roll) * np.cos(pitch)]])
    return gMat

def R(rpy):
    """
    Construct a rotation matrix from roll, pitch, and yaw angles.

    Parameters:
        rpy (np.ndarray): Array containing roll, pitch, and yaw angles in radians.

    Returns:
        rMat (np.ndarray): 3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

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

def capPi(angles):
    """
    Function to cap angle about pi

    Parameters:
        angles (np.ndarray or float): Array containing angles.

    Returns:
        anglescap (np.ndarray or float): Capped angles
    """
    if isinstance(angles, float):
        if angles > np.pi:
            anglescap = angles - 2*np.pi
        elif angles < -np.pi:
            anglescap = angles + 2*np.pi
        else:
            anglescap = angles
    elif isinstance(angles, np.ndarray):
        anglescap = np.zeros(np.size(angles))
        for i, angle in enumerate(angles):
            if angle > np.pi:
                anglescap[i] = angle - 2*np.pi
            elif angle < -np.pi:
                anglescap[i] = angle + 2*np.pi
            else:
                anglescap[i] = angle
    else:
        anglescap = -999

    return anglescap

def cap2Pi(angles):
    """
    Function to cap angle about 2pi

    Parameters:
        angles (np.ndarray or float): Array containing angles.

    Returns:
        anglescap (np.ndarray or float): Capped angles
    """
    
    if isinstance(angles, float):
        if angles > 2*np.pi:
            anglescap = angles - 2*np.pi
        elif angles < -2*np.pi:
            anglescap = angles + 2*np.pi
        else:
            anglescap = angles
    elif isinstance(angles, np.ndarray):
        anglescap = np.zeros(np.size(angles))
        for i, angle in enumerate(angles):
            if angle > 2*np.pi:
                anglescap[i] = angle - 2*np.pi
            elif angle < -2*np.pi:
                anglescap[i] = angle + 2*np.pi
            else:
                anglescap[i] = angle
    else:
        anglescap = -999

    return anglescap

        