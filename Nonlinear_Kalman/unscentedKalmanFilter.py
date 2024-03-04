import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from observationModel import *

def UKF(dataPath, R, Q):
    dataFull = scipy.io.loadmat(dataPath, simplify_cells=True)
    data = dataFull['data']
    n = 15

    x = np.zeros([15,1])
    P = np.eye(15)

    zold = x[0:6]
    # print(f"zold: {zold}, np.shape(zold): {np.shape(zold)}")

    last_t = 0
    for i in range(len(data)):
        print(f"i: {i}/{len(data)}")
        dt = data[i]['t'] - last_t
        success, rvecIMU, tvecIMU = estimate_pose(data[i])
        if success:
            z = np.concatenate((tvecIMU, rvecIMU.T)).reshape(-1,1)
            uw = data[i]['omg']
            ua = data[i]['acc']
            u = np.concatenate((uw, ua)).reshape(-1,1)
            x, P = stepUFK(x, P, u, dt, n, z, R, Q)
        else:
            print("FAILLLLLLLLLLL")
            z = x[0:6]
            u = np.concatenate((uw, ua)).reshape(-1,1)
            x, P = stepUFK(x, P, u, dt, n, z, 100*R, Q)

        last_t = data[i]['t']
    pass

def stepUFK(x, P, u, dt, n, measurement, R = np.eye(6), Q = np.eye(15)):
    # Step through a single time step of UKF
    print("1)")
    if not is_positive_semidefinite(P):
        print(P)
        print(f"x: {x}, u: {u}, dt: {dt}, n = {n}, measurement: {measurement}")
    # Get sigma points X0 with wights wm, wc
    X0, wm, wc = getSigmaPoints(x, P, n)

    # Propogate sigma points through state transition
    X1 = np.zeros(np.shape(X0))
    for i in range(2*n+1):
        thisX = X0[:,i]
        X1[:,i] = state_transition(thisX, u, dt)

    # Recover mean
    x = np.sum(X1 * wm, axis=1)
    # Recover variance
    diff = X1-np.vstack(x)
    # diff = X1 - np.vstack(x)[:, np.newaxis]
    # print(f"np.shape(diff): {np.shape(diff)}")
    P = np.zeros((15, 15))
    # d = diff[:, 0].reshape(-1,1)
    # print(f"d: {d}")
    # P = wc[0] * d @ d.T
    # print(wc)
    # print("1.5)")
    # is_positive_semidefinite(P)
    # for i in range(2 * n + 1):
    #     d = diff[:, i].reshape(-1,1)
    #     P += wc[i] * d @ d.T
    #     # print(wc[i] * d @ d.T)

    diff2 = X1-np.vstack(X1[:,0])
    for i in range(2 * n + 1):
        d = diff2[:, i].reshape(-1,1)
        P += wc[i] * d @ d.T
    P += Q

    print("2)")
    is_positive_semidefinite(P)
    # Get new sigma points
    X2, wm, wc = getSigmaPoints(x, P, n)

    # Put sigma points through measurement model
    Z = np.zeros([6,2*n+1])
    for i in range(2*n+1):
        thisX = X2[:,i]
        Z[:,i] = measurement_model(thisX)
    
    # Recover mean
    z = np.sum(Z * wm, axis=1)

    # Recover variance
    # diff = Z-np.vstack(z)
    S = np.zeros((6,6))
    # for i in range(2 * n + 1):
    #     S += wc[i] * np.outer(diff[:, i], diff[:, i])
    # S += R

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
    print("3)")
    is_positive_semidefinite(P)
    # P = P - K @ S @ np.linalg.pinv(K)
    P = P - K @ S @ np.transpose(K)
    print("4)")
    is_positive_semidefinite(P)

    return x, P

def is_positive_semidefinite(matrix):
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

    x1 = np.zeros(np.shape(x0))
    g = np.array([[0], [0], [-9.81]])
    g[2] = 0

    # P1 = P0 + P0d * dt
    x1[0:3] = x0[0:3] + x0[6:9] * dt
    # q1 = q0 + Inv(G(q)) * Uw * dt
    x1[3:6] = x0[3:6] + np.squeeze(np.linalg.inv(G(x0[6:9])) @ u[0:3] * dt)
    # p1d = p0d + (g + R(q)*ua) * dt
    x1[6:9] = x0[6:9] + np.squeeze((g + R(x0[3:6]) @ u[3:6]) * dt)
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
    lam = (alpha**2) * (n + k) - n

    # Set point i=0
    X[:, 0] = x
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    # Set i=1:2n
    # print(f"np.shape(P): {np.shape(P)}")
    # print(f"P: {P}")
    # print(f"(n + lam) * P: P{(n + lam) * P}")
    offset = np.linalg.cholesky((n + lam) * P)
    for i in range(1, n+1):
        X[:,i] = x + offset[:, i-1]
        X[:,n+i] = x - offset[:, i-1]
        wm[i] = wm[n+i] = wc[i] = wc[n+i] = 0.5 / (n + lam)
    
    return X, wm, wc


def G(rpy):
    roll = rpy[0]
    pitch = rpy[1]
    # print(f'pitch: {pitch}, roll: {roll}, yaw: {yaw}')

    gMat = np.array([[np.cos(pitch),0, -np.cos(roll) * np.sin(pitch)], \
                    [0,             1, np.sin(roll)                 ], \
                    [np.sin(pitch), 0, np.cos(roll) * np.cos(pitch) ]])
    return gMat

def R(rpy):
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