import numpy as np

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

def getSigmaPoints(x, P, n):
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
    
    for i in range(1, n+1):
        X[:,i] = x + offset[:, i-1]
        X[:,n+i] = x - offset[:, i-1]
        wm[i] = wm[n+i] = wc[i] = wc[n+i] = 0.5 / (n + lam)
    
    return X, wm, wc