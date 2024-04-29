import numpy as np
from scipy.spatial.transform import Rotation
from earth import *

euler_type = 'xyz'

def skew(x):
    # Computes the skew-symmetric matrix of a 3D vector.
    # Inputs: x (np.ndarray): A 3x1 numpy array representing the 3D vector.
    # Outputs: (np.ndarray): A 3x3 skew-symmetric matrix.

    return np.array([[0, -x[2,0], x[1,0]],
                     [x[2,0], 0, -x[0,0]],
                     [-x[1,0], x[0,0], 0]])

def feedback_propogation_model(x_prior, w_i_b, f_bt, dt):
    # Implements the propagation model with feedback.
    # Inputs:
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     w_i_b (np.ndarray): A 3x1 numpy array representing gyro measurement.
    #     f_bt (np.ndarray): A 3x1 numpy array representing accelerometer input.
    #     dt (float): Time step.
    # Outputs:
    #     (np.ndarray): A 15x1 numpy array representing the updated state.

## Attitude Update:
    w_e = 7.2921157E-5 # Earths rate of rotation (rad/s)
    omega_i_e = np.zeros((3, 3)); omega_i_e[0, 1] = -w_e; omega_i_e[1, 0] = w_e

    Rn, Re, Rp = principal_radii(x_prior[0,0], x_prior[2,0])

    w_e_n = np.array([\
        x_prior[7,0] / (Re + x_prior[2,0]),\
        -x_prior[6,0] / (Rn + x_prior[2,0]),\
        -x_prior[7,0] * np.tan(np.deg2rad(x_prior[0,0])) / (Re + x_prior[2,0])])[:,np.newaxis]
    omega_e_n = skew(w_e_n)

    w_bar = w_i_b - x_prior[9:12,:] # Only for feedback filter
    omega_i_b = skew(w_bar)

    # convert current rotation euler to matrix R_prev
    R_prior = Rotation.from_euler(euler_type, x_prior[3:6,0]).as_matrix()

    # Calc R_post
    R_post = R_prior @ (np.eye(3) + omega_i_b * dt) \
        - (omega_i_e + omega_e_n) @ R_prior * dt

## Velocity Update
    f_bar = f_bt - x_prior[12:15,0][:,np.newaxis]

    f_nt = 0.5 * (R_prior + R_post)@f_bar

    g_LH = gravity_n(x_prior[0,0], x_prior[2,0])[:,np.newaxis]

    v_post = x_prior[6:9,0][:,np.newaxis] + \
        dt*(f_nt + g_LH - (omega_e_n + 2*omega_i_e)@x_prior[6:9,0][:,np.newaxis])

## Posistion Update
    h_post = x_prior[2,0] - 0.5 * dt * (x_prior[8,0] + v_post[2,0])
    Rn, Re, Rp = principal_radii(x_prior[0,0], h_post)
    
    Lat_post = x_prior[0,0] + 0.5 * dt * (x_prior[6,0] / (Rn + x_prior[2,0]) + \
                    v_post[0,0] / (Rn + h_post))

    Rn_post, Re_post, Rp_post = principal_radii(Lat_post, h_post)
    Lon_post = x_prior[1,0] + 0.5 * dt * (x_prior[7,0] / ((Re + x_prior[2,0]))*np.cos(np.deg2rad(x_prior[0,0])) + \
                    v_post[1,0]/(Re_post + h_post)*np.cos(np.deg2rad(Lat_post)))

    LLA_post = np.vstack([Lat_post, Lon_post, h_post])

## Return update state matrix
    q_post = Rotation.from_matrix(R_post).as_euler(euler_type)[:, np.newaxis]
    x_post = np.vstack([LLA_post, q_post, v_post, x_prior[9:15]])

    return x_post
