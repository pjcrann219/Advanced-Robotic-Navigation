import numpy as np
from scipy.spatial.transform import Rotation
from earth import *

def skew(x):
    # Computes the skew-symmetric matrix of a 3D vector.
    # Inputs: x (np.ndarray): A 3x1 numpy array representing the 3D vector.
    # Outputs: (np.ndarray): A 3x3 skew-symmetric matrix.

    return np.array([[0, -x[2,0], x[1,0]],
                     [x[2,0], 0, -x[0,0]],
                     [-x[1,0], x[0,0], 0]])

def attitude_update_feedback(x_prior, w_i_b, dt):
    # Updates the attitude using the feedback architecture.
    # Inputs: 
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     w_i_b (np.ndarray): A 3x1 numpy array representing gyro measurement.
    #     dt (float): Time step.
    # Outputs: 
    #     tuple[np.ndarray, np.ndarray]: A tuple containing the prior rotation matrix and the updated rotation matrix.

    w_e = 7.2921157E-5 # Earths rate of rotation (rad/s)
    omega_i_e = np.zeros((3, 3)); omega_i_e[0, 1] = -w_e; omega_i_e[1, 0] = w_e

    Rn, Re, Rp = principal_radii(x_prior[0,0], x_prior[1,0])

    w_e_n = np.array([\
        x_prior[7,0] / (Re + x_prior[2,0]),\
        -x_prior[7,0] / (Rn + x_prior[2,0]),\
        -x_prior[7,0] * np.tan(x_prior[0,0]) / (Re + x_prior[2,0])])[:,np.newaxis]
    
    omega_e_n = skew(w_e_n)

    w_bar = w_i_b - x_prior[9:12,:] # Only for feedback filter
    omega_i_b = skew(w_bar)

    # convert current rotation euler to matrix R_prev
    R_prior = Rotation.from_euler('zyx', x_prior[3:6,0]).as_matrix()

    # Calc R_post
    R_post = R_prior @ (np.eye(3) + omega_i_b * dt) \
        - (omega_i_e + omega_e_n) @ R_prior * dt


    return R_prior, R_post

def attitude_update_feedforward(x_prior, w_i_b, dt):
    # Updates the attitude using the feedback architecture.
    # Inputs: 
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     w_i_b (np.ndarray): A 3x1 numpy array representing gyro measurement.
    #     dt (float): Time step.
    # Outputs: 
    #     tuple[np.ndarray, np.ndarray]: A tuple containing the prior rotation matrix and the updated rotation matrix.

    w_e = 7.2921157E-5 # Earths rate of rotation (rad/s)
    omega_i_e = np.zeros((3, 3)); omega_i_e[0, 1] -w_e; omega_i_e[1, 0] = w_e

    Rn, Re, Rp = principal_radii(x_prior[0,0], x_prior[1,0])

    w_e_n = np.array([\
        x_prior[7,0] / (Re + x_prior[2,0]),\
        -x_prior[7,0] / (Rn + x_prior[2,0]),\
        -x_prior[7,0] * np.tan(np.deg2rad(x_prior[0,0])) / (Re + x_prior[2,0])])[:,np.newaxis]
    
    omega_e_n = skew(w_e_n)
    omega_i_b = skew(w_i_b)

    # convert current rotation euler to matrix R_prev
    R_prior = Rotation.from_euler('zyx', x_prior[3:6,0], degrees=True).as_matrix()

    # Calc R_post
    R_post = R_prior @ (np.eye(3) + omega_i_b * dt) \
        - (omega_i_e + omega_e_n) @ R_prior * dt


    return R_prior, R_post

def velocity_update_feedback(x_prior, R_prior, R_post, f_bt, dt):
    # Updates the velocity using the feedback architecture.
    # Inputs:
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     R_prior (np.ndarray): A 3x3 numpy array representing the prior rotation matrix.
    #     R_post (np.ndarray): A 3x3 numpy array representing the updated rotation matrix.
    #     f_bt (np.ndarray): A 3x1 numpy array representing accelerometer input.
    #     dt (float): Time step.
    # Outputs:
    #     (np.ndarray): A 3x1 numpy array representing the updated velocity.

    # TODO: this section gets repeated a lot
    w_e = 7.2921157E-5 # Earths rate of rotation (rad/s)
    omega_i_e = np.zeros((3, 3)); omega_i_e[0, 1] = omega_i_e[1, 0] = w_e
    Rn, Re, Rp = principal_radii(x_prior[0,0], x_prior[1,0])

    w_e_n = np.array([\
        x_prior[7,0] / (Re + x_prior[2,0]),\
        -x_prior[7,0] / (Rn + x_prior[2,0]),\
        -x_prior[7,0] * np.tan(np.deg2rad(x_prior[0,0])) / (Re + x_prior[2,0])])[:,np.newaxis]
    
    omega_e_n = skew(w_e_n)

    f_bar = f_bt - x_prior[12:15,0][:,np.newaxis]

    f_nt = 0.5 * (R_prior + R_post)@f_bar

    g_LH = gravity_n(x_prior[0,0], x_prior[2,0])[:,np.newaxis]

    v_post = x_prior[6:9,0][:,np.newaxis] + \
        dt*(f_nt + g_LH - (omega_e_n + 2*omega_i_e)@x_prior[6:9,0][:,np.newaxis])

    return v_post

def velocity_update_feedforward(x_prior, R_prior, R_post, f_bt, dt):
    # Updates the velocity using the feedforward architecture.
    # Inputs:
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     R_prior (np.ndarray): A 3x3 numpy array representing the prior rotation matrix.
    #     R_post (np.ndarray): A 3x3 numpy array representing the updated rotation matrix.
    #     f_bt (np.ndarray): A 3x1 numpy array representing accelerometer input.
    #     dt (float): Time step.
    # Outputs:
    #     (np.ndarray): A 3x1 numpy array representing the updated velocity.

    f_nt = 0.5 * (R_prior + R_post)@f_bt

    g_LH = gravity_n(x_prior[0,0], x_prior[2,0])[:,np.newaxis]

    v_post = x_prior[6:9,0][:,np.newaxis] + dt*(f_nt - g_LH)# - (1-1)*x_prior[6:9,0])

    return v_post

def position_update(x_prior, v_post, dt):
    # Updates the position using the prior state and updated velocity.
    # Inputs:
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     v_post (np.ndarray): A 3x1 numpy array representing the updated velocity.
    #     dt (float): Time step.
    # Outputs:
    #     (np.ndarray): A 3x1 numpy array representing the updated latitude, longitude, and altitude.

    h_post = x_prior[2,0] - 0.5 * dt * (x_prior[8,0] + v_post[2,0])
    Rn, Re, Rp = principal_radii(x_prior[0,0], x_prior[1,0])
    Rn_post, Re_post, Rp_post = principal_radii(x_prior[0,0], x_prior[1,0])

    Lat_post = x_prior[0,0] + 0.5 * dt * (x_prior[6,0] / (Rn + x_prior[2,0]) + \
                    v_post[0,0] / (Rn + h_post))

    Lon_post = x_prior[1,0] + 0.5 * dt * (x_prior[7,0] / ((Re + x_prior[2,0])) + \
                    v_post[1,0]/(Re_post + h_post)*np.cos(np.deg2rad(Lat_post)))

    LLA_post = np.vstack([Lat_post, Lon_post, h_post])

    return LLA_post

def propogation_model_feedback(x_prior, w_i_b, f_bt, dt):
    # Implements the propagation model with feedback.
    # Inputs:
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     w_i_b (np.ndarray): A 3x1 numpy array representing gyro measurement.
    #     f_bt (np.ndarray): A 3x1 numpy array representing accelerometer input.
    #     dt (float): Time step.
    # Outputs:
    #     (np.ndarray): A 15x1 numpy array representing the updated state.

    R_prior, R_post = attitude_update_feedback(x_prior, w_i_b, dt)
    v_post = velocity_update_feedback(x_prior, R_prior, R_post, f_bt, dt)
    LLA_post = position_update(x_prior, v_post, dt)

    q_post = Rotation.from_matrix(R_post).as_euler('zyx', degrees=True)[:, np.newaxis]

    x_post = np.vstack([LLA_post, q_post, v_post, x_prior[9:15]])

    return x_post

def propogation_model_feedforward(x_prior, w_i_b, f_bt, dt):
    # Implements the propagation model with feedforward.
    # Inputs:
    #     x_prior (np.ndarray): A 15x1 numpy array representing the prior state.
    #     w_i_b (np.ndarray): A 3x1 numpy array representing gyro measurement.
    #     f_bt (np.ndarray): A 3x1 numpy array representing accelerometer input.
    #     dt (float): Time step.
    # Outputs:
    #     (np.ndarray): A 15x1 numpy array representing the updated state.

    R_prior, R_post = attitude_update_feedforward(x_prior, w_i_b, dt)
    v_post = velocity_update_feedforward(x_prior, R_prior, R_post, f_bt, dt)
    LLA_post = position_update(x_prior, v_post, dt)

    q_post = Rotation.from_matrix(R_post).as_euler('zyx', degrees=True)[:, np.newaxis]

    x_post = np.vstack([LLA_post, q_post, v_post, x_prior[9:12]])

    return x_post


# x_prior = np.ones([12,1]) # Prior state
# w_i_b = np.ones([3,1])*2 # Gyro measurement
# f_bt = np.ones([3,1])*2# Accel input
# dt = 1

# x_post = propogation_model_feedforward(x_prior, w_i_b, f_bt, dt)

# print(x_post)