import numpy as np
import scipy as sp


def precondition(L, scaling_factor):
    tau = [None] * L.shape[1]
    sigma = [None] * L.shape[0]
    q = np.count_nonzero(L, axis=1)
    for j in range(L.shape[1]):
        tau[j] = 1 / (scaling_factor * np.linalg.norm(L[:, j]) ** 2)
    for i in range(L.shape[0]):
        sigma[i] = scaling_factor / q[i]
    T = np.diagflat(tau)
    Sigma = np.diagflat(sigma)
    return T, Sigma


def proximal_of_h_online_part(prediction_horizon, proximal_lambda, initial_state, initial_guess_vector, state_dynamics,
                              control_dynamics, control_weight, P_seq, R_tilde_seq, K_seq,
                              A_bar_seq):
    """
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param proximal_lambda: a parameter lambda for proximal operator
    :param initial_state: initial state of dynamic system
    :param initial_guess_vector: initial guess vector w for proximal process
    :param state_dynamics: matrix (A), describing the state dynamics
    :param control_dynamics: matrix (B), describing control dynamics
    :param control_weight: scalar or matrix (R), input cost matrix or scalar
    :param P_seq: tensor, matrix sequence of (P) from proximal of h offline part
    :param R_tilde_seq: tensor, matrix sequence of (R_tilde) from proximal of h offline part
    :param K_seq: tensor, matrix sequence of (K) from proximal of h offline part
    :param A_bar_seq: tensor, matrix sequence of (A_bar) from proximal of h offline part
    """
    N = prediction_horizon
    x_0 = initial_state
    w = initial_guess_vector
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    n_x = A.shape[1]
    n_u = B.shape[1]

    if w.shape[0] != N * (n_x + n_u) + n_x:
        raise ValueError("Initial guess vector w row is not correct")
    if x_0.shape[0] != n_x:
        raise ValueError("Initial state x0 row is not correct")

    chi_N = w[N * (n_x + n_u): N * (n_x + n_u) + n_x]
    q_0 = - 1 / proximal_lambda * chi_N

    q_seq = np.zeros((n_x, 1, N + 1))  # tensor
    d_seq = np.zeros((n_u, 1, N))  # tensor
    q_seq[:, :, N] = q_0

    for t in range(N):
        v = w[(N - t - 1) * (n_x + n_u) + n_x: (N - t) * (n_x + n_u)]
        chi = w[(N - t - 1) * (n_x + n_u): (N - t - 1) * (n_x + n_u) + n_x]
        c, low = sp.linalg.cho_factor(R_tilde_seq[:, :, N - t - 1])
        d_seq[:, :, N - t - 1] = sp.linalg.cho_solve((c, low), 1 / proximal_lambda * v - B.T @ q_seq[:, :, N - t])
        q_seq[:, :, N - t - 1] = K_seq[:, :, N - t - 1].T \
                                 @ ((R + 1 / proximal_lambda * np.eye(n_u)) @ d_seq[:, :, N - t - 1]
                                    - 1 / proximal_lambda * v) - 1 / proximal_lambda * chi \
                                 + A_bar_seq[:, :, N - t - 1].T @ (P_seq[:, :, N - t] @ B @ d_seq[:, :, N - t - 1]
                                                                   + q_seq[:, :, N - t])
    x_seq = np.zeros((n_x, 1, N + 1))  # tensor
    u_seq = np.zeros((n_u, 1, N))  # tensor
    x_seq[:, :, N] = np.reshape(x_0, (n_x, 1))

    # Construct Proximal of h at w
    prox = np.reshape(x_seq[:, :, N], (n_x, 1))  # x_0
    for t in range(N):
        u_seq[:, :, N - t - 1] = K_seq[:, :, t] @ x_seq[:, :, N - t] + d_seq[:, :, t]
        x_seq[:, :, N - t - 1] = A @ x_seq[:, :, N - t] + B @ u_seq[:, :, N - t - 1]
        prox = np.vstack((prox, np.reshape(u_seq[:, :, N - t - 1], (n_u, 1))))
        prox = np.vstack((prox, np.reshape(x_seq[:, :, N - t - 1], (n_x, 1))))

    return prox


def proximal_of_h_online_part_precondition(prediction_horizon, T_pre, Sigma_pre, initial_state, initial_guess_vector,
                                           state_dynamics, control_dynamics, control_weight, P_seq, R_tilde_seq, K_seq,
                                           A_bar_seq):
    """
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param proximal_lambda: a parameter lambda for proximal operator
    :param initial_state: initial state of dynamic system
    :param initial_guess_vector: initial guess vector w for proximal process
    :param state_dynamics: matrix (A), describing the state dynamics
    :param control_dynamics: matrix (B), describing control dynamics
    :param control_weight: scalar or matrix (R), input cost matrix or scalar
    :param P_seq: tensor, matrix sequence of (P) from proximal of h offline part
    :param R_tilde_seq: tensor, matrix sequence of (R_tilde) from proximal of h offline part
    :param K_seq: tensor, matrix sequence of (K) from proximal of h offline part
    :param A_bar_seq: tensor, matrix sequence of (A_bar) from proximal of h offline part
    """
    N = prediction_horizon
    x_0 = initial_state
    w = initial_guess_vector
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    n_x = A.shape[1]
    n_u = B.shape[1]

    if w.shape[0] != N * (n_x + n_u) + n_x:
        raise ValueError("Initial guess vector w row is not correct")
    if x_0.shape[0] != n_x:
        raise ValueError("Initial state x0 row is not correct")

    chi_N = w[N * (n_x + n_u): N * (n_x + n_u) + n_x]
    tau_N = T_pre[:, N * (n_x + n_u): N * (n_x + n_u) + n_x]
    q_0 = - 1 / tau_N @ chi_N

    q_seq = np.zeros((n_x, 1, N + 1))  # tensor
    d_seq = np.zeros((n_u, 1, N))  # tensor
    q_seq[:, :, N] = q_0

    for t in range(N):
        v = w[(N - t - 1) * (n_x + n_u) + n_x: (N - t) * (n_x + n_u)]
        chi = w[(N - t - 1) * (n_x + n_u): (N - t - 1) * (n_x + n_u) + n_x]
        sigma = Sigma_pre[:, (N - t - 1) * (n_x + n_u) + n_x: (N - t) * (n_x + n_u)]
        tau = Sigma_pre[:, (N - t - 1) * (n_x + n_u): (N - t - 1) * (n_x + n_u) + n_x]

        c, low = sp.linalg.cho_factor(R_tilde_seq[:, :, N - t - 1])
        d_seq[:, :, N - t - 1] = sp.linalg.cho_solve((c, low), 1 / sigma @ v - B.T @ q_seq[:, :, N - t])
        q_seq[:, :, N - t - 1] = K_seq[:, :, N - t - 1].T \
                                 @ ((R + 1 / sigma @ np.eye(n_u)) @ d_seq[:, :, N - t - 1]
                                    - 1 / sigma @ v) - 1 / tau @ chi \
                                 + A_bar_seq[:, :, N - t - 1].T @ (P_seq[:, :, N - t] @ B @ d_seq[:, :, N - t - 1]
                                                                   + q_seq[:, :, N - t])
    x_seq = np.zeros((n_x, 1, N + 1))  # tensor
    u_seq = np.zeros((n_u, 1, N))  # tensor
    x_seq[:, :, N] = np.reshape(x_0, (n_x, 1))

    # Construct Proximal of h at w
    prox = np.reshape(x_seq[:, :, N], (n_x, 1))  # x_0
    for t in range(N):
        u_seq[:, :, N - t - 1] = K_seq[:, :, t] @ x_seq[:, :, N - t] + d_seq[:, :, t]
        x_seq[:, :, N - t - 1] = A @ x_seq[:, :, N - t] + B @ u_seq[:, :, N - t - 1]
        prox = np.vstack((prox, np.reshape(u_seq[:, :, N - t - 1], (n_u, 1))))
        prox = np.vstack((prox, np.reshape(x_seq[:, :, N - t - 1], (n_x, 1))))

    return prox
