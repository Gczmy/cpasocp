import numpy as np


def make_residual_R(alpha, L, L_adj, prediction_horizon, A, B):
    # make residual R
    N = prediction_horizon
    n_x = A.shape[1]
    n_u = B.shape[1]
    n_z = N * (n_x + n_u) + n_x
    L = L * np.identity(L.shape[1])
    L_adj = L_adj * np.identity(L_adj.shape[1])

    op_A = np.hstack((np.eye(n_z), -alpha * L_adj))
    op_A = np.vstack((op_A, np.hstack((-alpha * L, np.eye(n_z)))))
    op_H = np.hstack((np.zeros((n_z, n_z)), L_adj))
    op_H = np.vstack((op_H, np.hstack((-L, np.zeros((n_z, n_z))))))
    op_T = np.linalg.inv(op_A + alpha * op_H) @ op_A
    res_R = np.eye(2 * n_z) - op_T
    return res_R


def SuperMann(z, eta, alpha, R):
    n_z = z.shape[0]
    x_0 = np.vstack((z, eta))
    # define parameters
    c0 = 0.5
    c1 = 0.5
    q = 0.5
    beta = 0.5
    sigma = 0
    lambda_ = 0.1

    N_max = 1000
    r_safe = np.linalg.norm(R @ x_0)
    eta_k = r_safe
    x_k = x_0
    for k in range(N_max):
        if np.linalg.norm(R @ x_0, np.inf) <= 1e-5:
            break
        d = 1  # Choose an update direction
        if np.linalg.norm(R @ x_k) <= c0 * eta_k:
            eta_k = np.linalg.norm(R @ x_k)
            x_k = x_k + d
            continue
        tau = 1
        for i in range(N_max):
            w_k = x_k + tau * d
            if np.linalg.norm(R @ x_k) <= r_safe and np.linalg.norm(R @ w_k) <= c1 * np.linalg.norm(R @ x_k):
                x_k = w_k
                r_safe = np.linalg.norm(R @ w_k) + q
                break
            rho = np.linalg.norm(R @ w_k) ** 2 \
                  - 2 * alpha * np.inner(np.reshape(R @ w_k, (1, -1)), np.reshape(w_k - x_k, (1, -1)))
            if rho >= sigma * np.linalg.norm(R @ w_k) * np.linalg.norm(R @ x_k):
                x_k = x_k - lambda_ / (np.linalg.norm(R @ w_k) ** 2) * rho * R @ w_k
                break
            else:
                tau = beta * tau
                continue
    z_k = x_k[0: n_z]
    eta_k = x_k[n_z: 2 * n_z]
    return z_k, eta_k
