import numpy as np
import scipy as sp
import cpasocp.core.linear_operators as core_lin_op
import cpasocp.core.proximal_online_part as core_online
from math import sqrt as sqrt
import cpasocp.core.SuperMann as SuperMann


def make_alpha(prediction_horizon, state_dynamics, control_dynamics, stage_state, control_state, terminal_state,
               initial_guess_z):
    """
    :param initial_guess_z: vector initial guess of (z0) of Chambolle-Pock algorithm
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param state_dynamics: matrix (A), describing the state dynamics
    :param control_dynamics: matrix (B), describing control dynamics
    :param stage_state: matrix (Gamma_x), describing the state constraints
    :param control_state: matrix (Gamma_u), describing the control constraints
    :param terminal_state: matrix (Gamma_N), describing terminal constraints
    """
    n_z = initial_guess_z.shape[0]
    L = core_lin_op.LinearOperator(prediction_horizon, state_dynamics, control_dynamics, stage_state, control_state,
                                   terminal_state).make_L_op()
    L_z = L @ np.ones((n_z, 1))
    L_adj = core_lin_op.LinearOperator(prediction_horizon, state_dynamics, control_dynamics, stage_state,
                                       control_state, terminal_state).make_L_adj()
    # Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1
    eigs = np.real(sp.sparse.linalg.eigs(L_adj @ L, k=n_z - 2, return_eigenvectors=False))
    L_norm = np.sqrt(max(eigs))
    alpha = 0.99 / L_norm
    return L, L_z, L_adj, alpha


def proj_to_c(vector, prediction_horizon, stage_state, terminal_state, stage_sets, terminal_set):
    """
    :param vector: the vector to be projected to sets (C_t) and C_N
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param stage_state: matrix (Gamma_x), describing the state constraints
    :param terminal_state: matrix (Gamma_N), describing terminal constraints
    :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
    state-control constraints
    :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
    """
    N = prediction_horizon
    n_c = stage_state.shape[0]
    n_f = terminal_state.shape[0]
    if type(stage_sets).__name__ == 'Cartesian':
        vector_list = [None] * N
        for i in range(N):
            vector_list[i] = vector[i * n_c: (i + 1) * n_c]
        vector_stage = stage_sets.project(vector_list)
    else:
        vector_stage = stage_sets.project(vector[0: N * n_c])
    vector_terminal = terminal_set.project(vector[N * n_c: N * n_c + n_f])
    vector = np.vstack((vector_stage, vector_terminal))
    return vector


def CP_for_ocp(epsilon, initial_guess_z, initial_guess_eta, alpha, L, L_z, L_adj,
               prediction_horizon, initial_state, state_dynamics, control_dynamics,
               control_weight, P_seq, R_tilde_seq, K_seq, A_bar_seq, stage_state,
               terminal_state, stage_sets, terminal_set):
    """
    :param epsilon: scalar (epsilon) of Chambolle-Pock algorithm
    :param initial_guess_z: vector initial guess of (z0) of Chambolle-Pock algorithm
    :param initial_guess_eta: vector initial guess of (eta0) of Chambolle-Pock algorithm
    :param alpha: Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1, here α1 = α2 = 0.99/∥L∥
    :param L: LinearOperator of (L) of Chambolle-Pock algorithm
    :param L_z: vector of (L_z) of Chambolle-Pock algorithm
    :param L_adj: LinearOperator adjoint of (L_adj) of Chambolle-Pock algorithm
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param initial_state: initial state of dynamic system
    :param state_dynamics: matrix (A), describing the state dynamics
    :param control_dynamics: matrix (B), describing control dynamics
    :param control_weight: scalar or matrix (R), input cost matrix or scalar
    :param P_seq: tensor, matrix sequence of (P) from proximal of h offline part
    :param R_tilde_seq: tensor, matrix sequence of (R) from proximal of h offline part
    :param K_seq: tensor, matrix sequence of (K) from proximal of h offline part
    :param A_bar_seq: tensor, matrix sequence of (A_bar) from proximal of h offline part
    :param stage_state: matrix (Gamma_x), describing the state constraints
    :param terminal_state: matrix (Gamma_N), describing terminal constraints
    :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
    state-control constraints
    :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
    """
    N = prediction_horizon
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    Gamma_x = stage_state
    Gamma_N = terminal_state
    C_t = stage_sets
    C_N = terminal_set
    n_x = A.shape[1]
    n_u = B.shape[1]
    n_z = N * (n_x + n_u) + n_x
    n_L = L_z.shape[0]
    x0 = initial_state.copy()
    z0 = initial_guess_z.copy()
    eta0 = initial_guess_eta.copy()
    if z0.shape[0] != n_z:
        raise ValueError("Initial guess vector z row is not correct")
    if eta0.shape[0] != n_L:
        raise ValueError("Initial guess vector eta row is not correct")

    z_next = z0
    eta_next = eta0
    n_max = 10000
    residuals_cache = np.zeros((n_max, 3))

    for i in range(n_max):
        z_prev = z_next
        eta_prev = eta_next
        z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
                                                       proximal_lambda=alpha,
                                                       initial_state=x0,
                                                       initial_guess_vector=z_prev - alpha * L_adj @ eta_prev,
                                                       state_dynamics=A,
                                                       control_dynamics=B,
                                                       control_weight=R,
                                                       P_seq=P_seq,
                                                       R_tilde_seq=R_tilde_seq,
                                                       K_seq=K_seq,
                                                       A_bar_seq=A_bar_seq)
        eta_half_next = eta_prev + alpha * L @ (2 * z_next - z_prev)
        eta_next = eta_half_next - alpha * proj_to_c(eta_half_next / alpha, N, Gamma_x, Gamma_N, C_t, C_N)

        # Termination criteria
        xi_1 = (z_prev - z_next) / alpha - L_adj @ (eta_prev - eta_next)
        xi_2 = (eta_prev - eta_next) / alpha + L @ (z_next - z_prev)
        xi_gap = xi_1 + L_adj @ xi_2
        t_1 = np.linalg.norm(xi_1, np.inf)
        t_2 = np.linalg.norm(xi_2, np.inf)
        t_3 = np.linalg.norm(xi_gap, np.inf)
        residuals_cache[i, 0] = t_1
        residuals_cache[i, 1] = t_2
        residuals_cache[i, 2] = t_3

        if t_1 <= epsilon and t_2 <= epsilon and t_3 <= epsilon:
            break
    status = 0  # converge success
    if i >= 9000:
        status = 1  # converge failed
    residuals_cache = residuals_cache[0:i, :]
    return residuals_cache, z_next, status


def CP_scaling_for_ocp(scaling_factor, epsilon, initial_guess_z,
                       initial_guess_eta, alpha, L, L_z, L_adj,
                       prediction_horizon, initial_state, state_dynamics, control_dynamics,
                       control_weight, P_seq, R_tilde_seq, K_seq, A_bar_seq, stage_state,
                       terminal_state, stage_sets, terminal_set):
    """
    :param epsilon: scalar (epsilon) of Chambolle-Pock algorithm
    :param initial_guess_z: vector initial guess of (z0) of Chambolle-Pock algorithm
    :param initial_guess_eta: vector initial guess of (eta0) of Chambolle-Pock algorithm
    :param alpha: Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1, here α1 = α2 = 0.99/∥L∥
    :param L: LinearOperator of (L) of Chambolle-Pock algorithm
    :param L_z: vector of (L_z) of Chambolle-Pock algorithm
    :param L_adj: LinearOperator adjoint of (L_adj) of Chambolle-Pock algorithm
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param initial_state: initial state of dynamic system
    :param state_dynamics: matrix (A), describing the state dynamics
    :param control_dynamics: matrix (B), describing control dynamics
    :param control_weight: scalar or matrix (R), input cost matrix or scalar
    :param P_seq: tensor, matrix sequence of (P) from proximal of h offline part
    :param R_tilde_seq: tensor, matrix sequence of (R) from proximal of h offline part
    :param K_seq: tensor, matrix sequence of (K) from proximal of h offline part
    :param A_bar_seq: tensor, matrix sequence of (A_bar) from proximal of h offline part
    :param stage_state: matrix (Gamma_x), describing the state constraints
    :param terminal_state: matrix (Gamma_N), describing terminal constraints
    :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
    state-control constraints
    :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
    """
    N = prediction_horizon
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    Gamma_x = stage_state
    Gamma_N = terminal_state
    C_t = stage_sets
    C_N = terminal_set
    n_x = A.shape[1]
    n_u = B.shape[1]
    n_z = N * (n_x + n_u) + n_x
    n_L = L_z.shape[0]

    # scaling
    x0 = initial_state.copy()
    for i in range(n_x):
        x0[i] = x0[i] / scaling_factor[i]
    z0 = initial_guess_z.copy() / scaling_factor
    eta0 = initial_guess_eta.copy() / scaling_factor

    if z0.shape[0] != n_z:
        raise ValueError("Initial guess vector z row is not correct")
    if eta0.shape[0] != n_L:
        raise ValueError("Initial guess vector eta row is not correct")

    z_next = z0
    eta_next = eta0
    n_max = 10000
    residuals_cache = np.zeros((n_max, 3))

    for i in range(n_max):
        z_prev = z_next
        eta_prev = eta_next
        z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
                                                       proximal_lambda=alpha,
                                                       initial_state=x0,
                                                       initial_guess_vector=z_prev - alpha * L_adj @ eta_prev,
                                                       state_dynamics=A,
                                                       control_dynamics=B,
                                                       control_weight=R,
                                                       P_seq=P_seq,
                                                       R_tilde_seq=R_tilde_seq,
                                                       K_seq=K_seq,
                                                       A_bar_seq=A_bar_seq)
        eta_half_next = eta_prev + alpha * L @ (2 * z_next - z_prev)
        eta_next = eta_half_next - alpha * proj_to_c(eta_half_next / alpha, N, Gamma_x, Gamma_N, C_t, C_N)

        # Termination criteria
        # scaling back
        z_prev_scaling_back = z_prev * scaling_factor
        eta_prev_scaling_back = eta_prev * scaling_factor
        z_next_scaling_back = z_next * scaling_factor
        eta_next_scaling_back = eta_next * scaling_factor

        xi_1 = (z_prev_scaling_back - z_next_scaling_back) / alpha \
               - L_adj @ (eta_prev_scaling_back - eta_next_scaling_back)
        xi_2 = (eta_prev_scaling_back - eta_next_scaling_back) / alpha \
               + L @ (z_next_scaling_back - z_prev_scaling_back)
        xi_gap = xi_1 + L_adj @ xi_2
        t_1 = np.linalg.norm(xi_1, np.inf)
        t_2 = np.linalg.norm(xi_2, np.inf)
        t_3 = np.linalg.norm(xi_gap, np.inf)
        residuals_cache[i, 0] = t_1
        residuals_cache[i, 1] = t_2
        residuals_cache[i, 2] = t_3
        if t_1 <= epsilon and t_2 <= epsilon and t_3 <= epsilon:
            break
    status = 0  # converge success
    if i >= 9000:
        status = 1  # converge failed
    residuals_cache = residuals_cache[0:i, :]
    return residuals_cache, z_next_scaling_back, status


def CP_SuperMann(epsilon, initial_guess_z, initial_guess_eta, alpha, L, L_z, L_adj,
                 prediction_horizon, initial_state, state_dynamics, control_dynamics, stage_state_weight,
                 control_weight, terminal_state_weight, P_seq, R_tilde_seq, K_seq, A_bar_seq, stage_state,
                 terminal_state, stage_sets, terminal_set):
    """
    :param epsilon: scalar (epsilon) of Chambolle-Pock algorithm
    :param initial_guess_z: vector initial guess of (z0) of Chambolle-Pock algorithm
    :param initial_guess_eta: vector initial guess of (eta0) of Chambolle-Pock algorithm
    :param alpha: Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1, here α1 = α2 = 0.99/∥L∥
    :param L: LinearOperator of (L) of Chambolle-Pock algorithm
    :param L_z: vector of (L_z) of Chambolle-Pock algorithm
    :param L_adj: LinearOperator adjoint of (L_adj) of Chambolle-Pock algorithm
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param initial_state: initial state of dynamic system
    :param state_dynamics: matrix (A), describing the state dynamics
    :param control_dynamics: matrix (B), describing control dynamics
    :param stage_state_weight: matrix (Q), stage state cost matrix
    :param control_weight: scalar or matrix (R), input cost matrix or scalar
    :param terminal_state_weight: matrix (P), terminal state cost matrix
    :param P_seq: tensor, matrix sequence of (P) from proximal of h offline part
    :param R_tilde_seq: tensor, matrix sequence of (R) from proximal of h offline part
    :param K_seq: tensor, matrix sequence of (K) from proximal of h offline part
    :param A_bar_seq: tensor, matrix sequence of (A_bar) from proximal of h offline part
    :param stage_state: matrix (Gamma_x), describing the state constraints
    :param terminal_state: matrix (Gamma_N), describing terminal constraints
    :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
    state-control constraints
    :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
    """
    N = prediction_horizon
    A = state_dynamics
    B = control_dynamics
    Q = stage_state_weight
    R = control_weight
    P = terminal_state_weight
    Gamma_x = stage_state
    Gamma_N = terminal_state
    C_t = stage_sets
    C_N = terminal_set
    n_x = A.shape[1]
    n_u = B.shape[1]
    n_z = N * (n_x + n_u) + n_x
    n_L = L_z.shape[0]
    x0 = initial_state.copy()
    z0 = initial_guess_z.copy()
    eta0 = initial_guess_eta.copy()
    if z0.shape[0] != n_z:
        raise ValueError("Initial guess vector z row is not correct")
    if eta0.shape[0] != n_L:
        raise ValueError("Initial guess vector eta row is not correct")

    z_next = z0
    eta_next = eta0
    n_max = 10000
    residuals_cache = np.zeros((n_max, 3))

    # SuperMann parameter
    c0 = 0.99
    c1 = 0.99
    q = 0.99
    beta = 0.5
    sigma = 0.1
    lambda_ = 1.95
    m = 3
    s_cache = [None] * m
    y_cache = [None] * m

    op_A = np.hstack((np.eye(n_z), -alpha * L_adj @ np.identity(n_z)))
    op_A = np.vstack((op_A, np.hstack((-alpha * L @ np.identity(n_z), np.eye(n_z)))))

    for i in range(n_max):
        z_prev = z_next
        eta_prev = eta_next
        z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
                                                       proximal_lambda=alpha,
                                                       initial_state=x0,
                                                       initial_guess_vector=z_prev - alpha * L_adj @ eta_prev,
                                                       state_dynamics=A,
                                                       control_dynamics=B,
                                                       control_weight=R,
                                                       P_seq=P_seq,
                                                       R_tilde_seq=R_tilde_seq,
                                                       K_seq=K_seq,
                                                       A_bar_seq=A_bar_seq)
        eta_half_next = eta_prev + alpha * L @ (2 * z_next - z_prev)
        eta_next = eta_half_next - alpha * proj_to_c(eta_half_next / alpha, N, Gamma_x, Gamma_N, C_t, C_N)

        # use SuperMann
        # x_k = np.vstack((z_prev, eta_prev))
        # T_x_k = np.vstack((z_next, eta_next))
        # if i == 0:
        #     r_safe = sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k))
        #     supermann_eta = r_safe
        # if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= epsilon:
        #     break
        #
        # # Choose an update direction
        # m_i = min(m, i)
        # oldest_flag = i % m
        # T_x_k_z_prev = T_x_k[0: n_z]
        # T_x_k_eta_prev = T_x_k[n_z: 2 * n_z]
        # T_x_k_z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
        #                                                      proximal_lambda=alpha,
        #                                                      initial_state=x0,
        #                                                      initial_guess_vector=T_x_k_z_prev
        #                                                                           - alpha * L_adj @ T_x_k_eta_prev,
        #                                                      state_dynamics=A,
        #                                                      control_dynamics=B,
        #                                                      control_weight=R,
        #                                                      P_seq=P_seq,
        #                                                      R_tilde_seq=R_tilde_seq,
        #                                                      K_seq=K_seq,
        #                                                      A_bar_seq=A_bar_seq)
        # T_x_k_eta_half_next = T_x_k_eta_prev + alpha * L @ (2 * T_x_k_z_next - T_x_k_z_prev)
        # T_x_k_eta_next = T_x_k_eta_half_next - alpha * proj_to_c(T_x_k_eta_half_next / alpha,
        #                                                          N, Gamma_x, Gamma_N, C_t, C_N)
        # TT_x_k = np.vstack((T_x_k_z_next, T_x_k_eta_next))
        # s_cache[oldest_flag] = x_k - T_x_k
        # y_cache[oldest_flag] = (x_k - T_x_k) - (T_x_k - TT_x_k)
        # r_k = x_k - T_x_k
        # S_k = s_cache[0]
        # Y_k = y_cache[0]
        # for k in range(1, m_i+1):
        #     if k < m:
        #         S_k = np.hstack((S_k, s_cache[k]))
        #         Y_k = np.hstack((Y_k, y_cache[k]))
        # t_k = np.linalg.lstsq(Y_k, r_k, rcond=None)[0]
        # d_k = -r_k - (S_k - Y_k) @ t_k
        #
        # # K0
        # if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= c0 * supermann_eta:
        #     supermann_eta = sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k))
        #     w_k = x_k + d_k
        #     x_k = w_k
        #     # update z, eta to CP
        #     z_next = x_k[0:n_z]
        #     eta_next = x_k[n_z:2 * n_z]
        #     continue
        # tau_k = 1
        # for k in range(n_max):
        #     w_k = x_k + tau_k * d_k
        #     w_z_prev = w_k[0:n_z]
        #     w_eta_prev = w_k[n_z:2 * n_z]
        #     w_z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
        #                                                      proximal_lambda=alpha,
        #                                                      initial_state=x0,
        #                                                      initial_guess_vector=w_z_prev - alpha * L_adj @ w_eta_prev,
        #                                                      state_dynamics=A,
        #                                                      control_dynamics=B,
        #                                                      control_weight=R,
        #                                                      P_seq=P_seq,
        #                                                      R_tilde_seq=R_tilde_seq,
        #                                                      K_seq=K_seq,
        #                                                      A_bar_seq=A_bar_seq)
        #     w_eta_half_next = w_eta_prev + alpha * L @ (2 * w_z_next - w_z_prev)
        #     w_eta_next = w_eta_half_next - alpha * proj_to_c(w_eta_half_next / alpha, N, Gamma_x, Gamma_N, C_t, C_N)
        #     T_w_k = np.vstack((w_z_next, w_eta_next))
        #     # K1
        #     if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= r_safe \
        #             and sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) <= c1 * sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)):
        #         x_k = w_k
        #         r_safe = sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) + q ** i
        #         break
        #     # K2
        #     rho_k = sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) ** 2 \
        #             - 2 * 0.5 * (w_k - T_w_k).T @ op_A @ (w_k - x_k)
        #     if rho_k[0, 0] >= sigma * sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) \
        #             * sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)):
        #         x_k = x_k - lambda_ * rho_k[0, 0] / (sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) ** 2) * (w_k - T_w_k)
        #         break
        #     else:
        #         tau_k = beta * tau_k
        r_safe = 0
        supermann_eta = 0
        x_k, r_safe, supermann_eta, s_cache, y_cache = SuperMann.SuperMann(epsilon, z_prev, eta_prev, z_next, eta_next, op_A, m, i,
                                                            n_z, alpha, prediction_horizon,
                                                            initial_state, L, L_adj, state_dynamics, control_dynamics,
                                                            control_weight, P_seq,
                                                            R_tilde_seq, K_seq,
                                                            A_bar_seq, stage_state, terminal_state, stage_sets,
                                                            terminal_set, s_cache, y_cache, n_max,
                                                            c0, c1, q, sigma, lambda_, beta, r_safe, supermann_eta)
        # update z, eta to CP
        z_next = x_k[0:n_z]
        eta_next = x_k[n_z:2 * n_z]

        # Termination criteria
        xi_1 = (z_prev - z_next) / alpha - L_adj @ (eta_prev - eta_next)
        xi_2 = (eta_prev - eta_next) / alpha + L @ (z_next - z_prev)
        xi_gap = xi_1 + L_adj @ xi_2
        t_1 = np.linalg.norm(xi_1, np.inf)
        t_2 = np.linalg.norm(xi_2, np.inf)
        t_3 = np.linalg.norm(xi_gap, np.inf)
        residuals_cache[i, 0] = t_1
        residuals_cache[i, 1] = t_2
        residuals_cache[i, 2] = t_3
        if t_1 <= epsilon and t_2 <= epsilon and t_3 <= epsilon:
            break
    status = 0  # converge success
    if i >= 9000:
        status = 1  # converge failed
    residuals_cache = residuals_cache[0:i, :]
    return residuals_cache, z_next, status
