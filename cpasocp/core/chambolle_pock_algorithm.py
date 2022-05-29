import numpy as np
import cpasocp.core.proximal_online_part as core_online
import matplotlib.pyplot as plt


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
        vector_list = [None] * prediction_horizon
        for i in range(prediction_horizon):
            vector_list[i] = vector[i * n_c:(i + 1) * n_c]
        vector_stage = stage_sets.project(vector_list)
    else:
        vector_stage = stage_sets.project(vector[0:N * n_c])
    vector_terminal = terminal_set.project(vector[N * n_c:N * n_c + n_f])
    vector = np.vstack((vector_stage, vector_terminal))
    return vector


# def determine_tau(stage_sets, terminal_set):
#
#     return tau


def chambolle_pock_algorithm_for_ocp(epsilon, initial_guess_z, initial_guess_eta, alpha, L, L_z, L_adj,
                                     prediction_horizon, initial_state, state_dynamics, control_dynamics,
                                     control_weight, P_seq, R_tilde_seq, K_seq, A_bar_seq, stage_state, terminal_state,
                                     stage_sets, terminal_set):
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
    C_t = stage_sets
    C_N = terminal_set
    n_x = A.shape[1]
    n_u = B.shape[1]
    n_z = N * (n_x + n_u) + n_x
    n_L = L_z.shape[0]
    x0 = initial_state
    z0 = initial_guess_z
    eta0 = initial_guess_eta
    if z0.shape[0] != N * (n_x + n_u) + n_x:
        raise ValueError("Initial guess vector z row is not correct")
    if eta0.shape[0] != n_L:
        raise ValueError("Initial guess vector eta row is not correct")

    # Choose α1, α2 > 0 such that α1α2∥Phi∥^2 < 1

    z_next = z0
    eta_next = eta0
    n_max = 1000
    residuals_cache = np.zeros((n_max, 3))

    for i in range(n_max):
        z_prev = z_next
        eta_prev = eta_next
        z_next = core_online.proximal_of_h_online_part(prediction_horizon=prediction_horizon,
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
        eta_next = eta_half_next - alpha * proj_to_c(1 / alpha * eta_half_next, prediction_horizon, stage_state,
                                                     terminal_state, C_t, C_N)

        # Termination criteria
        xi_1 = (z_next - z_prev) / alpha - L_adj @ (eta_prev - eta_next)
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
    residuals_cache = residuals_cache[0:i, :]

    # plot
    plt.xlabel('Iterations')
    plt.ylabel('Residuals')
    # plt.plot(residuals_cache, label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
    plt.semilogy(residuals_cache, label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
    plt.legend()
    plt.show()
    return z_next, eta_next
