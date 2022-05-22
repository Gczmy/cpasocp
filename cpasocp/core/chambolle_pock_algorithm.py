import numpy as np
import cpasocp.core.proximal_online_part as core_online


def proj_to_c(vector, prediction_horizon, stage_state, terminal_state, stage_sets, terminal_set):
    """
    :param vector: the vector to be projected to sets (C_t) and C_N
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param stage_state: matrix (Gamma_x), describing the state constraints
    :param terminal_state: matrix (Gamma_N), describing terminal constraints
    :param stage_sets: nonempty convex closed sets (C) which is the cartesian product of sets (C_t), describing
    state-control constraints
    :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
    """
    N = prediction_horizon
    n_c = stage_state.shape[0]
    n_f = terminal_state.shape[0]
    vector_list = [None] * prediction_horizon
    for i in range(prediction_horizon):
        vector_list[i] = vector[i * n_c:(i + 1) * n_c]
    vector_stage = stage_sets.project(vector_list)
    vector_terminal = terminal_set.project(vector[N * n_c:N * n_c + n_f])
    vector = np.vstack((vector_stage, vector_terminal))
    return vector

# def determine_tau(stage_sets, terminal_set):
#
#     return tau


def chambolle_pock_algorithm_for_ocp(epsilon, initial_guess_z, initial_guess_eta, Phi, Phi_z, Phi_star,
                                     prediction_horizon, initial_state, state_dynamics, control_dynamics,
                                     control_weight, P_seq, R_tilde_seq, K_seq, A_bar_seq, stage_state, terminal_state,
                                     stage_sets, terminal_set):
    """
    :param epsilon: scalar (epsilon) of Chambolle-Pock algorithm
    :param initial_guess_z: vector initial guess of (z0) of Chambolle-Pock algorithm
    :param initial_guess_eta: vector initial guess of (eta0) of Chambolle-Pock algorithm
    :param Phi: LinearOperator of (Phi) of Chambolle-Pock algorithm
    :param Phi_z: vector of (Phi_z) of Chambolle-Pock algorithm
    :param Phi_star: LinearOperator of (Phi_star) of Chambolle-Pock algorithm
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
    :param stage_sets: nonempty convex closed sets (C_t), describing state-control constraints
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
    n_Phi = Phi_z.shape[0]
    x0 = initial_state
    z0 = initial_guess_z
    eta0 = initial_guess_eta
    if z0.shape[0] != N * (n_x + n_u) + n_x:
        raise ValueError("Initial guess vector z row is not correct")
    if eta0.shape[0] != n_Phi:
        raise ValueError("Initial guess vector eta row is not correct")

    # Choose α1, α2 > 0 such that α1α2∥Phi∥^2 < 1
    alpha_1 = 0.99 / np.linalg.norm(Phi @ np.eye(n_z) * 0.2)
    alpha_2 = 0.99 / np.linalg.norm(Phi @ np.eye(n_z) * 0.2)

    z_next = z0
    eta_next = eta0

    loop_end = False

    while loop_end is False:
        z_prev = z_next
        eta_prev = eta_next
        z_next = core_online.proximal_of_h_online_part(prediction_horizon=prediction_horizon,
                                                       proximal_lambda=alpha_1,
                                                       initial_state=x0,
                                                       initial_guess_vector=z_prev - alpha_1 * Phi_star @ eta_prev,
                                                       state_dynamics=A,
                                                       control_dynamics=B,
                                                       control_weight=R,
                                                       P_seq=P_seq,
                                                       R_tilde_seq=R_tilde_seq,
                                                       K_seq=K_seq,
                                                       A_bar_seq=A_bar_seq)
        eta_half_next = eta_prev + alpha_2 * Phi @ (2 * z_next - z_prev)
        eta_next = eta_half_next - alpha_2 * proj_to_c(1 / alpha_2 * eta_half_next, prediction_horizon,
                                                       stage_state, terminal_state, C_t, C_N)

        # Termination criteria
        xi_1 = (z_next - z_prev) / alpha_1 - Phi_star @ (eta_prev - eta_next)
        xi_2 = (eta_prev - eta_next) / alpha_1 + Phi @ (z_next - z_prev)
        if np.linalg.norm(xi_1, np.inf) <= epsilon and np.linalg.norm(xi_2, np.inf) <= epsilon \
                and np.linalg.norm(xi_1 + Phi_star @ xi_2, np.inf) <= epsilon:
            loop_end = True

    return z_next, eta_next
