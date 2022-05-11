import numpy as np
import cpasocp.core.proximal_online_part as core_online


def proj_c(stage_ncc_sets_constraints, terminal_ncc_set_constraints, vector, prediction_horizon,
           stage_state_constraints, terminal_state_constraints):
    """
    :param stage_ncc_sets_constraints: nonempty convex closed sets C_t, describing state-control constraints
    :param terminal_ncc_set_constraints: nonempty convex closed set C_N, describing terminal constraints
    :param vector: the vector to be projected to sets C_t and C_N
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param stage_state_constraints: matrix Gamma_x, describing the state constraints
    :param terminal_state_constraints: matrix Gamma_N, describing terminal constraints
    """
    N = prediction_horizon
    n_c = stage_state_constraints.shape[0]
    n_f = terminal_state_constraints.shape[0]
    vector_stage = stage_ncc_sets_constraints.project(vector[0:N * n_c])
    vector_terminal = terminal_ncc_set_constraints.project(vector[N * n_c:N * n_c + n_f])
    vector = np.vstack((vector_stage, vector_terminal))
    return vector


def chambolle_pock_algorithm_for_ocp(epsilon, initial_guess_z, initial_guess_eta, Phi, Phi_z, Phi_star,
                                     prediction_horizon, initial_state, state_dynamics, control_dynamics,
                                     control_weight, P_seq, R_tilde_seq, K_seq, A_bar_seq, stage_state_constraints,
                                     terminal_state_constraints, stage_ncc_sets_constraints,
                                     terminal_ncc_set_constraints):
    N = prediction_horizon
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    C_t = stage_ncc_sets_constraints
    C_N = terminal_ncc_set_constraints
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
    alpha_1 = 0.99 / np.linalg.norm(Phi @ np.eye(n_z) * 0.5)
    alpha_2 = 0.99 / np.linalg.norm(Phi @ np.eye(n_z) * 0.5)

    z_next = z0
    eta_next = eta0

    z_proximal_end = [False] * n_z
    eta_proximal_end = [False] * n_Phi

    while (all(z_proximal_end) and all(eta_proximal_end)) is False:
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
        eta_next = eta_half_next - alpha_2 * proj_c(C_t, C_N, 1 / alpha_2 * eta_half_next, prediction_horizon,
                                                    stage_state_constraints, terminal_state_constraints)
        for i in range(n_z):
            if np.abs(z_next[i] - z_prev[i]) < epsilon:
                z_proximal_end[i] = True
        for i in range(n_Phi):
            if np.abs(eta_next[i] - eta_prev[i]) < epsilon:
                eta_proximal_end[i] = True
    return z_next, eta_next
