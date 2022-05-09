import numpy as np
import cpasocp.core.proximal_online_part as core_online


def chambolle_pock_algorithm_for_ocp(epsilon, initial_guess_z, initial_guess_eta, Phi, Phi_z, Phi_star,
                                     prediction_horizon,
                                     initial_state, state_dynamics, control_dynamics, control_weight, P_seq,
                                     R_tilde_seq, K_seq, A_bar_seq):
    N = prediction_horizon
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    n_x = A.shape[1]
    n_u = B.shape[1]
    n_z = (N + 1) * n_x + N * n_u
    n_Phi = Phi_z.shape[0]
    x0 = initial_state
    z0 = initial_guess_z
    eta0 = initial_guess_eta
    if z0.shape[0] != N * (n_x + n_u) + n_x:
        raise ValueError("Initial guess vector z row is not correct")
    if eta0.shape[0] != n_Phi:
        raise ValueError("Initial guess vector eta row is not correct")

    # Choose α1, α2 > 0 such that α1α2∥Phi∥^2 < 1
    alpha_1 = 0.99 / np.linalg.norm(Phi_z)
    alpha_2 = 0.99 / np.linalg.norm(Phi_z)

    z_prev = z0
    z_next = z0
    eta_prev = eta0
    eta_half_next = eta0
    eta_next = eta0

    while (np.abs(z_next - z_prev) < epsilon).all() and (np.abs(eta_next - eta_prev) < epsilon).all():
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
        eta_half_next = eta_prev + alpha_2 * Phi @ (
                2 * z_next - z_prev)
        eta_next = eta_half_next - alpha_2 * (1 / alpha_2 * eta_half_next)

    return z_next, eta_next