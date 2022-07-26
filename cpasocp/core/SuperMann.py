import numpy as np
from math import sqrt as sqrt
import cpasocp.core.proximal_online_part as core_online
import cpasocp.core.chambolle_pock_algorithm as core_cp


def SuperMann(epsilon, z_prev, eta_prev, z_next, eta_next, op_A, memory_num, loop_num, n_z, alpha, prediction_horizon,
              initial_state, L, L_adj, state_dynamics, control_dynamics, control_weight, P_seq, R_tilde_seq, K_seq,
              A_bar_seq, stage_state, terminal_state, stage_sets, terminal_set, s_cache, y_cache, n_max, c0, c1, q,
              sigma, lambda_, beta, r_safe, supermann_eta):
    N = prediction_horizon
    x0 = initial_state
    A = state_dynamics
    B = control_dynamics
    R = control_weight
    Gamma_x = stage_state
    Gamma_N = terminal_state
    C_t = stage_sets
    C_N = terminal_set
    m = memory_num
    i = loop_num

    x_k = np.vstack((z_prev, eta_prev))
    T_x_k = np.vstack((z_next, eta_next))
    if i == 0:
        r_safe = sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k))
        supermann_eta = r_safe
    if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= epsilon:
        return x_k, r_safe, supermann_eta, s_cache, y_cache

    # Choose an update direction
    m_i = min(m, i)
    oldest_flag = i % m
    T_x_k_z_prev = T_x_k[0: n_z]
    T_x_k_eta_prev = T_x_k[n_z: 2 * n_z]
    T_x_k_z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
                                                         proximal_lambda=alpha,
                                                         initial_state=x0,
                                                         initial_guess_vector=T_x_k_z_prev
                                                                              - alpha * L_adj @ T_x_k_eta_prev,
                                                         state_dynamics=A,
                                                         control_dynamics=B,
                                                         control_weight=R,
                                                         P_seq=P_seq,
                                                         R_tilde_seq=R_tilde_seq,
                                                         K_seq=K_seq,
                                                         A_bar_seq=A_bar_seq)
    T_x_k_eta_half_next = T_x_k_eta_prev + alpha * L @ (2 * T_x_k_z_next - T_x_k_z_prev)
    T_x_k_eta_next = T_x_k_eta_half_next - alpha * core_cp.proj_to_c(T_x_k_eta_half_next / alpha,
                                                                     N, Gamma_x, Gamma_N, C_t, C_N)
    TT_x_k = np.vstack((T_x_k_z_next, T_x_k_eta_next))
    s_cache[oldest_flag] = x_k - T_x_k
    y_cache[oldest_flag] = (x_k - T_x_k) - (T_x_k - TT_x_k)
    r_k = x_k - T_x_k
    S_k = s_cache[0]
    Y_k = y_cache[0]
    for k in range(1, m_i + 1):
        if k < m:
            S_k = np.hstack((S_k, s_cache[k]))
            Y_k = np.hstack((Y_k, y_cache[k]))
    t_k = np.linalg.lstsq(Y_k, r_k, rcond=None)[0]
    d_k = -r_k - (S_k - Y_k) @ t_k

    # K0
    if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= c0 * supermann_eta:
        supermann_eta = sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k))
        w_k = x_k + d_k
        x_k = w_k
        # # update z, eta to CP
        # z_next = x_k[0:n_z]
        # eta_next = x_k[n_z:2 * n_z]
        return x_k, r_safe, supermann_eta, s_cache, y_cache
    tau_k = 1
    for k in range(n_max):
        w_k = x_k + tau_k * d_k
        w_z_prev = w_k[0:n_z]
        w_eta_prev = w_k[n_z:2 * n_z]
        w_z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
                                                         proximal_lambda=alpha,
                                                         initial_state=x0,
                                                         initial_guess_vector=w_z_prev - alpha * L_adj @ w_eta_prev,
                                                         state_dynamics=A,
                                                         control_dynamics=B,
                                                         control_weight=R,
                                                         P_seq=P_seq,
                                                         R_tilde_seq=R_tilde_seq,
                                                         K_seq=K_seq,
                                                         A_bar_seq=A_bar_seq)
        w_eta_half_next = w_eta_prev + alpha * L @ (2 * w_z_next - w_z_prev)
        w_eta_next = w_eta_half_next - alpha * core_cp.proj_to_c(w_eta_half_next / alpha, N, Gamma_x, Gamma_N, C_t, C_N)
        T_w_k = np.vstack((w_z_next, w_eta_next))
        # K1
        if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= r_safe \
                and sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) <= c1 * sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)):
            x_k = w_k
            r_safe = sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) + q ** i
            break
        # K2
        rho_k = sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) ** 2 \
                - 2 * 0.5 * (w_k - T_w_k).T @ op_A @ (w_k - x_k)
        if rho_k[0, 0] >= sigma * sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) \
                * sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)):
            x_k = x_k - lambda_ * rho_k[0, 0] / (sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) ** 2) * (w_k - T_w_k)
            break
        else:
            tau_k = beta * tau_k
    return x_k, r_safe, supermann_eta, s_cache, y_cache
