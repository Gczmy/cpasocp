# import numpy as np
# from math import sqrt as sqrt
# import cpasocp.core.proximal_online_part as core_online
# import cpasocp.core.chambolle_pock as core_cp
#
#
# class SuperMann:
#     """
#     SuperMann acceleration
#     """
#
#     def __init__(self, problem_spec: CPASOCP):
#         self.__solver = problem_spec.solver
#         self.__s_cache = None
#         self.__y_cache = None
#         self.__r_safe = None
#         self.__supermann_eta = None
#         self.__N = problem_spec.get_prediction_horizon
#         # self.__A = state_dynamics
#         # self.__B = control_dynamics
#         # self.__R = control_weight
#         # self.__x0 = initial_state
#         # self.__z0 = initial_guess_z
#         # self.__eta0 = initial_guess_eta
#         # self.__epsilon = epsilon
#         # self.__P_seq = P_seq
#         # self.__R_tilde_seq = R_tilde_seq
#         # self.__K_seq = K_seq
#         # self.__A_bar_seq = A_bar_seq
#         # self.__alpha = alpha
#         # self.__L = L
#         # self.__L_adj = L_adj
#         self.__loop_time = None
#
#     def main(self, loop_time, z_prev, eta_prev, z_next, eta_next, op_A, memory_num, c0, c1, q, beta, sigma, lambda_):
#         N = self.__N
#         print(N)
#         # x0 = self.__x0
#         # A = self.__A
#         # B = self.__B
#         # R = self.__R
#         # m = memory_num
#         # i = loop_time
#         # n_x = A.shape[1]
#         # n_u = B.shape[1]
#         # n_z = N * (n_x + n_u) + n_x
#         # alpha = self.__alpha
#         # L = self.__L
#         # L_adj = self.__L_adj
#         # P_seq = self.__P_seq
#         # R_tilde_seq = self.__R_tilde_seq
#         # K_seq = self.__K_seq
#         # A_bar_seq = self.__A_bar_seq
#         # n_max = 10000
#         # x_k = np.vstack((z_prev, eta_prev))
#         # T_x_k = np.vstack((z_next, eta_next))
#         # if i == 0:
#         #     self.__r_safe = sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k))
#         #     self.__supermann_eta = self.__r_safe
#         # if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= self.__epsilon:
#         #     return x_k
#         #
#         # # Choose an update direction
#         # m_i = min(m, i)
#         # oldest_flag = i % m
#         # T_x_k_z_prev = T_x_k[0: n_z]
#         # T_x_k_eta_prev = T_x_k[n_z: 2 * n_z]
#         # T_x_k_z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
#         #                                                      proximal_lambda=alpha,
#         #                                                      initial_state=x0,
#         #                                                      initial_guess_vector=T_x_k_z_prev
#         #                                                                           - alpha * L_adj @ T_x_k_eta_prev,
#         #                                                      state_dynamics=A,
#         #                                                      control_dynamics=B,
#         #                                                      control_weight=R,
#         #                                                      P_seq=P_seq,
#         #                                                      R_tilde_seq=R_tilde_seq,
#         #                                                      K_seq=K_seq,
#         #                                                      A_bar_seq=A_bar_seq)
#         # T_x_k_eta_half_next = T_x_k_eta_prev + alpha * L @ (2 * T_x_k_z_next - T_x_k_z_prev)
#         # T_x_k_eta_next = T_x_k_eta_half_next - alpha * self.__solver.proj_to_c(self.__solver,
#         #                                                                        T_x_k_eta_half_next / alpha)
#         # TT_x_k = np.vstack((T_x_k_z_next, T_x_k_eta_next))
#         # self.__s_cache[oldest_flag] = x_k - T_x_k
#         # self.__y_cache[oldest_flag] = (x_k - T_x_k) - (T_x_k - TT_x_k)
#         # r_k = x_k - T_x_k
#         # S_k = self.__s_cache[0]
#         # Y_k = self.__y_cache[0]
#         # for k in range(1, m_i + 1):
#         #     if k < m:
#         #         S_k = np.hstack((S_k, self.__s_cache[k]))
#         #         Y_k = np.hstack((Y_k, self.__y_cache[k]))
#         # t_k = np.linalg.lstsq(Y_k, r_k, rcond=None)[0]
#         # d_k = -r_k - (S_k - Y_k) @ t_k
#         #
#         # # K0
#         # if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= c0 * self.__supermann_eta:
#         #     self.__supermann_eta = sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k))
#         #     w_k = x_k + d_k
#         #     x_k = w_k
#         #     return x_k
#         # tau_k = 1
#         # for k in range(n_max):
#         #     w_k = x_k + tau_k * d_k
#         #     w_z_prev = w_k[0:n_z]
#         #     w_eta_prev = w_k[n_z:2 * n_z]
#         #     w_z_next = core_online.proximal_of_h_online_part(prediction_horizon=N,
#         #                                                      proximal_lambda=alpha,
#         #                                                      initial_state=x0,
#         #                                                      initial_guess_vector=w_z_prev - alpha * L_adj @ w_eta_prev,
#         #                                                      state_dynamics=A,
#         #                                                      control_dynamics=B,
#         #                                                      control_weight=R,
#         #                                                      P_seq=P_seq,
#         #                                                      R_tilde_seq=R_tilde_seq,
#         #                                                      K_seq=K_seq,
#         #                                                      A_bar_seq=A_bar_seq)
#         #     w_eta_half_next = w_eta_prev + alpha * L @ (2 * w_z_next - w_z_prev)
#         #     w_eta_next = w_eta_half_next - alpha * self.__solver.proj_to_c(self.__solver, w_eta_half_next / alpha)
#         #     T_w_k = np.vstack((w_z_next, w_eta_next))
#         #     # K1
#         #     if sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)) <= self.__r_safe \
#         #             and sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) <= c1 * sqrt(
#         #         (x_k - T_x_k).T @ op_A @ (x_k - T_x_k)):
#         #         x_k = w_k
#         #         self.__r_safe = sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) + q ** i
#         #         break
#         #     # K2
#         #     rho_k = sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) ** 2 \
#         #             - 2 * 0.5 * (w_k - T_w_k).T @ op_A @ (w_k - x_k)
#         #     if rho_k[0, 0] >= sigma * sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) \
#         #             * sqrt((x_k - T_x_k).T @ op_A @ (x_k - T_x_k)):
#         #         x_k = x_k - lambda_ * rho_k[0, 0] / (sqrt((w_k - T_w_k).T @ op_A @ (w_k - T_w_k)) ** 2) * (
#         #                 w_k - T_w_k)
#         #         break
#         #     else:
#         #         tau_k = beta * tau_k
#         # return x_k
