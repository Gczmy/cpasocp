import numpy as np


class L_BFGS:
    def __init__(self, prediction_horizon, initial_state, memory_num, state_dynamics, stage_state_weight, control_weight, terminal_state_weight):
        self.__N = prediction_horizon
        self.__x0 = initial_state
        self.__m = memory_num
        self.__A = state_dynamics
        self.__Q = stage_state_weight
        self.__R = control_weight
        self.__P = terminal_state_weight
        self.__alpha = None
        self.__s_cache = [None] * memory_num
        self.__y_cache = [None] * memory_num
        self.__yTs_cache = [None] * memory_num
        self.__state_cache = [None] * prediction_horizon

    def make_alpha(self):
        alpha = 1
        return alpha

    def l_bfgs_algorithm(self):
        N = self.__N
        x0 = self.__x0
        m = self.__m
        A = self.__A
        Q = self.__Q
        R = self.__R
        P = self.__P


        n_x = A.shape[1]
        N_max = 10
        for converge_loop in range(N_max):
            state_cache = np.zeros((n_x, 1, N + 1))  # tensor
            state_cache[:, :, N] = x0
            z = x0
            x_prev = x0
            gradient_prev = Q @ x0
            for k in range(N):
                oldest_flag = k % m

                gamma_k = self.__yTs_cache[k] / (self.__y_cache[k].T @ self.__y_cache[k])
                H_0_k = gamma_k * np.eye(n_x)

                alpha_k = L_BFGS.make_alpha()

                gradient_0 = Q @ x0
                x1 = x0 - alpha_k * H_0_k @ gradient_0
                gradient_1 = Q @ x1
                self.__s_cache[oldest_flag] = x1 - x0
                self.__y_cache[oldest_flag] = gradient_1 - gradient_0
                self.__yTs_cache[oldest_flag] = self.__y_cache[oldest_flag].T @ self.__y_cache[oldest_flag]
                rho_k = 1 / self.__yTs_cache[oldest_flag]
                V_k = np.eye(n_x) - rho_k * self.__y_cache[oldest_flag] @ self.__s_cache[oldest_flag].T

                # L-BFGS two-loop recursion
                q = gradient_prev
                if k >= m:
                    for i in range(m):
                        rho_i = 1 / self.__yTs_cache[m - i]
                        alpha_i = rho_i * self.__s_cache[m - i].T @ q
                        q = q - alpha_i * self.__y_cache[m - i]
                    r = H_0_k @ q
                    for i in range(m):
                        rho_i = 1 / self.__yTs_cache[i]
                        beta = rho_i * self.__y_cache[i].T @ r
                        r = r + self.__s_cache[i] * (alpha_i - beta)
                else:
                    for i in range(k):
                        rho_i = 1 / self.__yTs_cache[k - i]
                        alpha_i = rho_i * self.__s_cache[k - i].T @ q
                        q = q - alpha_i * self.__y_cache[k - i]
                    r = H_0_k @ q
                    for i in range(k):
                        rho_i = 1 / self.__yTs_cache[i]
                        beta = rho_i * self.__y_cache[i].T @ r
                        r = r + self.__s_cache[i] * (alpha_i - beta)
                p_k = - r
                oldest_flag = (k + 1) % m
                x_next = x_prev + alpha_k * p_k
                gradient_next = Q @ x_next
                self.__s_cache[oldest_flag] = x_next - x_prev
                self.__y_cache[oldest_flag] = gradient_next - gradient_prev
                self.__yTs_cache[oldest_flag] = self.__y_cache[oldest_flag].T @ self.__y_cache[oldest_flag]
                x_prev = x_next
                gradient_prev = gradient_next

                state_cache[:, :, N - k - 1] = x_next
                z = np.vstack((z, x_next))
            return z
