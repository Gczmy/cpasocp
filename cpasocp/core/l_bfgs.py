import numpy as np


class LBFGS:
    def __init__(self, prediction_horizon, epsilon, initial_state, memory_num, state_dynamics, stage_state_weight,
                 control_weight, terminal_state_weight, stage_state_weight2):
        self.__N = prediction_horizon
        self.__epsilon = epsilon
        self.__x0 = initial_state
        self.__m = memory_num
        self.__A = state_dynamics
        self.__Q = stage_state_weight
        self.__R = control_weight
        self.__P = terminal_state_weight
        self.__q = stage_state_weight2
        self.__alpha = None
        self.__s_cache = [None] * memory_num
        self.__y_cache = [None] * memory_num
        self.__yTs_cache = [None] * memory_num

    def f(self, v):
        # f = 0.5 * x.T @ self.__Q @ x + self.__q.T @ x
        f = 0
        n_x = self.__Q.shape[0]
        n_u = self.__R.shape[0]
        for i in range(self.__N):
            x = v[i * (n_x + n_u): i * (n_x + n_u) + n_x]
            u = v[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
            f += 0.5 * (x.T @ self.__Q @ x + u.T @ self.__R @ u)
        x = v[self.__N * (n_x + n_u): self.__N * (n_x + n_u) + n_x]
        f += 0.5 * x.T @ self.__P @ x
        return f

    def gf(self, v):
        # gf = self.__Q @ x + self.__q
        n_x = self.__Q.shape[0]
        n_u = self.__R.shape[0]
        n_z = self.__N * (n_x + n_u) + n_x
        z = v[0:n_z]
        x = z[0: n_x]
        gf_x = self.__Q @ x
        u = z[n_x: n_x + n_u]
        gf_u = self.__R @ u
        gf = np.vstack((gf_x, gf_u))
        for i in range(1, self.__N):
            x = z[i * (n_x + n_u): i * (n_x + n_u) + n_x]
            gf_x = self.__Q @ x
            gf = np.vstack((gf, gf_x))
            u = z[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
            gf_u = self.__R @ u
            gf = np.vstack((gf, gf_u))
        x = z[self.__N * (n_x + n_u): self.__N * (n_x + n_u) + n_x]
        gf_N = self.__P @ x
        gf = np.vstack((gf, gf_N))
        return gf

    def two_loop_recursion(self, gf, H_0_k, k, m, oldest_flag):
        q = gf
        alpha = [None] * m
        rho = [None] * m
        if k >= m:
            for i in range(m):
                new_to_old_index = oldest_flag - 1 - i
                rho[new_to_old_index] = 1 / self.__yTs_cache[new_to_old_index]
            for i in range(m):
                new_to_old_index = oldest_flag - 1 - i
                alpha[new_to_old_index] = rho[new_to_old_index] * self.__s_cache[new_to_old_index].T @ q
                q = q - alpha[new_to_old_index] * self.__y_cache[new_to_old_index]
            r = H_0_k @ q
            for i in range(m):
                old_to_new_index = oldest_flag - m + i
                beta = rho[old_to_new_index] * self.__y_cache[old_to_new_index].T @ r
                r = r + self.__s_cache[old_to_new_index] * (alpha[old_to_new_index] - beta)
        else:
            for i in range(k):
                new_to_old_index = oldest_flag - 1 - i
                rho[new_to_old_index] = 1 / self.__yTs_cache[new_to_old_index]
            for i in range(k):
                new_to_old_index = oldest_flag - 1 - i
                alpha[new_to_old_index] = rho[new_to_old_index] * (self.__s_cache[new_to_old_index].T @ q)[0, 0]
                q = q - alpha[new_to_old_index] * self.__y_cache[new_to_old_index]
            r = H_0_k @ q
            for i in range(k):
                old_to_new_index = i
                beta = rho[old_to_new_index] * (self.__y_cache[old_to_new_index].T @ r)[0, 0]
                r = r + self.__s_cache[old_to_new_index] * (alpha[old_to_new_index] - beta)
        return -r

    def make_alpha(self, x, p):
        alpha = 1
        c_1 = 1e-4
        c_2 = 0.9
        running = True
        while running:
            Armijo_condition = LBFGS.f(self, x + alpha * p) <= LBFGS.f(self, x) + c_1 * alpha * LBFGS.gf(self, x).T @ p
            curvature_condition = LBFGS.gf(self, x + alpha * p).T @ p >= c_2 * LBFGS.gf(self, x).T @ p
            if Armijo_condition and curvature_condition:
                break
            alpha *= 0.5
        return alpha

    def make_x_s_y(self, x_prev, p):
        alpha = LBFGS.make_alpha(self, x_prev, p)
        gf_prev = LBFGS.gf(self, x_prev)
        x_next = x_prev + alpha * p
        gf_next = LBFGS.gf(self, x_next)

        s = x_next - x_prev
        y = gf_next - gf_prev
        yTs = (y.T @ s)[0, 0]

        return x_next, gf_next, s, y, yTs

    def l_bfgs_algorithm(self):
        epsilon = self.__epsilon
        x0 = self.__x0
        m = self.__m
        A = self.__A
        R = self.__R
        n_x = A.shape[1]
        n_u = R.shape[1]
        n_z = self.__N * (n_x + n_u) + n_x
        x0 = np.reshape(x0, (n_z, 1))
        x_prev = x0
        gradient_prev = LBFGS.gf(self, x0)
        runing = True
        k = 0
        grad_cache = []
        while runing:
            alpha = LBFGS.make_alpha(self, x_prev, -gradient_prev)
            if k == 0:
                # gradient descent method for k=0
                p = - alpha * gradient_prev
                x_next, gradient_next, self.__s_cache[0], self.__y_cache[0], self.__yTs_cache[0] \
                    = LBFGS.make_x_s_y(self, x_prev, p)
            else:
                # L-BFGS two-loop recursion
                oldest_flag = k % m
                top = self.__s_cache[(k - 1) % m].T @ self.__y_cache[(k - 1) % m]
                bottom = self.__y_cache[(k - 1) % m].T @ self.__y_cache[(k - 1) % m]
                gamma_k = top / bottom
                H_0_k = gamma_k * np.eye(n_z)
                p = LBFGS.two_loop_recursion(self, gradient_prev, H_0_k, k, m, oldest_flag)
                x_next, gradient_next, self.__s_cache[oldest_flag], self.__y_cache[oldest_flag], self.__yTs_cache[
                    oldest_flag] \
                    = LBFGS.make_x_s_y(self, x_prev, p)

            x_prev = x_next
            gradient_prev = gradient_next
            max_grad = np.linalg.norm(gradient_next, np.inf)
            grad_cache.append(max_grad)
            if max_grad < epsilon:
                break
            k = k + 1

        return x_next, k, grad_cache
