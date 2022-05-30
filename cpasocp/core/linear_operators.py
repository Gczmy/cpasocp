import numpy as np
import scipy.sparse.linalg as lingalg


class LinearOperator:
    def __init__(self, prediction_horizon, state_dynamics, control_dynamics, stage_state, stage_control,
                 terminal_state):
        """
        :param prediction_horizon: prediction horizon (N) of dynamic system
        :param state_dynamics: matrix A, describing the state dynamics
        :param control_dynamics: matrix B, describing control dynamics
        :param stage_state: matrix Gamma_x, describing the state constraints
        :param stage_control: matrix Gamma_u, describing control constraints
        :param terminal_state: matrix Gamma_N, describing terminal constraints
        """
        self.__prediction_horizon = prediction_horizon
        self.__A = state_dynamics
        self.__B = control_dynamics
        self.__Gamma_x = stage_state
        self.__Gamma_u = stage_control
        self.__Gamma_N = terminal_state
        self.make_L_op()
        self.make_L_adj()

    def make_L_op(self):
        """Construct LinearOperator L"""
        N = self.__prediction_horizon
        Gamma_x = self.__Gamma_x
        Gamma_u = self.__Gamma_u
        Gamma_N = self.__Gamma_N
        n_x = self.__A.shape[1]
        n_u = self.__B.shape[1]
        n_c = self.__Gamma_x.shape[0]
        n_f = self.__Gamma_N.shape[0]
        n_z = (N + 1) * n_x + N * n_u
        n_L = N * n_c + n_f

        def matvec(v):
            L = Gamma_x @ v[0:n_x] + Gamma_u @ v[n_x:n_x + n_u]
            L = np.reshape(L, (n_c, 1))

            for i in range(1, N):
                L_i = Gamma_x @ v[i * (n_x + n_u): i * (n_x + n_u) + n_x] + Gamma_u \
                      @ v[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
                L = np.vstack((L, np.reshape(L_i, (n_c, 1))))

            L_N = Gamma_N @ v[N * (n_x + n_u): N * (n_x + n_u) + n_x]
            L = np.vstack((L, np.reshape(L_N, (n_f, 1))))
            return L

        return lingalg.LinearOperator((n_L, n_z), matvec=matvec)

    def make_L_adj(self):
        """Construct LinearOperator adjoint L_adj"""
        N = self.__prediction_horizon
        Gamma_x = self.__Gamma_x
        Gamma_u = self.__Gamma_u
        Gamma_N = self.__Gamma_N
        n_x = self.__A.shape[1]
        n_u = self.__B.shape[1]
        n_c = self.__Gamma_x.shape[0]
        n_f = self.__Gamma_N.shape[0]
        n_z = (N + 1) * n_x + N * n_u
        n_L = N * n_c + n_f

        def matvec(v):
            L_adj_x = np.reshape(Gamma_x.T @ v[0:n_c], (n_x, 1))
            L_adj_u = np.reshape(Gamma_u.T @ v[0:n_c], (n_u, 1))
            L_adj = np.vstack((L_adj_x, L_adj_u))

            for i in range(1, N):
                L_adj_i_x = np.reshape(Gamma_x.T @ v[i * n_c: (i + 1) * n_c], (n_x, 1))
                L_adj_i_u = np.reshape(Gamma_u.T @ v[i * n_c: (i + 1) * n_c], (n_u, 1))
                L_adj = np.vstack((L_adj, L_adj_i_x, L_adj_i_u))

            L_adj_N = np.reshape(Gamma_N.T @ v[N * n_c: (N + 1) * n_f], (n_x, 1))
            L_adj = np.vstack((L_adj, L_adj_N))
            return L_adj

        return lingalg.LinearOperator((n_z, n_L), matvec=matvec)
