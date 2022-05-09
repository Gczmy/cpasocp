import numpy as np
import scipy.sparse.linalg as lingalg


def proximal_of_h_offline_part(prediction_horizon, proximal_lambda, state_dynamics, control_dynamics,
                               stage_state_weight, control_weight, terminal_state_weight):
    """
    :param prediction_horizon: matrix Gamma_x, describing the state constraints
    :param proximal_lambda: a parameter lambda for proximal operator
    :param prediction_horizon: prediction horizon (N) of dynamic system
    :param state_dynamics: matrix A, describing the state dynamics
    :param control_dynamics: matrix B, describing control dynamics
    :param stage_state_weight: matrix Q, stage state cost matrix
    :param control_weight: scalar or matrix R, input cost matrix or scalar
    :param terminal_state_weight: matrix P, terminal state cost matrix
    """
    A = state_dynamics
    B = control_dynamics
    Q = stage_state_weight
    R = control_weight
    P = terminal_state_weight
    n_x = A.shape[1]
    n_u = B.shape[1]
    N = prediction_horizon
    P_seq = np.zeros((n_x, n_x, N + 1))  # tensor
    R_tilde_seq = np.zeros((n_x, n_x, N))  # tensor
    K_seq = np.zeros((n_x, n_x, N))  # tensor
    A_bar_seq = np.zeros((n_x, n_x, N))  # tensor
    P_0 = P + proximal_lambda * np.eye(n_x)
    P_seq[:, :, N] = P_0

    for i in range(N):
        R_tilde_seq[:, :, N - i - 1] = R + 1 / proximal_lambda * np.eye(n_u) + B.T @ P_seq[:, :, N - i - 1] @ B
        K_seq[:, :, N - i - 1] = - np.linalg.inv(R_tilde_seq[:, :, N - i - 1]) @ B.T @ P_seq[:, :, N - i - 1] @ A
        A_bar_seq[:, :, N - i - 1] = A + B @ K_seq[:, :, N - i - 1]
        P_seq[:, :, N - i - 1] = Q + 1 / proximal_lambda * np.eye(n_x) \
                                 + K_seq[:, :, N - i - 1].T @ (R + np.eye(n_u)) @ K_seq[:, :, N - i - 1] \
                                 + A_bar_seq[:, :, N - i - 1].T @ P_seq[:, :, N - i - 1] @ A_bar_seq[:, :, N - i - 1]

    return P_seq, R_tilde_seq, K_seq, A_bar_seq


class ProximalOfflinePart:
    def __init__(self, prediction_horizon, state_dynamics, control_dynamics, stage_state_constraints,
                 stage_control_constraints, terminal_state_constraints):
        """
        :param prediction_horizon: prediction horizon (N) of dynamic system
        :param state_dynamics: matrix A, describing the state dynamics
        :param control_dynamics: matrix B, describing control dynamics
        :param stage_state_constraints: matrix Gamma_x, describing the state constraints
        :param stage_control_constraints: matrix Gamma_u, describing control constraints
        :param terminal_state_constraints: matrix Gamma_N, describing terminal constraints
        """
        self.__prediction_horizon = prediction_horizon
        self.__A = state_dynamics
        self.__B = control_dynamics
        self.__Gamma_x = stage_state_constraints
        self.__Gamma_u = stage_control_constraints
        self.__Gamma_N = terminal_state_constraints
        self.make_Phi()
        self.make_Phi_star()

    def make_Phi(self):
        """Construct LinearOperator Phi"""
        N = self.__prediction_horizon
        Gamma_x = self.__Gamma_x
        Gamma_u = self.__Gamma_u
        Gamma_N = self.__Gamma_N
        n_x = self.__A.shape[1]
        n_u = self.__B.shape[1]
        n_c = self.__Gamma_x.shape[0]
        n_f = self.__Gamma_N.shape[0]
        n_z = (N + 1) * n_x + N * n_u
        n_Phi = N * n_c + n_f

        def matvec(v):
            Phi = Gamma_x @ v[0:n_x] + Gamma_u @ v[n_x:n_x + n_u]
            Phi = np.reshape(Phi, (n_c, 1))
            for i in range(1, N):
                Phi_n = Gamma_x @ v[i * (n_x + n_u):i * (n_x + n_u) + n_x] + Gamma_u @ v[
                                                                                       i * (n_x + n_u) + n_x:(i + 1) * (
                                                                                               n_x + n_u)]
                Phi = np.vstack((Phi, np.reshape(Phi_n, (n_c, 1))))

            Phi_N = Gamma_N @ v[N * (n_x + n_u):N * (n_x + n_u) + n_x]
            Phi = np.vstack((Phi, np.reshape(Phi_N, (n_f, 1))))
            return Phi

        return lingalg.LinearOperator((n_Phi, n_z), matvec=matvec)

    def make_Phi_star(self):
        """Construct LinearOperator Phi_star"""
        N = self.__prediction_horizon
        Gamma_x = self.__Gamma_x
        Gamma_u = self.__Gamma_u
        Gamma_N = self.__Gamma_N
        n_x = self.__A.shape[1]
        n_u = self.__B.shape[1]
        n_c = self.__Gamma_x.shape[0]
        n_f = self.__Gamma_N.shape[0]
        n_z = (N + 1) * n_x + N * n_u
        n_Phi = N * n_c + n_f

        def matvec(v):
            Phi_star_x = np.reshape(Gamma_x.T @ v[0:n_c], (n_x, 1))
            Phi_star_u = np.reshape(Gamma_u.T @ v[0:n_c], (n_u, 1))
            Phi_star = np.vstack((Phi_star_x, Phi_star_u))

            for i in range(1, N):
                phi_n_x = np.reshape(Gamma_x.T @ v[(i+1) * n_c:(i+2) * n_c], (n_x, 1))
                phi_n_u = np.reshape(Gamma_u.T @ v[(i+1) * n_c:(i+2) * n_c], (n_u, 1))
                Phi_star = np.vstack((Phi_star, phi_n_x, phi_n_u))

            Phi_star_N = np.reshape(Gamma_N.T @ v[N * n_c:(N+1) * n_c], (n_x, 1))
            Phi_star = np.vstack((Phi_star, Phi_star_N))

            return Phi_star

        return lingalg.LinearOperator((n_z, n_Phi), matvec=matvec)
