import numpy as np
import scipy.sparse.linalg as lingalg


class ProximalOfflinePart:
    def __init__(self, prediction_horizon, proximal_lambda, state_dynamics, control_dynamics, stage_state_weight,
                 control_weight, terminal_state_weight):
        """
        :param prediction_horizon: prediction horizon (N) of dynamic system
        :param proximal_lambda: a parameter (lambda) for proximal operator
        :param state_dynamics: matrix A, describing the state dynamics
        :param control_dynamics: matrix B, describing control dynamics
        :param stage_state_weight: matrix (Q), stage state cost matrix
        :param control_weight: scalar or matrix (R), input cost matrix or scalar
        :param terminal_state_weight: matrix (P), terminal state cost matrix
        """
        self.__prediction_horizon = prediction_horizon
        self.__lambda = proximal_lambda
        self.__A = state_dynamics
        self.__B = control_dynamics
        self.__Q = stage_state_weight
        self.__R = control_weight
        self.__P = terminal_state_weight
        self.algorithm()

    def algorithm(self):
        """Construct the offline algorithm"""
        A = self.__A
        B = self.__B
        Q = self.__Q
        R = self.__R
        P = self.__P
        n_x = A.shape[1]
        n_u = B.shape[1]
        N = self.__prediction_horizon
        P_seq = np.zeros((n_x, n_x, N + 1))  # tensor
        R_tilde_Cholesky_seq = np.zeros((n_u, n_u, N))  # tensor
        K_seq = np.zeros((n_u, n_x, N))  # tensor
        A_bar_seq = np.zeros((n_x, n_x, N))  # tensor
        P_0 = P + 1 / self.__lambda * np.eye(n_x)
        P_seq[:, :, N] = P_0

        for i in range(N):
            R_tilde_seq = R + 1 / self.__lambda * np.eye(n_u) + B.T @ P_seq[:, :, N - i] @ B
            R_tilde_Cholesky_seq[:, :, N - i - 1] = np.linalg.cholesky(R_tilde_seq)
            y = np.linalg.solve(R_tilde_Cholesky_seq[:, :, N - i - 1], - B.T @ P_seq[:, :, N - i] @ A)
            K_seq[:, :, N - i - 1] = np.linalg.solve(R_tilde_Cholesky_seq[:, :, N - i - 1].T.conj(), y)
            A_bar_seq[:, :, N - i - 1] = A + B @ K_seq[:, :, N - i - 1]
            P_seq[:, :, N - i - 1] = Q + 1 / self.__lambda * np.eye(n_x) + K_seq[:, :, N - i - 1].T \
                                     @ (R + 1 / self.__lambda * np.eye(n_u)) @ K_seq[:, :, N - i - 1] \
                                     + A_bar_seq[:, :, N - i - 1].T @ P_seq[:, :, N - i] @ A_bar_seq[:, :, N - i - 1]

        return P_seq, R_tilde_Cholesky_seq, K_seq, A_bar_seq
