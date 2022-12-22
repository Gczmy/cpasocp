import numpy as np
import scipy as sp


class ProximalOfflinePart:
    def __init__(self):
        self.__prediction_horizon = None
        self.__lambda = None  # proximal_lambda
        self.__A = None  # state_dynamics
        self.__B = None  # control_dynamics
        self.__Q = None  # stage_state_weight
        self.__R = None  # control_weight
        self.__P = None  # terminal_state_weight
        self.__P_seq = None
        self.__R_tilde_seq = None
        self.__K_seq = None
        self.__A_bar_seq = None

    @property
    def prediction_horizon(self):
        """
        :return: prediction_horizon: prediction horizon (N) of dynamic system
        """
        return self.__prediction_horizon

    @prediction_horizon.setter
    def prediction_horizon(self, value):
        self.__prediction_horizon = value

    @property
    def proximal_lambda(self):
        """
        :return: proximal_lambda: a parameter (lambda) for proximal operator
        """
        return self.__lambda

    @proximal_lambda.setter
    def proximal_lambda(self, value):
        self.__lambda = value

    @property
    def state_dynamics(self):
        """
        :return: state_dynamics: matrix A, describing the state dynamics
        """
        return self.__A

    @state_dynamics.setter
    def state_dynamics(self, value):
        self.__A = value

    @property
    def control_dynamics(self):
        """
        :return: control_dynamics: matrix B, describing control dynamics
        """
        return self.__B

    @control_dynamics.setter
    def control_dynamics(self, value):
        self.__B = value

    @property
    def stage_state_weight(self):
        """
        :return: stage_state_weight: matrix (Q), stage state cost matrix
        """
        return self.__Q

    @stage_state_weight.setter
    def stage_state_weight(self, value):
        self.__Q = value

    @property
    def control_weight(self):
        """
        :return: control_weight: scalar or matrix (R), input cost matrix or scalar
        """
        return self.__R

    @control_weight.setter
    def control_weight(self, value):
        self.__R = value

    @property
    def terminal_state_weight(self):
        """
        :return: terminal_state_weight: matrix (P), terminal state cost matrix
        """
        return self.__P

    @terminal_state_weight.setter
    def terminal_state_weight(self, value):
        self.__P = value

    @property
    def P_seq(self):
        """
        :return: P_seq: tensor, matrix sequence of (P) from proximal of h offline part
        """
        return self.__P_seq

    @P_seq.setter
    def P_seq(self, value):
        self.__P_seq = value

    @property
    def R_tilde_seq(self):
        """
        :return: R_tilde_seq: tensor, matrix sequence of (R) from proximal of h offline part
        """
        return self.__R_tilde_seq

    @R_tilde_seq.setter
    def R_tilde_seq(self, value):
        self.__R_tilde_seq = value

    @property
    def K_seq(self):
        """
        :return: K_seq: tensor, matrix sequence of (K) from proximal of h offline part
        """
        return self.__K_seq

    @K_seq.setter
    def K_seq(self, value):
        self.__K_seq = value

    @property
    def A_bar_seq(self):
        """
        :return: A_bar_seq: tensor, matrix sequence of (A_bar) from proximal of h offline part
        """
        return self.__A_bar_seq

    @A_bar_seq.setter
    def A_bar_seq(self, value):
        self.__A_bar_seq = value

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
        R_tilde_seq = np.zeros((n_u, n_u, N))  # tensor
        K_seq = np.zeros((n_u, n_x, N))  # tensor
        A_bar_seq = np.zeros((n_x, n_x, N))  # tensor
        P_0 = P + 1 / self.__lambda * np.eye(n_x)
        P_seq[:, :, N] = P_0

        for i in range(N):
            R_tilde_seq[:, :, N - i - 1] = R + 1 / self.__lambda * np.eye(n_u) + B.T @ P_seq[:, :, N - i] @ B
            c, low = sp.linalg.cho_factor(R_tilde_seq[:, :, N - i - 1])
            K_seq[:, :, N - i - 1] = sp.linalg.cho_solve((c, low), - B.T @ P_seq[:, :, N - i] @ A)
            A_bar_seq[:, :, N - i - 1] = A + B @ K_seq[:, :, N - i - 1]
            P_seq[:, :, N - i - 1] = Q + 1 / self.__lambda * np.eye(n_x) + K_seq[:, :, N - i - 1].T \
                                     @ (R + 1 / self.__lambda * np.eye(n_u)) @ K_seq[:, :, N - i - 1] \
                                     + A_bar_seq[:, :, N - i - 1].T @ P_seq[:, :, N - i] @ A_bar_seq[:, :, N - i - 1]
        self.__P_seq = P_seq
        self.__R_tilde_seq = R_tilde_seq
        self.__K_seq = K_seq
        self.__A_bar_seq = A_bar_seq
        return self
