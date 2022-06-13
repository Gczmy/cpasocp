import unittest
import numpy as np
import cpasocp.core.precondition as core_pre
import cpasocp.core.linear_operators as core_lin_op
import cpasocp.core.proximal_online_part as core_online

class TestPrecondition(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_precondition(self):
        # define dynamic system parameters
        prediction_horizon = 10

        A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
        B = np.array([[1, 1], [0.5, 1]])  # n x u matrices

        N = prediction_horizon
        n_x = A.shape[1]
        n_u = B.shape[1]
        n_z = (N + 1) * n_x + N * n_u
        Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
        Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
        Gamma_N = np.eye(n_x)

        # create LinearOperators
        L = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_op()

        # L = np.array([[1, 2, 3, 0],
        #              [0, 2, 2, 0],
        #              [1, 2, 3, 4]])

        T, Sigma = core_pre.precondition(L @ np.identity(n_z), 1)
        print(T)
        print(Sigma)


if __name__ == '__main__':
    unittest.main()
