import unittest
import numpy as np
import cpasocp.core.linear_operators as core_lin_op


class TestLinearOperators(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_linearoperators(self):
        tol = 1e-10

        # define dynamic system parameters
        prediction_horizon = 10

        A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
        B = np.array([[1, 1], [0.5, 1]])  # n x u matrices

        N = prediction_horizon
        n_x = A.shape[1]
        n_u = B.shape[1]

        Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
        Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
        Gamma_N = np.eye(n_x)

        n_c = Gamma_x.shape[0]
        n_f = Gamma_N.shape[0]
        n_z = (N + 1) * n_x + N * n_u
        n_y = N * n_c + n_f
        # create random samples
        num_samples = 100
        multiplier = 10

        # create LinearOperators
        L = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_op()
        L_adj = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_adj()

        # test LinearOperators
        for i in range(num_samples):
            z = np.array(multiplier * np.random.rand(n_z)).reshape((n_z, 1))
            y = np.array(multiplier * np.random.rand(n_y)).reshape((n_y, 1))
            inner_1 = np.inner(y.T, (L @ z).T)[0, 0]
            inner_2 = np.inner((L_adj @ y).T, z.T)[0, 0]
            self.assertAlmostEqual(inner_1, inner_2, delta=tol)


if __name__ == '__main__':
    unittest.main()
