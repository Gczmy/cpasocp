import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import matplotlib.pyplot as plt


class TestLBFGS(unittest.TestCase):
    prediction_horizon = 5

    n_x = 1000
    n_u = 2
    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    cost_type = "Quadratic"
    # Q = 0.1 * np.eye(n_x)  # n x n matrix
    # q = np.zeros((n_x, 1))

    C = 0.1 * np.random.randn(n_x, n_x)
    Q = np.eye(n_x) + C.T @ C
    q = np.random.randn(n_x).reshape(-1, 1)

    # algorithm parameters
    epsilon = 1e-3
    # initial_state = np.array([0.2, 0.5])
    initial_state = np.random.randn(n_x).reshape(-1, 1)

    # ----
    # Solve in cvxpy
    x = cp.Variable(n_x)
    cost = 0.5 * cp.quad_form(x, Q) + q.T @ x
    cp_prob = cp.Problem(cp.Minimize(cost))
    cp_prob.solve()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_l_bfgs(self):
        tol = 1e-3
        m = 2
        epsilon = 1e-3

        # L-BFGS
        # --------------------------------------------------------------------------------------------------------------
        solution = cpa.core.CPASOCP(TestLBFGS.prediction_horizon) \
            .with_dynamics(TestLBFGS.A, TestLBFGS.B) \
            .with_cost(TestLBFGS.cost_type, TestLBFGS.Q, stage_state_weight2=TestLBFGS.q) \
            .L_BFGS(epsilon, TestLBFGS.initial_state, m)

        x = solution.get_z_value
        if not np.allclose(x.T, TestLBFGS.x.value, atol=tol):
            raise Exception("solutions not close")
        cache_grad = solution.get_L_BFGS_grad_cache
        plt.semilogy(cache_grad)
        plt.show()


if __name__ == '__main__':
    unittest.main()
