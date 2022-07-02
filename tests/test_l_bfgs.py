import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import time


class TestLBFGS(unittest.TestCase):
    prediction_horizon = 1

    n_x = 100
    n_u = 2
    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    B = np.random.rand(n_x, n_u)  # n x n matrices

    cost_type = "Quadratic"
    # Q = 0.1 * np.eye(n_x)  # n x n matrix
    # q = np.zeros((n_x, 1))

    C = 0.1 * np.random.randn(n_x, n_x)
    Q = np.eye(n_x) + C.T @ C
    R = np.zeros((n_u, n_u))
    P = np.zeros((n_x, n_x))
    q = np.random.randn(n_x).reshape(-1, 1)

    # algorithm parameters
    epsilon = 1e-3
    # initial_state = np.array([0.2, 0.5])
    initial_state = np.random.randn(n_x).reshape(-1, 1)
    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1) + 0.1
    eta0 = 0.5 * np.random.rand(n_z, 1) + 0.1

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
        tol = 1e-5
        m = 3
        epsilon = 1e-5

        # L-BFGS
        # --------------------------------------------------------------------------------------------------------------
        start_L_BFGS = time.time()
        solution = cpa.core.CPASOCP(TestLBFGS.prediction_horizon) \
            .with_dynamics(TestLBFGS.A, TestLBFGS.B) \
            .with_cost(TestLBFGS.cost_type, TestLBFGS.Q, TestLBFGS.R, TestLBFGS.P, TestLBFGS.q) \
            .L_BFGS(epsilon, TestLBFGS.initial_state, m)
        L_BFGS_time = time.time() - start_L_BFGS
        print(L_BFGS_time)
        x_L_BFGS = solution.get_z_value
        print(x_L_BFGS)
        if not np.allclose(x_L_BFGS.T, TestLBFGS.x.value, atol=tol):
            raise Exception("solutions not close")
        # Chambolle-Pock method
        # --------------------------------------------------------------------------------------------------------------
        start_CP = time.time()
        solution = cpa.core.CPASOCP(TestLBFGS.prediction_horizon) \
            .with_dynamics(TestLBFGS.A, TestLBFGS.B) \
            .with_cost(TestLBFGS.cost_type, TestLBFGS.Q, TestLBFGS.R, TestLBFGS.P, TestLBFGS.q) \
            .with_constraints() \
            .chambolle_pock_algorithm(TestLBFGS.epsilon, TestLBFGS.initial_state, TestLBFGS.z0,
                                      TestLBFGS.eta0)
        CP_time = time.time() - start_CP
        print(CP_time)
        z_chambolle_pock = solution.get_z_value
        print(z_chambolle_pock)
        # error_CP = np.linalg.norm(z_chambolle_pock - TestLBFGS.x.value, np.inf)
        # print('error_CP:', error_CP)
        # self.assertAlmostEqual(error_CP, 0, delta=tol)

        # cache_grad = solution.get_L_BFGS_grad_cache
        # plt.semilogy(cache_grad)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
