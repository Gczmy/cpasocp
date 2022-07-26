import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import cpasocp.core.sets as core_sets
import time
import matplotlib.pyplot as plt


class TestSuperMann(unittest.TestCase):
    prediction_horizon = 10
    n_x = 50  # state dimension
    n_u = 20  # input dimension

    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    constraints_type = 'Rectangle'
    rect_min = [-20] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rect_max = [20] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rectangle = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)

    # algorithm parameters
    epsilon = 1e-4
    initial_state = 0.5 * np.random.rand(n_x)

    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1)
    eta0 = 0.5 * np.random.rand(n_z, 1)

    # solving OCP by cvxpy
    # -----------------------------
    N = prediction_horizon
    n_x = A.shape[1]
    n_u = B.shape[1]
    c_t_x_min = np.zeros(n_x)
    c_t_x_max = np.zeros(n_x)
    c_t_u_min = np.zeros(n_u)
    c_t_u_max = np.zeros(n_u)
    c_N_min = np.zeros(n_x)
    c_N_max = np.zeros(n_x)
    for i in range(n_x):
        c_t_x_min[i] = rect_min[i]
        c_t_x_max[i] = rect_max[i]
        c_N_min[i] = rect_min[i]
        c_N_max[i] = rect_max[i]
    for i in range(n_u):
        c_t_u_min[i] = rect_min[i + n_x]
        c_t_u_max[i] = rect_max[i + n_x]

    # Problem statement
    x0_cp = cp.Parameter(n_x)  # <--- x is a parameter of the optimisation problem P_N(x)
    u_seq = cp.Variable((n_u, N))  # <--- sequence of control actions
    x_seq = cp.Variable((n_x, N + 1))

    cost = 0
    constraints = [x_seq[:, 0] == x0_cp]  # Initial Condition x_0 = x

    for t in range(N):
        xt_var = x_seq[:, t]  # x_t
        ut_var = u_seq[:, t]  # u_t
        cost += 0.5 * (cp.quad_form(xt_var, Q) + cp.quad_form(ut_var, R))  # Stage cost
        constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var,  # Dynamics
                        c_t_x_min <= xt_var,
                        xt_var <= c_t_x_max,
                        c_t_u_min <= ut_var,
                        ut_var <= c_t_u_max]  # Input Constraints

    xN = x_seq[:, N]
    cost += 0.5 * cp.quad_form(xN, P)  # Terminal cost
    constraints += [c_N_min <= xN, xN <= c_N_max]

    # Solution
    x0_cp.value = initial_state
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Construct z, z are all the states and inputs in a big vector
    z_cvxpy = np.reshape(x_seq.value[:, 0], (n_x, 1))  # x_0
    z_cvxpy = np.vstack((z_cvxpy, np.reshape(u_seq.value[:, 0], (n_u, 1))))  # u_0

    for i in range(1, N):
        z_cvxpy = np.vstack((z_cvxpy, np.reshape(x_seq.value[:, i], (n_x, 1))))
        z_cvxpy = np.vstack((z_cvxpy, np.reshape(u_seq.value[:, i], (n_u, 1))))

    z_cvxpy = np.vstack((z_cvxpy, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_supermann(self):
        tol = 1e-3
        # Chambolle-Pock method
        # --------------------------------------------------------------------------------------------------------------
        start_CP = time.time()
        solution_CP = cpa.core.CPASOCP(TestSuperMann.prediction_horizon) \
            .with_dynamics(TestSuperMann.A, TestSuperMann.B) \
            .with_cost(TestSuperMann.cost_type, TestSuperMann.Q, TestSuperMann.R, TestSuperMann.P) \
            .with_constraints(TestSuperMann.constraints_type, TestSuperMann.stage_sets, TestSuperMann.terminal_set) \
            .chambolle_pock(TestSuperMann.epsilon, TestSuperMann.initial_state, TestSuperMann.z0, TestSuperMann.eta0)

        CP_time = time.time() - start_CP
        z_CP = solution_CP.get_z_value
        print('CP_time:', CP_time)
        plt.figure(1)
        plt.title('CP semilogy')
        plt.xlabel('Iterations')
        plt.ylabel('Residuals')
        plt.semilogy(solution_CP.get_residuals_cache,
                     label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
        plt.legend()

        # Chambolle-Pock method with SuperMann
        # --------------------------------------------------------------------------------------------------------------
        c0 = 0.99
        c1 = 0.99
        q = 0.99
        beta = 0.5
        sigma = 0.1
        lambda_ = 1.95
        m = 3
        start_CP_SuperMann = time.time()
        solution_CP_SuperMann = cpa.core.CPASOCP(TestSuperMann.prediction_horizon) \
            .with_dynamics(TestSuperMann.A, TestSuperMann.B) \
            .with_cost(TestSuperMann.cost_type, TestSuperMann.Q, TestSuperMann.R, TestSuperMann.P) \
            .with_constraints(TestSuperMann.constraints_type, TestSuperMann.stage_sets, TestSuperMann.terminal_set) \
            .CP_SupperMann(TestSuperMann.epsilon, TestSuperMann.initial_state, TestSuperMann.z0, TestSuperMann.eta0, m, c0, c1, q, beta, sigma, lambda_)
        CP_SuperMann_time = time.time() - start_CP_SuperMann
        z_CP_SuperMann = solution_CP_SuperMann.get_z_value
        error_CP_SuperMann = np.linalg.norm(z_CP_SuperMann - TestSuperMann.z_cvxpy, np.inf)
        self.assertAlmostEqual(error_CP_SuperMann, 0, delta=tol)
        print('CP_SuperMann_time:', CP_SuperMann_time)

        plt.figure(2)
        plt.title('CP_SuperMann semilogy')
        plt.xlabel('Iterations')
        plt.ylabel('Residuals')
        plt.semilogy(solution_CP_SuperMann.get_residuals_cache, label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
