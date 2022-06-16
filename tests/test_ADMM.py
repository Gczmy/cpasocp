import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import cpasocp.core.sets as core_sets
import time


class TestADMM(unittest.TestCase):
    prediction_horizon = 10
    n_x = 4
    n_u = 3

    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    constraints_type = 'Rectangle'
    rect_min = [-1] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rect_max = [1] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    for i in range(n_x + n_u):
        rect_min[i] = np.random.random()
        rect_max[i] = np.random.random()
        if (i % 2) == 0:
            rect_min[i] = rect_min[i] * -10000 - 10000
            rect_max[i] = rect_max[i] * 10000 + 10000
        else:
            rect_min[i] = rect_min[i] * -10 - 2
            rect_max[i] = rect_max[i] * 10 + 2
    rectangle = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)

    # algorithm parameters
    epsilon = 1e-7
    # initial_state = np.array([0.2, 0.5])
    initial_state = 0.5 * np.random.rand(n_x) + 0.1
    for i in range(n_x):
        initial_state[i] = np.random.random()
        if (i % 2) == 0:
            initial_state[i] = initial_state[i] * 5 + 5
        else:
            initial_state[i] = initial_state[i]

    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1) + 0.1
    eta0 = 0.5 * np.random.rand(n_z, 1) + 0.1

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

    def test_ADMM(self):
        tol = 1e-3
        # ADMM
        # --------------------------------------------------------------------------------------------------------------
        start_ADMM = time.time()
        solution_ADMM = cpa.core.CPASOCP(TestADMM.prediction_horizon) \
            .with_dynamics(TestADMM.A, TestADMM.B) \
            .with_cost(TestADMM.cost_type, TestADMM.Q, TestADMM.R, TestADMM.P) \
            .with_constraints(TestADMM.constraints_type, TestADMM.stage_sets, TestADMM.terminal_set) \
            .ADMM(TestADMM.epsilon, TestADMM.initial_state, TestADMM.z0, TestADMM.eta0)
        ADMM_time = time.time() - start_ADMM
        z_ADMM = solution_ADMM.get_z_value

        error_ADMM = np.linalg.norm(z_ADMM - TestADMM.z_cvxpy, np.inf)
        self.assertAlmostEqual(error_ADMM, 0, delta=tol)


if __name__ == '__main__':
    unittest.main()
