import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import cpasocp.core.sets as core_sets
import time


class TestScaling(unittest.TestCase):
    # dynamics
    prediction_horizon = 10
    n_x = 20  # state dimension
    n_u = 10  # input dimension

    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    # costs
    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    # constraints
    constraints_type = 'Rectangle'
    rect_min = [0] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rect_max = [0] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    for i in range(n_x + n_u):
        rect_min[i] = np.random.rand()
        rect_max[i] = np.random.rand()
        if (i % 2) == 0:
            rect_min[i] = rect_min[i] * -100000 - 100000
            rect_max[i] = rect_max[i] * 100000 + 100000
        else:
            rect_min[i] = rect_min[i] * -100 - 100
            rect_max[i] = rect_max[i] * 100 + 100
    rectangle = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    # initial_state = np.array([1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5])
    initial_state = 0.5 * np.random.rand(n_x) + 0.1
    for i in range(n_x):
        initial_state[i] = np.random.rand() - 0.5
        if (i % 2) == 0:
            initial_state[i] = initial_state[i]
        else:
            initial_state[i] = initial_state[i]
    # algorithm parameters
    epsilon = 1e-7
    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1) + 0.1
    eta0 = 0.5 * np.random.rand(n_z, 1) + 0.1

    # solving OCP by cvxpy scaling
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
        c_t_x_min[i] = -0.5
        c_t_x_max[i] = 0.5
        c_N_min[i] = -0.5
        c_N_max[i] = 0.5
    for i in range(n_u):
        c_t_u_min[i] = -0.5
        c_t_u_max[i] = 0.5

    scaling_factor = np.zeros(n_z)
    for i in range(N):
        x_index = i * (n_x + n_u)
        u_index = i * (n_x + n_u) + n_x
        for j in range(n_x):
            scaling_factor[x_index + j] = rect_max[j] - rect_min[j]
        for j in range(n_u):
            scaling_factor[u_index + j] = rect_max[j+n_x] - rect_min[j+n_x]
    N_index = N * (n_x + n_u)
    for j in range(n_x):
        scaling_factor[N_index + j] = rect_max[j] - rect_min[j]
    scaling_factor = np.reshape(scaling_factor, (n_z, 1))
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
    x0_cp.value = initial_state.copy()
    for i in range(n_x):
        x0_cp.value[i] = x0_cp.value[i] / scaling_factor[i]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Construct z, z are all the states and inputs in a big vector
    z_cvxpy_scaling = np.reshape(x_seq.value[:, 0], (n_x, 1))  # x_0
    z_cvxpy_scaling = np.vstack((z_cvxpy_scaling, np.reshape(u_seq.value[:, 0], (n_u, 1))))  # u_0

    for i in range(1, N):
        z_cvxpy_scaling = np.vstack((z_cvxpy_scaling, np.reshape(x_seq.value[:, i], (n_x, 1))))
        z_cvxpy_scaling = np.vstack((z_cvxpy_scaling, np.reshape(u_seq.value[:, i], (n_u, 1))))

    z_cvxpy_scaling = np.vstack((z_cvxpy_scaling, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN
    z_cvxpy_scaling = z_cvxpy_scaling * scaling_factor
    f = 0
    gradient_f = 0
    for i in range(N):
        x_t = z_cvxpy_scaling[i * (n_x + n_u): i * (n_x + n_u) + n_x]
        u_t = z_cvxpy_scaling[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
        f += 0.5 * (x_t.T @ Q @ x_t + u_t.T @ R @ u_t)
        gradient_f += Q @ x_t
    x_N = z_cvxpy_scaling[N * (n_x + n_u): N * (n_x + n_u) + n_x]
    f += 0.5 * x_N.T @ Q @ x_N
    f = f[0, 0]
    gradient_f += P @ x_N
    print(f)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_CP_scaling(self):
        tol = 1e-3
        # CP scaling
        # ------------------------------------------------------------------------------------------------------------------
        start_scaling = time.time()
        solution_scaling = cpa.core.CPASOCP(TestScaling.prediction_horizon) \
            .with_dynamics(TestScaling.A, TestScaling.B) \
            .with_cost(TestScaling.cost_type, TestScaling.Q, TestScaling.R, TestScaling.P) \
            .with_constraints_scaling(TestScaling.constraints_type, TestScaling.stage_sets, TestScaling.terminal_set) \
            .cp_scaling(TestScaling.epsilon, TestScaling.initial_state, TestScaling.z0, TestScaling.eta0)
        scaling_time = time.time() - start_scaling
        z_scaling = solution_scaling.z
        print('scaling_time:', scaling_time)
        error_scaling_cvxpy = np.linalg.norm(z_scaling - TestScaling.z_cvxpy_scaling, np.inf)
        print('error_scaling_cvxpy:', error_scaling_cvxpy)
        self.assertAlmostEqual(error_scaling_cvxpy, 0, delta=tol)

        N = TestScaling.prediction_horizon
        n_x = TestScaling.A.shape[1]
        n_u = TestScaling.B.shape[1]
        f = 0
        gradient_f = 0
        for i in range(N):
            x_t = z_scaling[i * (n_x + n_u): i * (n_x + n_u) + n_x]
            u_t = z_scaling[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
            f += 0.5 * (x_t.T @ TestScaling.Q @ x_t + u_t.T @ TestScaling.R @ u_t)
            gradient_f += TestScaling.Q @ x_t
        x_N = z_scaling[N * (n_x + n_u): N * (n_x + n_u) + n_x]
        f += 0.5 * x_N.T @ TestScaling.Q @ x_N
        f = f[0, 0]
        gradient_f += TestScaling.P @ x_N
        print(f)

    def test_ADMM_scaling(self):
        tol = 1e-3
        # ADMM scaling
        # ------------------------------------------------------------------------------------------------------------------
        start_ADMM_scaling = time.time()
        solution_ADMM_scaling = cpa.core.CPASOCP(TestScaling.prediction_horizon) \
            .with_dynamics(TestScaling.A, TestScaling.B) \
            .with_cost(TestScaling.cost_type, TestScaling.Q, TestScaling.R, TestScaling.P) \
            .with_constraints_scaling(TestScaling.constraints_type, TestScaling.stage_sets, TestScaling.terminal_set) \
            .admm_scaling(TestScaling.epsilon, TestScaling.initial_state, TestScaling.z0, TestScaling.eta0)
        ADMM_scaling_time = time.time() - start_ADMM_scaling
        z_ADMM_scaling = solution_ADMM_scaling.z

        error_ADMM_scaling_cvxpy = np.linalg.norm(z_ADMM_scaling - TestScaling.z_cvxpy_scaling, np.inf)
        print('error_ADMM_scaling_cvxpy:', error_ADMM_scaling_cvxpy)
        self.assertAlmostEqual(error_ADMM_scaling_cvxpy, 0, delta=tol)

        N = TestScaling.prediction_horizon
        n_x = TestScaling.A.shape[1]
        n_u = TestScaling.B.shape[1]
        f = 0
        gradient_f = 0
        for i in range(N):
            x_t = z_ADMM_scaling[i * (n_x + n_u): i * (n_x + n_u) + n_x]
            u_t = z_ADMM_scaling[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
            f += 0.5 * (x_t.T @ TestScaling.Q @ x_t + u_t.T @ TestScaling.R @ u_t)
            gradient_f += TestScaling.Q @ x_t
        x_N = z_ADMM_scaling[N * (n_x + n_u): N * (n_x + n_u) + n_x]
        f += 0.5 * x_N.T @ TestScaling.Q @ x_N
        f = f[0, 0]
        gradient_f += TestScaling.P @ x_N
        print(f)


if __name__ == '__main__':
    unittest.main()
