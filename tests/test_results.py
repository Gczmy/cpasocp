import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import cpasocp.core.sets as core_sets


class TestResults(unittest.TestCase):
    prediction_horizon = np.random.randint(15, 20)

    n_x = np.random.randint(10, 20)  # state dimension
    n_u = np.random.randint(9, n_x)  # input dimension

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
        rect_min[i] = np.random.rand()
        rect_max[i] = np.random.rand()
        if (i % 2) == 0:
            rect_min[i] = rect_min[i] * -100000 - 100000
            rect_max[i] = rect_max[i] * 100000 + 100000
        else:
            rect_min[i] = rect_min[i] * -1000 - 1000
            rect_max[i] = rect_max[i] * 1000 + 1000
    rectangle = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)

    # algorithm parameters
    epsilon = 1e-4
    # initial_state = np.array([0.2, 0.5])
    initial_state = 0.5 * np.random.rand(n_x) + 0.1
    for i in range(n_x):
        initial_state[i] = np.random.rand() - 0.5
        if (i % 2) == 0:
            initial_state[i] *= 1000
        else:
            initial_state[i] *= 100

    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1) + 0.1
    eta0 = 0.5 * np.random.rand(n_z, 1) + 0.1
    for j in range(prediction_horizon):
        for i in range(n_x + n_u):
            z0[j * (n_x + n_u) + i] = np.random.random()
            eta0[j * (n_x + n_u) + i] = np.random.random()
            if ((j * (n_x + n_u) + i) % 2) == 0:
                z0[j * (n_x + n_u) + i] *= 1000
                eta0[j * (n_x + n_u) + i] *= 1000
            else:
                z0[j * (n_x + n_u) + i] *= 100
                eta0[j * (n_x + n_u) + i] *= 100

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
    f = 0
    gradient_f = 0
    for i in range(N):
        x_t = z_cvxpy[i * (n_x + n_u): i * (n_x + n_u) + n_x]
        u_t = z_cvxpy[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
        f += 0.5 * (x_t.T @ Q @ x_t + u_t.T @ R @ u_t)
        gradient_f += Q @ x_t
    x_N = z_cvxpy[N * (n_x + n_u): N * (n_x + n_u) + n_x]
    f += 0.5 * x_N.T @ Q @ x_N
    f = f[0, 0]
    gradient_f += P @ x_N
    print(f)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_chambolle_pock_results(self):
        tol = 1e-3
        # Chambolle-Pock method
        # --------------------------------------------------------------------------------------------------------------
        solution = cpa.core.CPASOCP(TestResults.prediction_horizon) \
            .with_dynamics(TestResults.A, TestResults.B) \
            .with_cost(TestResults.cost_type, TestResults.Q, TestResults.R, TestResults.P) \
            .with_constraints(TestResults.constraints_type, TestResults.stage_sets, TestResults.terminal_set) \
            .chambolle_pock_algorithm(TestResults.epsilon, TestResults.initial_state, TestResults.z0, TestResults.eta0)

        z_chambolle_pock = solution.get_z_value
        N = TestResults.prediction_horizon
        n_x = TestResults.A.shape[1]
        n_u = TestResults.B.shape[1]
        error_CP = np.linalg.norm(z_chambolle_pock - TestResults.z_cvxpy, np.inf)
        print('error_CP:', error_CP)
        self.assertAlmostEqual(error_CP, 0, delta=tol)
        f = 0
        gradient_f = 0
        for i in range(N):
            x_t = z_chambolle_pock[i * (n_x + n_u): i * (n_x + n_u) + n_x]
            u_t = z_chambolle_pock[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
            f += 0.5 * (x_t.T @ TestResults.Q @ x_t + u_t.T @ TestResults.R @ u_t)
            gradient_f += TestResults.Q @ x_t
        x_N = z_chambolle_pock[N * (n_x + n_u): N * (n_x + n_u) + n_x]
        f += 0.5 * x_N.T @ TestResults.Q @ x_N
        f = f[0, 0]
        gradient_f += TestResults.P @ x_N
        print(f)
        # print(gradient_f)

    def test_ADMM_results(self):
        tol = 1e-3
        # ADMM
        # --------------------------------------------------------------------------------------------------------------
        solution_ADMM = cpa.core.CPASOCP(TestResults.prediction_horizon) \
            .with_dynamics(TestResults.A, TestResults.B) \
            .with_cost(TestResults.cost_type, TestResults.Q, TestResults.R, TestResults.P) \
            .with_constraints(TestResults.constraints_type, TestResults.stage_sets, TestResults.terminal_set) \
            .ADMM(TestResults.epsilon, TestResults.initial_state, TestResults.z0, TestResults.eta0)
        z_ADMM = solution_ADMM.get_z_value

        N = TestResults.prediction_horizon
        n_x = TestResults.A.shape[1]
        n_u = TestResults.B.shape[1]
        error_ADMM = np.linalg.norm(z_ADMM - TestResults.z_cvxpy, np.inf)
        print('error_ADMM:', error_ADMM)
        self.assertAlmostEqual(error_ADMM, 0, delta=tol)
        f = 0
        gradient_f = 0
        for i in range(N):
            x_t = z_ADMM[i * (n_x + n_u): i * (n_x + n_u) + n_x]
            u_t = z_ADMM[i * (n_x + n_u) + n_x: (i + 1) * (n_x + n_u)]
            f += 0.5 * (x_t.T @ TestResults.Q @ x_t + u_t.T @ TestResults.R @ u_t)
            gradient_f += TestResults.Q @ x_t
        x_N = z_ADMM[N * (n_x + n_u): N * (n_x + n_u) + n_x]
        f += 0.5 * x_N.T @ TestResults.Q @ x_N
        f = f[0, 0]
        gradient_f += TestResults.P @ x_N
        print(f)
        # print(gradient_f)
        # distance = np.inner(-gradient_f, )


if __name__ == '__main__':
    unittest.main()
