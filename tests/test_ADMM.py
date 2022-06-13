import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.proximal_online_part as core_online
import cpasocp.core.chambolle_pock_algorithm as core_cpa
import cpasocp.core.linear_operators as core_lin_op
import cpasocp.core.sets as core_sets
import time


class TestADMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_ADMM(self):
        tol = 1e-4

        prediction_horizon = 10
        n_x = 10
        n_u = 10

        # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
        A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
        # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
        B = np.random.rand(n_x, n_u)  # n x u matrices

        cost_type = "Quadratic"
        Q = 10 * np.eye(n_x)  # n x n matrix
        R = np.eye(n_u)  # u x u matrix OR scalar
        P = 5 * np.eye(n_x)  # n x n matrix

        constraints_type = 'Rectangle'
        rectangle = core_sets.Rectangle(rect_min=-2, rect_max=2)
        stage_sets_list = [rectangle] * prediction_horizon
        stage_sets = core_sets.Cartesian(stage_sets_list)
        terminal_set = core_sets.Rectangle(rect_min=-2, rect_max=2)

        # algorithm parameters
        epsilon = 1e-4
        # initial_state = np.array([0.2, 0.5])
        initial_state = 0.5 * np.random.rand(n_x)
        n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
        z0 = 0.5 * np.random.rand(n_z, 1)
        eta0 = 0.5 * np.random.rand(n_z, 1)
        start_CP = time.time()
        solution = cpa.core.CPASOCP(prediction_horizon) \
            .with_dynamics(A, B) \
            .with_cost(cost_type, Q, R, P) \
            .with_constraints(constraints_type, stage_sets, terminal_set) \
            .chambolle_pock_algorithm(epsilon, initial_state, z0, eta0)
        z_CP = solution.get_z_value
        CP_time = time.time() - start_CP
        print("CP_time:", CP_time)

        # solving OCP by cvxpy
        # -----------------------------
        N = prediction_horizon
        n_x = A.shape[1]
        n_u = B.shape[1]
        c_t_x_min = - 2 * np.ones(n_x)
        c_t_x_max = 2 * np.ones(n_x)
        c_t_u_min = - 2 * np.ones(n_u)
        c_t_u_max = 2 * np.ones(n_u)
        c_N_min = - 2 * np.ones(n_x)
        c_N_max = 2 * np.ones(n_x)

        # Problem statement
        x0_cp = cp.Parameter(n_x)  # <--- x is a parameter of the optimisation problem P_N(x)
        u_seq = cp.Variable((n_u, N))  # <--- sequence of control actions
        x_seq = cp.Variable((n_x, N + 1))

        cost = 0
        constraints = [x_seq[:, 0] == x0_cp]  # Initial Condition x_0 = x

        for t in range(N):
            xt_var = x_seq[:, t]  # x_t
            ut_var = u_seq[:, t]  # u_t
            chi_t = np.reshape(z0[t * (n_x + n_u): t * (n_x + n_u) + n_x], n_x)
            v_t = np.reshape(z0[t * (n_x + n_u) + n_x: (t + 1) * (n_x + n_u)], n_u)
            cost += 0.5 * (cp.quad_form(xt_var, Q) + cp.quad_form(ut_var, R)
                           # + 1 / alpha
                           # * (cp.sum_squares(xt_var - chi_t) + cp.sum_squares(ut_var - v_t))
                           )  # Stage cost
            constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var,  # Dynamics
                            c_t_x_min <= xt_var,
                            xt_var <= c_t_x_max,
                            c_t_u_min <= ut_var,
                            ut_var <= c_t_u_max]  # Input Constraints

        xN = x_seq[:, N]
        chi_N = np.reshape(z0[N * (n_x + n_u): N * (n_x + n_u) + n_x], n_x)
        cost += 0.5 * (cp.quad_form(xN, P)
                       # + 1 / alpha * cp.sum_squares(xN - chi_N)
                       )  # Terminal cost
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

        error_CP = np.linalg.norm(z_CP - z_cvxpy, np.inf)
        print("error_CP:", error_CP)

        # ADMM method
        alpha = 1.02
        Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
        Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
        Gamma_N = np.eye(n_x)

        L = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_op()
        L_adj = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_adj()
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            prediction_horizon, alpha, A, B, Q, R, P).algorithm()

        proximal_lambda = alpha
        rho = 1 / proximal_lambda
        z_next = z0
        eta_next = eta0
        u0 = 1 / rho * L_adj @ eta0
        u_next = u0
        n_max = 10000

        start_ADMM = time.time()

        for i in range(n_max):
            z_prev = z_next
            eta_prev = eta_next
            u_prev = u_next
            z_next = core_online.proximal_of_h_online_part(prediction_horizon=prediction_horizon,
                                                           proximal_lambda=proximal_lambda,
                                                           initial_state=initial_state,
                                                           initial_guess_vector=L_adj @ eta_prev - u_prev,
                                                           state_dynamics=A,
                                                           control_dynamics=B,
                                                           control_weight=R,
                                                           P_seq=P_seq,
                                                           R_tilde_seq=R_tilde_seq,
                                                           K_seq=K_seq,
                                                           A_bar_seq=A_bar_seq)
            eta_next = core_cpa.proj_to_c(
                z_next + u_prev, prediction_horizon, Gamma_x, Gamma_N, stage_sets, terminal_set)
            u_next = u_prev + z_next - L_adj @ eta_next

            s = rho * L_adj @ (eta_next - eta_prev)
            r = z_next - L_adj @ eta_next
            t_1 = np.linalg.norm(s)
            t_2 = np.linalg.norm(r)
            epsilon_pri = np.sqrt(n_z) * epsilon + epsilon * max(np.linalg.norm(z_prev),
                                                                 np.linalg.norm(L_adj @ eta_prev))
            epsilon_dual = np.sqrt(n_z) * epsilon + epsilon * np.linalg.norm(L_adj @ eta_prev)
            # if t_2 <= epsilon_pri and t_1 <= epsilon_dual:
            #     break
            if np.linalg.norm(z_next - z_cvxpy, np.inf) <= error_CP:
                break
        z_ADMM = z_next
        ADMM_time = time.time() - start_ADMM
        print("ADMM loop:", i)
        print("ADMM time:", ADMM_time)

        error_ADMM = np.linalg.norm(z_ADMM - z_cvxpy, np.inf)
        print("error_ADMM:", error_ADMM)
        self.assertAlmostEqual(error_ADMM, 0, delta=tol)

    def test_ADMM_py(self):
        tol = 1e-4

        prediction_horizon = 10
        n_x = 10
        n_u = 10

        # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
        A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
        # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
        B = np.random.rand(n_x, n_u)  # n x u matrices

        cost_type = "Quadratic"
        Q = 10 * np.eye(n_x)  # n x n matrix
        R = np.eye(n_u)  # u x u matrix OR scalar
        P = 5 * np.eye(n_x)  # n x n matrix

        constraints_type = 'Rectangle'
        rectangle = core_sets.Rectangle(rect_min=-2, rect_max=2)
        stage_sets_list = [rectangle] * prediction_horizon
        stage_sets = core_sets.Cartesian(stage_sets_list)
        terminal_set = core_sets.Rectangle(rect_min=-2, rect_max=2)

        # algorithm parameters
        epsilon = 1e-4
        # initial_state = np.array([0.2, 0.5])
        initial_state = 0.5 * np.random.rand(n_x)
        n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
        z0 = 0.5 * np.random.rand(n_z, 1)
        eta0 = 0.5 * np.random.rand(n_z, 1)
        start_CP = time.time()
        solution = cpa.core.CPASOCP(prediction_horizon) \
            .with_dynamics(A, B) \
            .with_cost(cost_type, Q, R, P) \
            .with_constraints(constraints_type, stage_sets, terminal_set) \
            .chambolle_pock_algorithm(epsilon, initial_state, z0, eta0)
        z_CP = solution.get_z_value
        CP_time = time.time() - start_CP
        print("CP_time:", CP_time)

        # solving OCP by cvxpy
        # -----------------------------
        N = prediction_horizon
        n_x = A.shape[1]
        n_u = B.shape[1]
        c_t_x_min = - 2 * np.ones(n_x)
        c_t_x_max = 2 * np.ones(n_x)
        c_t_u_min = - 2 * np.ones(n_u)
        c_t_u_max = 2 * np.ones(n_u)
        c_N_min = - 2 * np.ones(n_x)
        c_N_max = 2 * np.ones(n_x)

        # Problem statement
        x0_cp = cp.Parameter(n_x)  # <--- x is a parameter of the optimisation problem P_N(x)
        u_seq = cp.Variable((n_u, N))  # <--- sequence of control actions
        x_seq = cp.Variable((n_x, N + 1))

        cost = 0
        constraints = [x_seq[:, 0] == x0_cp]  # Initial Condition x_0 = x

        for t in range(N):
            xt_var = x_seq[:, t]  # x_t
            ut_var = u_seq[:, t]  # u_t
            chi_t = np.reshape(z0[t * (n_x + n_u): t * (n_x + n_u) + n_x], n_x)
            v_t = np.reshape(z0[t * (n_x + n_u) + n_x: (t + 1) * (n_x + n_u)], n_u)
            cost += 0.5 * (cp.quad_form(xt_var, Q) + cp.quad_form(ut_var, R)
                           # + 1 / alpha
                           # * (cp.sum_squares(xt_var - chi_t) + cp.sum_squares(ut_var - v_t))
                           )  # Stage cost
            constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var,  # Dynamics
                            c_t_x_min <= xt_var,
                            xt_var <= c_t_x_max,
                            c_t_u_min <= ut_var,
                            ut_var <= c_t_u_max]  # Input Constraints

        xN = x_seq[:, N]
        chi_N = np.reshape(z0[N * (n_x + n_u): N * (n_x + n_u) + n_x], n_x)
        cost += 0.5 * (cp.quad_form(xN, P)
                       # + 1 / alpha * cp.sum_squares(xN - chi_N)
                       )  # Terminal cost
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

        error_CP = np.linalg.norm(z_CP - z_cvxpy, np.inf)
        print("error_CP:", error_CP)

        start_ADMM = time.time()
        solution_ADMM = cpa.core.CPASOCP(prediction_horizon) \
            .with_dynamics(A, B) \
            .with_cost(cost_type, Q, R, P) \
            .with_constraints(constraints_type, stage_sets, terminal_set) \
            .ADMM(z_cvxpy, z_CP, initial_state, z0, eta0)
        z_ADMM = solution_ADMM.get_z_ADMM_value
        ADMM_time = time.time() - start_ADMM
        print("ADMM time:", ADMM_time)

        error_ADMM = np.linalg.norm(z_ADMM - z_cvxpy, np.inf)
        print("error_ADMM:", error_ADMM)
        self.assertAlmostEqual(error_ADMM, 0, delta=tol)


if __name__ == '__main__':
    unittest.main()
