import unittest
import numpy as np
import cvxpy as cp
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.proximal_online_part as core_online


class TestOnlinePart(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_online_part(self):
        tol = 1e-10

        prediction_horizon = 20
        proximal_lambda = 0.1
        n_x = 10
        n_u = 5

        A = np.array(np.random.rand(n_x, n_x))  # n x n matrices
        B = np.array(np.random.rand(n_x, n_u))  # n x u matrices

        Q = 10 * np.eye(n_x)  # n x n matrix
        R = np.eye(n_u)  # u x u matrix OR scalar
        P = 5 * np.eye(n_x)  # n x n matrix

        initial_state = 10 * np.array(np.random.rand(n_x))
        n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
        z0 = np.array(np.random.rand(n_z, 1))

        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(prediction_horizon,
                                                                                proximal_lambda, A,
                                                                                B,
                                                                                Q, R, P).algorithm()
        # solving OCP by cvxpy
        # -----------------------------
        N = prediction_horizon
        n_x = A.shape[1]
        n_u = B.shape[1]
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
            cost += 0.5 * (cp.quad_form(xt_var, Q) + cp.quad_form(ut_var, R) + 1 / proximal_lambda
                           * (cp.sum_squares(xt_var - chi_t) + cp.sum_squares(ut_var - v_t)))  # Stage cost
            constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var]  # Input Constraints

        xN = x_seq[:, N]
        chi_N = np.reshape(z0[N * (n_x + n_u): N * (n_x + n_u) + n_x], n_x)
        cost += 0.5 * (cp.quad_form(xN, P) + 1 / proximal_lambda * cp.sum_squares(xN - chi_N))  # Terminal cost

        # Solution
        x0_cp.value = initial_state
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Construct z, z are all the states and inputs in a big vector
        z_cp = np.reshape(x_seq.value[:, 0], (n_x, 1))  # x_0
        z_cp = np.vstack((z_cp, np.reshape(u_seq.value[:, 0], (n_u, 1))))  # u_0

        for i in range(1, N):
            z_cp = np.vstack((z_cp, np.reshape(x_seq.value[:, i], (n_x, 1))))
            z_cp = np.vstack((z_cp, np.reshape(u_seq.value[:, i], (n_u, 1))))

        z_cp = np.vstack((z_cp, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN

        # solving OCP by proximal
        z_online_part = core_online.proximal_of_h_online_part(prediction_horizon=prediction_horizon,
                                                              proximal_lambda=proximal_lambda,
                                                              initial_state=initial_state,
                                                              initial_guess_vector=z0,
                                                              state_dynamics=A,
                                                              control_dynamics=B,
                                                              control_weight=R,
                                                              P_seq=P_seq,
                                                              R_tilde_seq=R_tilde_seq,
                                                              K_seq=K_seq,
                                                              A_bar_seq=A_bar_seq)
        error = np.linalg.norm(z_cp - z_online_part, np.inf)
        self.assertAlmostEqual(error, 0, delta=tol)


if __name__ == '__main__':
    unittest.main()
