import unittest
import numpy as np
import scipy as sp
import cvxpy as cp
import cpasocp.core.linear_operators as core_lin_op
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.proximal_online_part as core_online


class TestOnlinePart(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_online_part(self):
        tol = 1e-10

        prediction_horizon = 1
        proximal_lambda = 0.1
        n_x = 2
        n_u = 2
        n_c = 2
        n_f = 2
        A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
        # A = np.array(np.random.rand(n_x, n_x))  # n x n matrices
        B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
        # B = np.array(np.random.rand(n_x, n_u))  # n x u matrices

        Q = 10 * np.eye(n_x)  # n x n matrix
        R = np.eye(n_u)  # u x u matrix OR scalar
        P = 5 * np.eye(n_x)  # n x n matrix

        Gamma_x = np.ones((n_c, n_x))  # n_c x n_x matrix
        Gamma_u = np.ones((n_c, n_u))  # n_c x n_u matrix
        Gamma_N = np.ones((n_f, n_x))  # n_f x n_x matrix

        initial_state = np.array([0.2, 0.5])
        # initial_state = 10 * np.array(np.random.rand(n_x))
        n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
        z0 = np.zeros((n_z, 1))
        # z0 = np.array(np.random.rand(n_z, 1))
        # for i in range(initial_state.shape[0]):
        #     z0[i] = initial_state[i]

        L = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_op()
        L_adj = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_adj()

        # Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1
        eigs = np.real(sp.sparse.linalg.eigs(L_adj @ L, k=n_x, return_eigenvectors=False, which='LR'))
        L_norm = np.sqrt(max(eigs))
        alpha = 0.99 / L_norm

        P_seq, R_tilde_Cholesky_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(prediction_horizon,
                                                                                         alpha, A,
                                                                                         B,
                                                                                         Q, R, P).algorithm()
        # solving OCP by cvxpy
        # -----------------------------
        N = prediction_horizon
        n_x = A.shape[1]
        n_u = B.shape[1]
        c_t_min = - 2 * np.ones(n_c)
        c_t_max = 2 * np.ones(n_c)
        c_N_min = - 2 * np.ones(n_f)
        c_N_max = 2 * np.ones(n_f)
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
                            c_t_min <= Gamma_x @ xt_var + Gamma_u @ ut_var,  # State Constraints
                            Gamma_x @ xt_var + Gamma_u @ ut_var <= c_t_max]  # Input Constraints

        xN = x_seq[:, N]
        cost += 0.5 * cp.quad_form(xN, P)  # Terminal cost
        constraints += [c_N_min <= Gamma_N @ xN, Gamma_N @ xN <= c_N_max]  # Terminal constraints

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
        z_in = z0
        for i in range(2000):
            z_online_part = core_online.proximal_of_h_online_part(prediction_horizon=prediction_horizon,
                                                                  proximal_lambda=alpha,
                                                                  initial_state=initial_state,
                                                                  initial_guess_vector=z_in,
                                                                  state_dynamics=A,
                                                                  control_dynamics=B,
                                                                  control_weight=R,
                                                                  P_seq=P_seq,
                                                                  R_tilde_Cholesky_seq=R_tilde_Cholesky_seq,
                                                                  K_seq=K_seq,
                                                                  A_bar_seq=A_bar_seq)
            z_in = z_online_part
        self.assertEqual(len(z_cp), len(z_online_part))
        # for i in range(n_z):
        #     self.assertAlmostEqual(z_cp[i, 0], z_online_part[i, 0], delta=tol)
        print(z_cp)
        print("\n\n\n", z_online_part)


if __name__ == '__main__':
    unittest.main()
