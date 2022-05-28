import unittest
import numpy as np
import cvxpy as cp
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.proximal_online_part as core_online


class TestSets(unittest.TestCase):
    prediction_horizon = 10
    proximal_lambda = 1e10

    A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    B = np.array([[1, 1], [0.5, 1]])  # n x u matrices

    Q = 10 * np.eye(2)  # n x n matrix
    R = np.eye(2)  # u x u matrix OR scalar
    P = 5 * np.eye(2)  # n x n matrix

    Gamma_x = np.eye(2)  # n_c x n_x matrix
    Gamma_u = np.eye(2)  # n_c x n_u matrix
    Gamma_N = np.eye(2)  # n_f x n_x matrix

    initial_state = np.array([0.2, 0.5])
    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = np.zeros((n_z, 1))
    for i in range(initial_state.shape[0]):
        z0[i] = initial_state[i]
    P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(prediction_horizon, proximal_lambda, A, B,
                                                                            Q, R, P, Gamma_x, Gamma_u,
                                                                            Gamma_N).algorithm()
    Phi = core_offline.ProximalOfflinePart(prediction_horizon, proximal_lambda, A, B, Q, R, P, Gamma_x, Gamma_u,
                                           Gamma_N).make_Phi()
    Phi_z = Phi * z0
    Phi_adj = core_offline.ProximalOfflinePart(prediction_horizon, proximal_lambda, A, B, Q, R, P, Gamma_x, Gamma_u,
                                               Gamma_N).make_Phi_adj()

    # solving OCP by cvxpy
    # -----------------------------
    N = prediction_horizon
    n_x = A.shape[1]
    n_u = B.shape[1]
    c_t_min = np.array([-2, -2])
    c_t_max = np.array([2, 2])
    c_N_min = np.array([-2, -2])
    c_N_max = np.array([2, 2])
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

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_online_part(self):
        tol = 1e-5
        z_online_part = core_online.proximal_of_h_online_part(prediction_horizon=TestSets.prediction_horizon,
                                                              proximal_lambda=TestSets.proximal_lambda,
                                                              initial_state=TestSets.initial_state,
                                                              initial_guess_vector=TestSets.z0,
                                                              state_dynamics=TestSets.A,
                                                              control_dynamics=TestSets.B,
                                                              control_weight=TestSets.R,
                                                              P_seq=TestSets.P_seq,
                                                              R_tilde_seq=TestSets.R_tilde_seq,
                                                              K_seq=TestSets.K_seq,
                                                              A_bar_seq=TestSets.A_bar_seq)
        self.assertEqual(len(TestSets.z_cp), len(z_online_part))
        for i in range(TestSets.n_z):
            self.assertAlmostEqual(TestSets.z_cp[i, 0], z_online_part[i, 0], delta=tol)


if __name__ == '__main__':
    unittest.main()
