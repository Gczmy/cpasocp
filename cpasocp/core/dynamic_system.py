import numpy as np
import cvxpy as cp
import scipy as sp


# Construct dynamic system
def dynamic_system(initial_state, n_x, n_u, n_c, n_f, N, A, B, Q, R, P, Gamma_x, Gamma_u, Gamma_N, c_t_min, c_t_max,
                   c_N_min, c_N_max):
    n_z = (N + 1) * n_x + N * n_u
    n_Phi = N * n_c + n_f

    # Problem statement
    # -----------------------------
    x0 = cp.Parameter(n_x)  # <--- x is a parameter of the optimisation problem P_N(x)
    u_seq = cp.Variable((n_u, N))  # <--- sequence of control actions
    x_seq = cp.Variable((n_x, N + 1))

    cost = 0
    constraints = [x_seq[:, 0] == x0]  # Initial Condition x_0 = x

    for t in range(N - 1):
        xt_var = x_seq[:, t]  # x_t
        ut_var = u_seq[:, t]  # u_t
        cost += 0.5 * (cp.quad_form(xt_var, Q) + cp.quad_form(ut_var, R))  # Stage cost

        constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var,  # Dynamics
                        c_t_min <= Gamma_x @ xt_var + Gamma_u @ ut_var,  # State Constraints
                        Gamma_x @ xt_var + Gamma_u @ ut_var <= c_t_max]  # Input Constraints

    xN = x_seq[:, N - 1]
    cost += 0.5 * cp.quad_form(xN, P)  # Terminal cost
    constraints += [c_N_min <= Gamma_N @ xN, Gamma_N @ xN <= c_N_max]  # Terminal constraints

    # Solution
    # -----------------------------
    x0.value = initial_state
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Construct z, z are all the states and inputs in a big vector
    z = np.reshape(x_seq.value[:, 0], (n_x, 1))  # x_0
    z = np.vstack((z, np.reshape(u_seq.value[:, 0], (n_u, 1))))  # u_0

    for i in range(1, N):
        z = np.vstack((z, np.reshape(x_seq.value[:, i], (n_x, 1))))
        z = np.vstack((z, np.reshape(u_seq.value[:, i], (n_u, 1))))

    z = np.vstack((z, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN

    # Construct LinearOperator Phi
    def matvec(v):
        Phi = Gamma_x @ v[0:n_x] + Gamma_u @ v[n_x:n_x + n_u]
        Phi = np.reshape(Phi, (n_c, 1))

        for i in range(1, N):
            Phi_n = Gamma_x @ v[i * (n_x + n_u):i * (n_x + n_u) + n_x] + Gamma_u @ v[
                                                                                   i * (n_x + n_u) + n_x:(i + 1) * (
                                                                                           n_x + n_u)]
            Phi = np.vstack((Phi, np.reshape(Phi_n, (n_c, 1))))

        Phi_N = Gamma_N @ v[N * (n_x + n_u):N * (n_x + n_u) + n_x]
        Phi = np.vstack((Phi, np.reshape(Phi_N, (n_f, 1))))
        return Phi

    Phi = sp.sparse.linalg.LinearOperator((n_Phi, n_z), matvec=matvec)
    Phi_z = Phi * z

    return z, Phi_z
