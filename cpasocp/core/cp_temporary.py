import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.sparse.linalg as lingalg

# Define the problem data
# -----------------------------
n_x = 2  # 2 states
n_u = 2  # 2 input
n_c = 2
n_f = 2

A = np.array([[1, 0.7], [-0.1, 1]])  # n_x by n_x
B = np.array([[1, 1], [0.5, 1]])  # n_x by n_u
N = 10  # prediction horizon

# Given matrices Q, R, P ...
# -----------------------------
Q = 10 * np.eye(2)  # n x n matrix
R = np.eye(2)  # u x u matrix OR scalar
P = 5 * np.eye(2)  # n x n matrix
Gamma_x = np.eye(2)  # n_c x n_x matrix
Gamma_u = np.eye(2)  # n_c x n_u matrix
Gamma_N = np.eye(2)  # n_f x n_x matrix
c_t_min = np.array([-2, -2])
c_t_max = np.array([2, 2])
c_N_min = np.array([-2, -2])
c_N_max = np.array([2, 2])
initial_state = np.array([0.2, 0.5])


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

    return z, Phi, Phi_z
z, phi, phi_z = dynamic_system(initial_state, n_x, n_u, n_c, n_f, N, A, B, Q, R, P, Gamma_x, Gamma_u, Gamma_N, c_t_min, c_t_max,
                   c_N_min, c_N_max)
print(z)
# Construct Proximal of h Offline part
def proximal_of_h_offline_part(lamda, A, B, Q, R, P):
    P_seq = np.zeros((n_x, n_x, N + 1))  # tensor
    R_tilde_seq = np.zeros((n_x, n_x, N))  # tensor
    K_seq = np.zeros((n_x, n_x, N))  # tensor
    A_bar_seq = np.zeros((n_x, n_x, N))  # tensor
    P_0 = P + lamda * np.eye(n_x)
    P_seq[:, :, N] = P_0

    for i in range(N):
        R_tilde_seq[:, :, N - i - 1] = R + 1 / lamda * np.eye(n_u) + B.T @ P_seq[:, :, N - i] @ B
        K_seq[:, :, N - i - 1] = - np.linalg.inv(R_tilde_seq[:, :, N - i - 1]) @ B.T @ P_seq[:, :, N - i] @ A
        A_bar_seq[:, :, N - i - 1] = A + B @ K_seq[:, :, N - i - 1]
        P_seq[:, :, N - i - 1] = Q + 1 / lamda * np.eye(n_x) \
                                 + K_seq[:, :, N - i - 1].T @ (R + np.eye(n_u)) @ K_seq[:, :, N - i - 1] \
                                 + A_bar_seq[:, :, N - i - 1].T @ P_seq[:, :, N - i - 1] @ A_bar_seq[:, :, N - i - 1]

    return P_seq, R_tilde_seq, K_seq, A_bar_seq


lamda = 5
P_seq, R_tilde_seq, K_seq, A_bar_seq = proximal_of_h_offline_part(lamda, A, B, Q, R, P)


# Construct Proximal of h Online part
def proximal_of_h_online_part(w, lamda, A, B, R_tilde_seq, K_seq):
    q_0 = - 1 / lamda * w[N * (n_x + n_u):N * (n_x + n_u) + n_x]

    q_seq = np.zeros((n_x, 1, N + 1))  # tensor
    d_seq = np.zeros((n_x, 1, N))  # tensor
    q_seq[:, :, N] = q_0

    for t in range(N):
        d_seq[:, :, N - t - 1] = np.linalg.inv(R_tilde_seq[:, :, N - t - 1]) \
                                 @ (1 / lamda * w[(N - t) * (n_x + n_u) - n_u:(N - t) * (n_x + n_u)]
                                    - B.T @ q_seq[:, :, N - t])

        q_seq[:, :, N - t - 1] = K_seq[:, :, N - t - 1].T \
                                 @ ((R + 1 / lamda * np.eye(n_u)) @ d_seq[:, :, N - t - 1]
                                    - 1 / lamda * w[(N - t - 1) * (n_x + n_u):(N - t - 1) * (n_x + n_u) + n_x]) \
                                 + 1 / lamda * w[(N - t) * (n_x + n_u) - n_u:(N - t) * (n_x + n_u)] \
                                 + A_bar_seq[:, :, N - t - 1] @ (P_seq[:, :, N - t] @ B @ d_seq[:, :, N - t - 1]
                                                                 + q_seq[:, :, N - t])

    x_0 = initial_state
    x_seq = np.zeros((n_x, 1, N + 1))  # tensor
    u_seq = np.zeros((n_x, 1, N))  # tensor
    x_seq[:, :, N] = np.reshape(x_0, (n_x, 1))

    for t in range(N):
        u_seq[:, :, N - t - 1] = K_seq[:, :, t] @ x_seq[:, :, N - t - 1] + d_seq[:, :, t]
        x_seq[:, :, N - t - 1] = A @ x_seq[:, :, N - t] + B @ u_seq[:, :, N - t - 1]

    # Construct Proximal of h at w
    prox = np.reshape(x_seq[:, :, 0], (n_x, 1))  # x_0
    prox = np.vstack((prox, np.reshape(u_seq[:, :, 0], (n_u, 1))))  # u_0

    for i in range(1, N):
        prox = np.vstack((prox, np.reshape(x_seq[:, :, 0], (n_x, 1))))
        prox = np.vstack((prox, np.reshape(u_seq[:, :, 0], (n_u, 1))))

    prox = np.vstack((prox, np.reshape(x_seq[:, :, 0], (n_x, 1))))  # xN
    return prox




# Chambolle-Pock algorithm for deterministic optimal control problems
# initial guess
n_z = (N + 1) * n_x + N * n_u
n_Phi = N * n_c + n_f
z_0 = np.ones((n_z, 1))
eta_0 = np.ones((n_Phi, 1))
z, Phi, Phi_z = dynamic_system(initial_state, n_x, n_u, n_c, n_f, N, A, B, Q, R, P, Gamma_x, Gamma_u, Gamma_N, c_t_min,
                        c_t_max, c_N_min, c_N_max)

w = z.copy()
prox = proximal_of_h_online_part(w, lamda, A, B, R_tilde_seq, K_seq)

def Chambolle_Pock_algorithm(z, Phi, Phi_z, z_0, eta_0):
    epsilon = 1e-10

    # Invoke Algorithm 1
    P_seq, R_tilde_seq, K_seq, A_bar_seq = proximal_of_h_offline_part(lamda, A, B, Q, R, P)
    print("K_seq:", K_seq[:, :, 10 - 1])
    # Choose α1, α2 > 0 such that α1α2∥Phi∥^2 < 1
    alpha_1 = 0.99 / np.linalg.norm(Phi_z)
    alpha_2 = 0.99 / np.linalg.norm(Phi_z)

    # Construct LinearOperator Phi_star
    def matvec(v):
        Phi_star_x = np.reshape(Gamma_x.T @ v[0:n_c], (n_x, 1))
        Phi_star_u = np.reshape(Gamma_u.T @ v[0:n_c], (n_u, 1))
        Phi_star = np.vstack((Phi_star_x, Phi_star_u))

        for i in range(1, N):
            phi_n_x = np.reshape(Gamma_x.T @ v[(i + 1) * n_c:(i + 2) * n_c], (n_x, 1))
            phi_n_u = np.reshape(Gamma_u.T @ v[(i + 1) * n_c:(i + 2) * n_c], (n_u, 1))
            Phi_star = np.vstack((Phi_star, phi_n_x, phi_n_u))

        Phi_star_N = np.reshape(Gamma_N.T @ v[N * n_c:(N + 1) * n_c], (n_x, 1))
        Phi_star = np.vstack((Phi_star, Phi_star_N))

        return Phi_star

    Phi_star = sp.sparse.linalg.LinearOperator((n_z, n_Phi), matvec=matvec)

    z_seq = np.zeros((n_z, 1, N + 1))  # tensor
    eta_seq = np.zeros((n_Phi, 1, 2 * N + 1))  # tensor
    z_seq[:, :, N] = z_0
    eta_seq[:, :, 2 * N] = eta_0

    for k in range(N + 1):
        z_seq[:, :, N - k - 1] = proximal_of_h_online_part(
            z_seq[:, :, N - k] - alpha_1 * Phi_star @ eta_seq[:, :, 2 * N - k - 1], alpha_1, A, B, R_tilde_seq, K_seq)
        eta_seq[:, :, 2 * N - k - 2] = eta_seq[:, :, 2 * N - k - 1] + alpha_2 * Phi @ (
                2 * z_seq[:, :, N - k - 1] - z_seq[:, :, N - k])
        eta_seq[:, :, 2 * N - k - 3] = eta_seq[:, :, 2 * N - k - 2] - alpha_2 * (
                1 / alpha_2 * eta_seq[:, :, 2 * N - k - 2])

    return z_seq, eta_seq


# z, eta = Chambolle_Pock_algorithm(z, Phi, Phi_z, z_0, eta_0)
# print(z[:, :, 0])
