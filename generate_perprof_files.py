import time
import cpasocp as cpa
import numpy as np
import cvxpy as cp
import scipy as sp
import cpasocp.core.sets as core_sets
import cpasocp.core.linear_operators as core_lin_op

f = open("Chambolle-Pock.txt", "w")
print("---\nalgname: Chambolle-Pock\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
f = open("cvxpy.txt", "w")
print("---\nalgname: cvxpy\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
for problem_loop in range(100):
    # CPASOCP generation
    # -----------------------------------------------------------------------------------------------------
    # dynamics
    prediction_horizon = np.random.randint(5, 20)

    n_x = np.random.randint(3, 20)  # state dimension
    n_u = n_x - 1  # input dimension
    n_c = np.random.randint(2, 10)
    n_f = np.random.randint(2, 10)

    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = np.array(np.random.rand(n_x, n_x))  # n x n matrices
    # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
    B = np.array(np.random.rand(n_x, n_u))  # n x u matrices

    # costs
    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    # constraints
    Gamma_x = np.ones((n_c, n_x))  # n_c x n_x matrix
    Gamma_u = np.ones((n_c, n_u))  # n_c x n_u matrix
    Gamma_N = np.ones((n_f, n_x))  # n_f x n_x matrix

    rectangle = core_sets.Rectangle(rect_min=-2, rect_max=2)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=-2, rect_max=2)
    # x0 = np.array([0.2, 0.5])
    x0 = 5 * np.array(np.random.rand(n_x))

    # algorithm parameters
    epsilon = 1e-4

    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = np.zeros((n_z, 1))
    n_L = prediction_horizon * Gamma_x.shape[0] + Gamma_N.shape[0]
    eta0 = np.zeros((n_L, 1))

    # start time for chambolle-pock method
    start = time.time()

    solution = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints(Gamma_x, Gamma_u, Gamma_N, stage_sets, terminal_set) \
        .chambolle_pock_algorithm(epsilon, x0, z0, eta0)
    cp_time = time.time() - start
    f = open("Chambolle-Pock.txt", "a")
    print(f"problem{problem_loop} converged {cp_time}", file=f)
    f.close()

    # solving OCP by cvxpy
    # -----------------------------
    N = prediction_horizon
    n_x = A.shape[1]
    n_u = B.shape[1]

    L = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_op()
    L_z = L @ z0
    L_adj = core_lin_op.LinearOperator(prediction_horizon, A, B, Gamma_x, Gamma_u, Gamma_N).make_L_adj()
    # Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1
    eigs = np.real(sp.sparse.linalg.eigs(L_adj @ L, k=n_z-2, return_eigenvectors=False))
    L_norm = np.sqrt(max(eigs))
    alpha = 0.99 / L_norm

    # start time for cvxpy
    start = time.time()

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
        cost += 0.5 * (cp.quad_form(xt_var, Q) + cp.quad_form(ut_var, R) + 1 / alpha
                       * (cp.sum_squares(xt_var - chi_t) + cp.sum_squares(ut_var - v_t)))  # Stage cost
        constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var]  # Input Constraints

    xN = x_seq[:, N]
    chi_N = np.reshape(z0[N * (n_x + n_u): N * (n_x + n_u) + n_x], n_x)
    cost += 0.5 * (cp.quad_form(xN, P) + 1 / alpha * cp.sum_squares(xN - chi_N))  # Terminal cost

    # Solution
    x0_cp.value = x0
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Construct z, z are all the states and inputs in a big vector
    z_cp = np.reshape(x_seq.value[:, 0], (n_x, 1))  # x_0
    z_cp = np.vstack((z_cp, np.reshape(u_seq.value[:, 0], (n_u, 1))))  # u_0

    for i in range(1, N):
        z_cp = np.vstack((z_cp, np.reshape(x_seq.value[:, i], (n_x, 1))))
        z_cp = np.vstack((z_cp, np.reshape(u_seq.value[:, i], (n_u, 1))))

    z_cp = np.vstack((z_cp, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN
    cvxpy_time = time.time() - start
    f = open("cvxpy.txt", "a")
    print(f"problem{problem_loop} converged {cvxpy_time}", file=f)
    f.close()


# error = np.linalg.norm(solution.get_z_value - z_cp, np.inf)
# print(error)
