import time
import cpasocp as cpa
import numpy as np
import cvxpy as cp
import cpasocp.core.sets as core_sets

f = open("Chambolle-Pock.txt", "w")
print("---\nalgname: Chambolle-Pock\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
f = open("cvxpy.txt", "w")
print("---\nalgname: cvxpy\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
f = open("ADMM.txt", "w")
print("---\nalgname: ADMM\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
for problem_loop in range(200):
    # Chambolle-Pock method
    # ------------------------------------------------------------------------------------------------------------------
    # dynamics
    prediction_horizon = np.random.randint(10, 20)

    n_x = np.random.randint(10, 20)  # state dimension
    n_u = np.random.randint(9, n_x)  # input dimension

    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = np.random.rand(n_x, n_x)  # n x n matrices
    # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    # costs
    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    # constraints
    constraints_type = 'Rectangle'
    rectangle = core_sets.Rectangle(rect_min=-2, rect_max=2)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=-2, rect_max=2)
    # x0 = np.array([0.2, 0.5])
    x0 = 0.5 * np.random.rand(n_x)

    # algorithm parameters
    epsilon_CP = 1e-4

    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1)
    eta0 = 0.5 * np.random.rand(n_z, 1)

    # start time for chambolle-pock method
    start_CP = time.time()

    solution = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints(constraints_type, stage_sets, terminal_set) \
        .chambolle_pock_algorithm(epsilon_CP, x0, z0, eta0)

    CP_time = time.time() - start_CP

    z_CP = solution.get_z_value

    if solution.get_status == 0:
        f = open("Chambolle-Pock.txt", "a")
        print(f"problem{problem_loop} converged {CP_time}", file=f)
        f.close()
    else:
        f = open("Chambolle-Pock.txt", "a")
        print(f"problem{problem_loop} failed {CP_time}", file=f)
        f.close()

    # solving OCP by cvxpy
    # ------------------------------------------------------------------------------------------------------------------
    N = prediction_horizon
    n_x = A.shape[1]
    n_u = B.shape[1]

    alpha = solution.get_alpha
    # start time for cvxpy
    start_cvxpy = time.time()

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
        constraints += [x_seq[:, t + 1] == A @ xt_var + B @ ut_var]  # Input Constraints

    xN = x_seq[:, N]
    chi_N = np.reshape(z0[N * (n_x + n_u): N * (n_x + n_u) + n_x], n_x)
    cost += 0.5 * (cp.quad_form(xN, P)
                   # + 1 / alpha * cp.sum_squares(xN - chi_N)
                   )  # Terminal cost

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

    z_cvxpy = np.vstack((z_cp, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN
    cvxpy_time = time.time() - start_cvxpy
    f = open("cvxpy.txt", "a")
    print(f"problem{problem_loop} converged {cvxpy_time}", file=f)
    f.close()

    # ADMM method
    # ------------------------------------------------------------------------------------------------------------------
    start_ADMM = time.time()

    solution_ADMM = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints(constraints_type, stage_sets, terminal_set) \
        .ADMM(z_cvxpy, z_CP, x0, z0, eta0)

    ADMM_time = time.time() - start_ADMM

    z_ADMM = solution_ADMM.get_z_ADMM_value

    f = open("ADMM.txt", "a")
    print(f"problem{problem_loop} converged {ADMM_time}", file=f)
    f.close()

    error_CP = np.linalg.norm(z_CP - z_cvxpy, np.inf)
    print("CP and cvxpy", error_CP)
    error_ADMM = np.linalg.norm(z_ADMM - z_cvxpy, np.inf)
    print("ADMM and cvxpy", error_ADMM)
