import time
import cpasocp as cpa
import numpy as np
import cvxpy as cp
import cpasocp.core.sets as core_sets

f = open("Chambolle-Pock.txt", "w")
print("---\nalgname: Chambolle-Pock\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
# f = open("cvxpy.txt", "w")
# print("---\nalgname: cvxpy\nsuccess: converged\nfree_format: True\n---", file=f)
# f.close()
f = open("CP_scaling.txt", "w")
print("---\nalgname: CP scaling\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
f = open("ADMM.txt", "w")
print("---\nalgname: ADMM\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
f = open("ADMM_scaling.txt", "w")
print("---\nalgname: ADMM scaling\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()

for problem_loop in range(100):
    # dynamics
    prediction_horizon = np.random.randint(8, 10)

    n_x = np.random.randint(10, 20)  # state dimension
    n_u = np.random.randint(9, n_x)  # input dimension
    # n_x = 10  # state dimension
    # n_u = 10  # input dimension

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
    rect_min = [-3] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rect_max = [3] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
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
    # x0 = np.array([0.2, 0.5])
    x0 = 0.5 * np.random.rand(n_x) + 0.1
    for i in range(n_x):
        x0[i] = np.random.rand() - 0.5
        if (i % 2) == 0:
            x0[i] *= 1000
        else:
            x0[i] *= 100

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

    # algorithm parameters
    epsilon_CP = 1e-3

    # Chambolle-Pock method
    # ------------------------------------------------------------------------------------------------------------------
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
    x0_cp.value = x0
    problem = cp.Problem(cp.Minimize(cost), constraints)
    # problem.solve()
    #
    # # Construct z, z are all the states and inputs in a big vector
    # z_cp = np.reshape(x_seq.value[:, 0], (n_x, 1))  # x_0
    # z_cp = np.vstack((z_cp, np.reshape(u_seq.value[:, 0], (n_u, 1))))  # u_0
    #
    # for i in range(1, N):
    #     z_cp = np.vstack((z_cp, np.reshape(x_seq.value[:, i], (n_x, 1))))
    #     z_cp = np.vstack((z_cp, np.reshape(u_seq.value[:, i], (n_u, 1))))
    #
    # z_cvxpy = np.vstack((z_cp, np.reshape(x_seq.value[:, N], (n_x, 1))))  # xN
    # cvxpy_time = time.time() - start_cvxpy
    # f = open("cvxpy.txt", "a")
    # print(f"problem{problem_loop} converged {cvxpy_time}", file=f)
    # f.close()

    # CP_scaling
    # ------------------------------------------------------------------------------------------------------------------
    start_CP_scaling = time.time()
    solution_CP_scaling = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints_scaling(constraints_type, stage_sets, terminal_set) \
        .CP_scaling(epsilon_CP, x0, z0, eta0)

    CP_scaling_time = time.time() - start_CP_scaling

    z_CP_scaling = solution_CP_scaling.get_z_value

    if solution_CP_scaling.get_status == 0:
        f = open("CP_scaling.txt", "a")
        print(f"problem{problem_loop} converged {CP_scaling_time}", file=f)
        f.close()
    else:
        f = open("CP_scaling.txt", "a")
        print(f"problem{problem_loop} failed {CP_scaling_time}", file=f)
        f.close()

    # ADMM method
    # ------------------------------------------------------------------------------------------------------------------
    epsilon_ADMM = 1e-6
    start_ADMM = time.time()

    solution_ADMM = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints(constraints_type, stage_sets, terminal_set) \
        .ADMM(epsilon_ADMM, x0, z0, eta0)

    ADMM_time = time.time() - start_ADMM

    z_ADMM = solution_ADMM.get_z_value

    if solution_ADMM.get_status == 0:
        f = open("ADMM.txt", "a")
        print(f"problem{problem_loop} converged {ADMM_time}", file=f)
        f.close()
    else:
        f = open("ADMM.txt", "a")
        print(f"problem{problem_loop} failed {ADMM_time}", file=f)
        f.close()

    # ADMM scaling
    # ------------------------------------------------------------------------------------------------------------------
    start_ADMM_scaling = time.time()

    solution_ADMM_scaling = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints_scaling(constraints_type, stage_sets, terminal_set) \
        .ADMM_scaling(epsilon_ADMM, x0, z0, eta0)

    ADMM_scaling_time = time.time() - start_ADMM_scaling

    z_ADMM_scaling = solution_ADMM_scaling.get_z_value

    if solution_ADMM_scaling.get_status == 0:
        f = open("ADMM_scaling.txt", "a")
        print(f"problem{problem_loop} converged {ADMM_scaling_time}", file=f)
        f.close()
    else:
        f = open("ADMM_scaling.txt", "a")
        print(f"problem{problem_loop} failed {ADMM_scaling_time}", file=f)
        f.close()

    print('problem_loop:', problem_loop)

    # error_CP = np.linalg.norm(z_CP - z_cvxpy, np.inf)
    # print("CP and cvxpy", error_CP)
    # error_CP_scaling = np.linalg.norm(z_CP_scaling - z_cvxpy, np.inf)
    # print("CP_scaling and cvxpy", error_CP_scaling)
