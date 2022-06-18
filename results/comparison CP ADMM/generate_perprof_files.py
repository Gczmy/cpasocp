import time
import cpasocp as cpa
import numpy as np
import cpasocp.core.sets as core_sets

f = open("Chambolle-Pock.txt", "w")
print("---\nalgname: Chambolle-Pock\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()
f = open("ADMM.txt", "w")
print("---\nalgname: ADMM\nsuccess: converged\nfree_format: True\n---", file=f)
f.close()

for problem_loop in range(100):
    # dynamics
    prediction_horizon = np.random.randint(15, 20)

    n_x = np.random.randint(10, 20)  # state dimension
    n_u = np.random.randint(9, n_x)  # input dimension
    # n_x = 4  # state dimension
    # n_u = 3  # input dimension

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

    rectangle = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    # x0 = np.array([0.2, 0.5])
    x0 = 0.5 * np.random.rand(n_x) + 0.1

    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1) + 0.1
    eta0 = 0.5 * np.random.rand(n_z, 1) + 0.1

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

    # ADMM method
    # ------------------------------------------------------------------------------------------------------------------
    epsilon_ADMM = 1e-3
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

    print('problem_loop:', problem_loop)
