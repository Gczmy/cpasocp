import cpasocp as cpa
import numpy as np
import matplotlib.pyplot as plt
import cpasocp.core.sets as core_sets
import time

# CPASOCP generation
# -----------------------------------------------------------------------------------------------------
for problem_loop in range(10):
    # dynamics
    prediction_horizon = 10
    n_x = 20
    n_u = 10
    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    # costs
    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    # constraints
    constraints_type = 'Rectangle'
    rect_min = [-5] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rect_max = [5] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rectangle = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    stage_sets_list = [rectangle] * prediction_horizon
    stage_sets = core_sets.Cartesian(stage_sets_list)
    terminal_set = core_sets.Rectangle(rect_min=rect_min, rect_max=rect_max)
    # x0 = np.array([0.2, 0.5])
    x0 = 0.5 * np.random.rand(n_x)

    # algorithm parameters
    epsilon = 1e-3
    n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
    z0 = 0.5 * np.random.rand(n_z, 1)
    eta0 = 0.5 * np.random.rand(n_z, 1)

    start_CP = time.time()
    solution = cpa.core.CPASOCP(prediction_horizon) \
        .with_dynamics(A, B) \
        .with_cost(cost_type, Q, R, P) \
        .with_constraints(constraints_type, stage_sets, terminal_set) \
        .chambolle_pock(epsilon, x0, z0, eta0)
    CP_time = time.time() - start_CP

    print(solution)
    print(CP_time)

plt.title('semilogy')
plt.xlabel('Iterations')
plt.ylabel('Residuals')
plt.semilogy(solution.residuals_cache, label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
plt.legend()
plt.show()



