import cpasocp as cpa
import numpy as np
import matplotlib.pyplot as plt
import cpasocp.core.sets as core_sets

# CPASOCP generation
# -----------------------------------------------------------------------------------------------------

# dynamics
prediction_horizon = 1
n_x = 30  # state dimension
n_u = 30  # input dimension
n_c = 10
n_f = 10

# A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
A = 2 * np.array(np.random.rand(n_x, n_x))  # n x n matrices
# B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
B = np.array(np.random.rand(n_x, n_u))  # n x u matrices

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
x0 = np.array(np.random.rand(n_x))

# algorithm parameters
epsilon = 1e-4
n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
z0 = np.zeros((n_z, 1))
eta0 = np.zeros((n_z, 1))

solution = cpa.core.CPASOCP(prediction_horizon) \
    .with_dynamics(A, B) \
    .with_cost(cost_type, Q, R, P) \
    .with_constraints(constraints_type, stage_sets, terminal_set) \
    .chambolle_pock_algorithm(epsilon, x0, z0, eta0)

print(solution)
print(solution.get_z_value)

plt.figure(1)
plt.title('plot')
plt.xlabel('Iterations')
plt.ylabel('Residuals')
plt.plot(solution.get_residuals_cache, label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
plt.legend()

plt.figure(2)
plt.title('semilogy')
plt.xlabel('Iterations')
plt.ylabel('Residuals')
plt.semilogy(solution.get_residuals_cache, label=['Primal Residual', 'Dual Residual', 'Duality Gap'])
plt.legend()
plt.show()


