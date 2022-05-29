import cpasocp as cpa
import numpy as np

import cpasocp.core.sets as core_sets

# CPASOCP generation
# -----------------------------------------------------------------------------------------------------

# dynamics
prediction_horizon = 10
A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
B = np.array([[1, 1], [0.5, 1]])  # n x u matrices

# costs
cost_type = "Quadratic"
Q = 10 * np.eye(2)  # n x n matrix
R = np.eye(2)  # u x u matrix OR scalar
P = 5 * np.eye(2)  # n x n matrix

# constraints
Gamma_x = np.eye(2)  # n_c x n_x matrix
Gamma_u = np.eye(2)  # n_c x n_u matrix
Gamma_N = np.eye(2)  # n_f x n_x matrix

rectangle = core_sets.Rectangle(rect_min=-2, rect_max=2)
stage_sets_list = [rectangle] * prediction_horizon
stage_sets = core_sets.Cartesian(stage_sets_list)
terminal_set = core_sets.Rectangle(rect_min=-2, rect_max=2)
x0 = np.array([0.2, 0.5])

# algorithm parameters
epsilon = 1e-5

n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
z0 = np.ones((n_z, 1))
for i in range(x0.shape[0]):
    z0[i] = x0[i]
n_Phi = prediction_horizon * Gamma_x.shape[0] + Gamma_N.shape[0]
eta0 = np.ones((n_Phi, 1))

solution = cpa.core.CPASOCP(prediction_horizon) \
    .with_dynamics(A, B) \
    .with_cost(cost_type, Q, R, P) \
    .with_constraints(Gamma_x, Gamma_u, Gamma_N, stage_sets, terminal_set) \
    .chambolle_pock_algorithm(epsilon, x0, z0, eta0)

print(solution)
print(solution.get_z_value)
print(solution.get_eta_value)
