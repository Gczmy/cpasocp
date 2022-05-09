import cpasocp as cpa
import numpy as np

import cpasocp.core.sets as core_sets

# CPASOCP generation
# -----------------------------------------------------------------------------------------------------

# dynamics
prediction_horizon = 10
proximal_lambda = 4
A = np.eye(2)  # n x n matrices
B = np.eye(2)  # n x u matrices

# costs
cost_type = "Quadratic"
Q = 10 * np.eye(2)  # n x n matrix
R = np.eye(2)  # u x u matrix OR scalar
P = 5 * np.eye(2)  # n x n matrix

# constraints
Gamma_x = np.eye(2)  # n_c x n_x matrix
Gamma_u = np.eye(2)  # n_c x n_u matrix
Gamma_N = np.eye(2)  # n_f x n_x matrix
ball = core_sets.Ball()
real = core_sets.Real()
x0 = np.array([0.2, 0.5])

# algorithm parameters
epsilon = 1e-10
n_z = (prediction_horizon + 1) * A.shape[1] + prediction_horizon * B.shape[1]
z0 = np.zeros((n_z, 1))
for i in range(x0.shape[0]):
    z0[i] = x0[i]
n_Phi = prediction_horizon * Gamma_x.shape[0] + Gamma_N.shape[0]
eta0 = np.zeros((n_Phi, 1))

problem = cpa.core.CPASOCP(prediction_horizon=prediction_horizon) \
    .with_dynamics(A, B) \
    .with_cost(cost_type, Q, R, P) \
    .with_constraints(Gamma_x, Gamma_u, Gamma_N, ball, real) \
    .chambolle_pock_algorithm(proximal_lambda=proximal_lambda, initial_state=x0,
                              epsilon=epsilon, initial_guess_z=z0, initial_guess_eta=eta0)

print(problem)
print(problem.get_z_value)
print(problem.get_eta_value)
