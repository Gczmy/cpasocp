import cpasocp as cpa
import numpy as np

import cpasocp.core.sets as core_sets

# CPASOCP generation
# -----------------------------------------------------------------------------------------------------

# dynamics
prediction_horizon = 10
A = np.eye(2)  # n x n matrices
B = np.eye(2)  # n x u matrices

# costs
cost_type = "Quadratic"
Q = 10*np.eye(2)  # n x n matrix
R = np.eye(2)  # u x u matrix OR scalar
P = 5*np.eye(2)  # n x n matrix

# constraints
Gamma_x = np.eye(2)  # n_c by n_x
Gamma_u = np.eye(2)  # n_c by n_u
Gamma_N = np.eye(2)  # n_f by n_x
ball = core_sets.Ball()
real = core_sets.Real()
initial_state = np.array([0.2, 0.5])

problem = cpa.core.CPASOCP(prediction_horizon=prediction_horizon)\
    .with_dynamics(A, B)\
    .with_cost(cost_type, Q, R, P)\
    .with_constraints(Gamma_x, Gamma_u, Gamma_N, ball, real)

print(problem)
