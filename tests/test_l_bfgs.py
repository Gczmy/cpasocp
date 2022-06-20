import unittest
import numpy as np
import cvxpy as cp
import cpasocp as cpa
import cpasocp.core.sets as core_sets


class TestLBFGS(unittest.TestCase):
    prediction_horizon = np.random.randint(15, 20)

    n_x = np.random.randint(10, 20)  # state dimension
    n_u = np.random.randint(9, n_x)  # input dimension

    # A = np.array([[1, 0.7], [-0.1, 1]])  # n x n matrices
    A = 2 * np.random.rand(n_x, n_x)  # n x n matrices
    # B = np.array([[1, 1], [0.5, 1]])  # n x u matrices
    B = np.random.rand(n_x, n_u)  # n x u matrices

    cost_type = "Quadratic"
    Q = 10 * np.eye(n_x)  # n x n matrix
    R = np.eye(n_u)  # u x u matrix OR scalar
    P = 5 * np.eye(n_x)  # n x n matrix

    constraints_type = 'Rectangle'
    rect_min = [-1] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
    rect_max = [1] * (n_x + n_u)  # constraints for x^0, ..., x^n, u^0, ..., u^n
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

    # algorithm parameters
    epsilon = 1e-3
    # initial_state = np.array([0.2, 0.5])
    initial_state = 0.5 * np.random.rand(n_x) + 0.1
    for i in range(n_x):
        initial_state[i] = np.random.rand() - 0.5
        if (i % 2) == 0:
            initial_state[i] *= 1000
        else:
            initial_state[i] *= 100

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

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_l_bfgs(self):
        tol = 1e-3
        memory_num = 5
        # Chambolle-Pock method
        # --------------------------------------------------------------------------------------------------------------
        solution = cpa.core.CPASOCP(TestLBFGS.prediction_horizon) \
            .with_dynamics(TestLBFGS.A, TestLBFGS.B) \
            .with_cost(TestLBFGS.cost_type, TestLBFGS.Q, TestLBFGS.R, TestLBFGS.P) \
            .with_constraints(TestLBFGS.constraints_type, TestLBFGS.stage_sets, TestLBFGS.terminal_set) \
            .L_BFGS(TestLBFGS.initial_state, memory_num)

        z_l_bfgs = solution.get_z_value
        print('z_l_bfgs:', z_l_bfgs)


if __name__ == '__main__':
    unittest.main()
