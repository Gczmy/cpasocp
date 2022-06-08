import numpy as np
import scipy as sp
import cpasocp.core.dynamics as core_dynamics
import cpasocp.core.costs as core_costs
import cpasocp.core.constraints as core_constraints
import cpasocp.core.linear_operators as core_lin_op
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.chambolle_pock_algorithm as core_cpa


class CPASOCP:
    """
    Chambolle-Pock algorithm solving optimal control problem creation and storage
    """

    def __init__(self, prediction_horizon):
        """
        :param prediction_horizon: prediction horizon (N) of dynamic system
        """
        self.__prediction_horizon = prediction_horizon
        self.__A = None
        self.__B = None
        self.__dynamics = None
        self.__Q = None
        self.__R = None
        self.__P = None
        self.__list_of_stage_cost = [None] * self.__prediction_horizon
        self.__terminal_cost = None
        self.__Gamma_x = None
        self.__Gamma_u = None
        self.__Gamma_N = None
        self.__C_t = None
        self.__C_N = None
        self.__constraints = None
        self.__residuals_cache = None
        self.__z = None
        self.__eta = None
        self.__alpha = None

    # GETTERS
    @property
    def dynamics(self):
        return self.__dynamics

    @property
    def list_of_stage_cost(self):
        return self.__list_of_stage_cost

    @property
    def terminal_cost(self):
        return self.__terminal_cost

    @property
    def get_z_value(self):
        return self.__z

    @property
    def get_eta_value(self):
        return self.__eta

    @property
    def get_alpha(self):
        return self.__alpha

    @property
    def get_residuals_cache(self):
        return self.__residuals_cache

    # Dynamics ---------------------------------------------------------------------------------------------------------

    def with_dynamics(self, state_dynamics, control_dynamics):
        self.__A = state_dynamics
        self.__B = control_dynamics
        self.__dynamics = core_dynamics.QuadraticDynamics(self.__prediction_horizon, self.__A, self.__B)
        return self

    # Costs ------------------------------------------------------------------------------------------------------------

    def with_cost(self, cost_type, stage_state_weight, control_weight, terminal_state_weight):
        if cost_type == "Quadratic":
            for i in range(self.__prediction_horizon):
                self.__list_of_stage_cost[i] = core_costs.QuadraticStage(stage_state_weight, control_weight)
            self.__Q = stage_state_weight
            self.__R = control_weight
            self.__P = terminal_state_weight
            self.__terminal_cost = core_costs.QuadraticTerminal(terminal_state_weight)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    # Constraints ------------------------------------------------------------------------------------------------------

    def with_constraints(self, constraints_type, stage_sets, terminal_set):
        n_x = self.__A.shape[1]
        n_u = self.__B.shape[1]

        self.__C_t = stage_sets
        self.__C_N = terminal_set

        # generate Gamma matrix depends on constraints type
        if constraints_type == 'No constraints' or constraints_type == 'Real':
            if type(stage_sets).__name__ == 'Cartesian':
                pass
            elif type(stage_sets).__name__ == 'Real':
                pass
            else:
                raise ValueError("stage sets are not Real!")
            if type(terminal_set).__name__ == 'Real':
                pass
            else:
                raise ValueError("terminal set is not Real!")
            self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
            self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
            self.__Gamma_N = np.eye(n_x)
        elif constraints_type == 'Rectangle':
            if type(stage_sets).__name__ == 'Cartesian':
                pass
            elif type(stage_sets).__name__ == 'Rectangle':
                pass
            else:
                raise ValueError("stage sets are not Rectangle!")
            if type(terminal_set).__name__ == 'Rectangle':
                pass
            else:
                raise ValueError("terminal set is not Rectangle!")
            self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
            self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
            self.__Gamma_N = np.eye(n_x)
        else:
            raise ValueError("Constraints type is not support!")
        self.__constraints = core_constraints.Constraints(constraints_type, stage_sets, terminal_set)
        return self

    # Chambolle-Pock algorithm for Optimal Control Problems -----------------------------------------------------------

    def chambolle_pock_algorithm(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        n_z = initial_guess_z.shape[0]

        L = core_lin_op.LinearOperator(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x, self.__Gamma_u,
                                       self.__Gamma_N).make_L_op()
        L_z = L @ initial_guess_z
        L_adj = core_lin_op.LinearOperator(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x,
                                           self.__Gamma_u, self.__Gamma_N).make_L_adj()
        # Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1
        eigs = np.real(sp.sparse.linalg.eigs(L_adj @ L, k=n_z-2, return_eigenvectors=False))
        L_norm = np.sqrt(max(eigs))
        self.__alpha = 0.99 / L_norm

        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            self.__prediction_horizon,
            self.__alpha, self.__A, self.__B,
            self.__Q, self.__R,
            self.__P).algorithm()
        self.__residuals_cache, self.__z, self.__eta = core_cpa.chambolle_pock_algorithm_for_ocp(
            epsilon,
            initial_guess_z,
            initial_guess_eta,
            self.__alpha, L, L_z, L_adj,
            self.__prediction_horizon,
            initial_state,
            self.__A, self.__B,
            self.__R,
            P_seq,
            R_tilde_seq,
            K_seq,
            A_bar_seq,
            self.__Gamma_x,
            self.__Gamma_N,
            self.__C_t,
            self.__C_N)
        return self

    # Class ------------------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"CPASOCP\n" \
               f"+ {self.__dynamics}\n" \
               f"+ {self.__list_of_stage_cost[0]}\n" \
               f"+ {self.__terminal_cost}\n" \
               f"+ {self.__constraints}"

    def __repr__(self):
        return f"CPASOCP with Quadratic dynamics, " \
               f"with first stage cost: {type(self.__list_of_stage_cost[0]).__name__}, " \
               f"with terminal cost: {type(self.__terminal_cost).__name__}, " \
               f"with constraints: {type(self.__constraints).__name__}."
