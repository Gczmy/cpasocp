import cpasocp.core.dynamics as core_dynamics
import cpasocp.core.costs as core_costs
import cpasocp.core.constraints as core_constraints
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.constraints_scaling as core_con_sca
import cpasocp.core.l_bfgs as core_l_bfgs
import cpasocp.core.sets as core_sets
import cpasocp.core.linear_operators as core_lin_op
import cpasocp.core.ocp_algorithms as core_algo
import numpy as np
import scipy as sp


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
        self.__q = None
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
        self.__alpha = None
        self.__status = None
        self.__scaling_factor = None
        self.__L_BFGS_k = None
        self.__L_BFGS_grad_cache = None

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
    def z(self):
        return self.__z

    @property
    def alpha(self):
        return self.__alpha

    @property
    def status(self):
        return self.__status

    @property
    def residuals_cache(self):
        return self.__residuals_cache

    @property
    def scaling_factor(self):
        return self.__scaling_factor

    @property
    def L_BFGS_k(self):
        return self.__L_BFGS_k

    @property
    def L_BFGS_grad_cache(self):
        return self.__L_BFGS_grad_cache

    def __make_alpha(self, initial_guess_z):
        """
        :param initial_guess_z: vector initial guess of (z0) of Chambolle-Pock algorithm
        """
        n_z = initial_guess_z.shape[0]
        L = core_lin_op.LinearOperator(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x, self.__Gamma_u,
                                       self.__Gamma_N).make_L_op()
        L_z = L @ np.ones((n_z, 1))
        L_adj = core_lin_op.LinearOperator(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x, self.__Gamma_u,
                                           self.__Gamma_N).make_L_adj()
        # Choose α1, α2 > 0 such that α1α2∥L∥^2 < 1
        eigs = np.real(sp.sparse.linalg.eigs(L_adj @ L, k=n_z - 2, return_eigenvectors=False))
        L_norm = np.sqrt(max(eigs))
        alpha = 0.99 / L_norm
        self.__L = L
        self.__L_z = L_z
        self.__L_adj = L_adj
        self.__alpha = alpha
        return self

    def __build_algorithm(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        algo = core_algo.Algorithms()
        algo.epsilon = epsilon
        algo.initial_state = initial_state
        algo.initial_guess_z = initial_guess_z
        algo.initial_guess_eta = initial_guess_eta
        algo.prediction_horizon = self.__prediction_horizon
        algo.state_dynamics = self.__A
        algo.control_dynamics = self.__B
        algo.control_weight = self.__R
        algo.stage_constraints_state = self.__Gamma_x
        algo.stage_constraints_input = self.__Gamma_u
        algo.stage_constraints_sets = self.__C_t
        algo.terminal_constraints_state = self.__Gamma_N
        algo.terminal_constraints_set = self.__C_N
        self.__make_alpha(initial_guess_z)

        offline = core_offline.ProximalOfflinePart()
        offline.prediction_horizon = self.__prediction_horizon
        offline.state_dynamics = self.__A
        offline.control_dynamics = self.__B
        offline.stage_state_weight = self.__Q
        offline.control_weight = self.__R
        offline.terminal_state_weight = self.__P
        offline.proximal_lambda = self.__alpha
        offline.algorithm()
        P_seq = offline.P_seq
        R_tilde_seq = offline.R_tilde_seq
        K_seq = offline.K_seq
        A_bar_seq = offline.A_bar_seq

        algo.alpha = self.__alpha
        algo.L = self.__L
        algo.L_z = self.__L_z
        algo.L_adj = self.__L_adj
        algo.P_seq = P_seq
        algo.K_seq = K_seq
        algo.R_tilde_seq = R_tilde_seq
        algo.A_bar_seq = A_bar_seq
        return algo

    # Dynamics ---------------------------------------------------------------------------------------------------------

    def with_dynamics(self, state_dynamics, control_dynamics):
        self.__A = state_dynamics
        self.__B = control_dynamics
        self.__dynamics = core_dynamics.QuadraticDynamics(self.__prediction_horizon, self.__A, self.__B)
        return self

    # Costs ------------------------------------------------------------------------------------------------------------

    def with_cost(self, cost_type, stage_state_weight, control_weight=None, terminal_state_weight=None,
                  stage_state_weight2=None):
        if cost_type == "Quadratic":
            for i in range(self.__prediction_horizon):
                self.__list_of_stage_cost[i] = core_costs.QuadraticStage(stage_state_weight, control_weight,
                                                                         stage_state_weight2)
            self.__Q = stage_state_weight
            self.__q = stage_state_weight2
            self.__R = control_weight
            self.__P = terminal_state_weight

            self.__terminal_cost = core_costs.QuadraticTerminal(terminal_state_weight)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    # Constraints ------------------------------------------------------------------------------------------------------

    def with_constraints(self, constraints_type=None, stage_sets=None, terminal_set=None):
        self.__C_t = stage_sets
        self.__C_N = terminal_set
        if constraints_type is None:
            constraints_type = 'No constraints'
            self.__C_t = core_sets.Real()
            self.__C_N = core_sets.Real()
        if self.__C_t is None:
            self.__C_t = core_sets.Real()
        if self.__C_N is None:
            self.__C_N = core_sets.Real()
        self.__constraints = core_constraints.Constraints(
            constraints_type, self.__A, self.__B, stage_sets, terminal_set)
        self.__Gamma_x, self.__Gamma_u, self.__Gamma_N = self.__constraints.make_gamma_matrix()
        return self

    # Constraints scaling ----------------------------------------------------------------------------------------------

    def with_constraints_scaling(self, constraints_type=None, stage_sets=None, terminal_set=None):
        if constraints_type is None:
            constraints_type = 'No constraints'
            self.__C_t = core_sets.Real()
            self.__C_N = core_sets.Real()
        if self.__C_t is None:
            self.__C_t = core_sets.Real()
        if self.__C_N is None:
            self.__C_N = core_sets.Real()
        self.__constraints = core_con_sca.Constraints(
            constraints_type, self.__A, self.__B, stage_sets, terminal_set)
        self.__scaling_factor = self.__constraints.scaling_factor
        self.__Gamma_x = self.__constraints.Gamma_x
        self.__Gamma_u = self.__constraints.Gamma_u
        self.__Gamma_N = self.__constraints.Gamma_N
        self.__C_t = self.__constraints.stage_sets
        self.__C_N = self.__constraints.terminal_set
        return self

    # Chambolle-Pock algorithm for Optimal Control Problems ------------------------------------------------------------

    def chambolle_pock(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        algo = self.__build_algorithm(epsilon, initial_state, initial_guess_z, initial_guess_eta)
        algo.chambolle_pock()
        self.__residuals_cache = algo.residuals_cache
        self.__z = algo.z
        self.__status = algo.status
        return self

    # SuperMann --------------------------------------------------------------------------------------------------------

    def cp_suppermann(self, epsilon, initial_state, initial_guess_z, initial_guess_eta, memory_num, c0, c1, q, beta,
                      sigma, lambda_, dirction=None):
        algo = self.__build_algorithm(epsilon, initial_state, initial_guess_z, initial_guess_eta)
        algo.chambolle_pock_supermann(memory_num, c0, c1, q, beta, sigma, lambda_, dirction)
        self.__residuals_cache = algo.residuals_cache
        self.__z = algo.z
        self.__status = algo.status
        return self

    # Chambolle-Pock algorithm scaling for Optimal Control Problems ----------------------------------------------------

    def cp_scaling(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        algo = self.__build_algorithm(epsilon, initial_state, initial_guess_z, initial_guess_eta)
        algo.chambolle_pock_scaling(self.__scaling_factor)
        self.__residuals_cache = algo.residuals_cache
        self.__z = algo.z
        self.__status = algo.status
        return self

    # ADMM for Optimal Control Problems --------------------------------------------------------------------------------

    def admm(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        algo = self.__build_algorithm(epsilon, initial_state, initial_guess_z, initial_guess_eta)
        algo.admm()
        self.__z = algo.z
        self.__status = algo.status
        self.__residuals_cache = algo.residuals_cache
        return self

    # ADMM scaling for Optimal Control Problems ------------------------------------------------------------------------

    def admm_scaling(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        algo = self.__build_algorithm(epsilon, initial_state, initial_guess_z, initial_guess_eta)
        algo.admm_scaling(self.__scaling_factor)
        self.__z = algo.z
        self.__status = algo.status
        self.__residuals_cache = algo.residuals_cache
        return self

    # L-BFGS -----------------------------------------------------------------------------------------------------------

    def L_BFGS(self, epsilon, initial_state, memory_num):
        self.__z, self.__L_BFGS_k, self.__L_BFGS_grad_cache = core_l_bfgs.LBFGS(
            self.__prediction_horizon, epsilon, initial_state, memory_num, self.__A, self.__Q, self.__R, self.__P,
            self.__q). \
            l_bfgs_algorithm()
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
