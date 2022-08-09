import cpasocp.core.dynamics as core_dynamics
import cpasocp.core.costs as core_costs
import cpasocp.core.constraints as core_constraints
import cpasocp.core.proximal_offline_part as core_offline
import cpasocp.core.ADMM as core_admm
import cpasocp.core.constraints_scaling as core_con_sca
import cpasocp.core.l_bfgs as core_l_bfgs
import cpasocp.core.sets as core_sets
import cpasocp.core.chambolle_pock as core_cp


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
    def get_z_value(self):
        return self.__z

    @property
    def get_alpha(self):
        return self.__alpha

    @property
    def get_status(self):
        return self.__status

    @property
    def get_residuals_cache(self):
        return self.__residuals_cache

    @property
    def get_scaling_factor(self):
        return self.__scaling_factor

    @property
    def get_L_BFGS_k(self):
        return self.__L_BFGS_k

    @property
    def get_L_BFGS_grad_cache(self):
        return self.__L_BFGS_grad_cache

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
        L, L_z, L_adj, alpha = core_cp.make_alpha(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x,
                                                      self.__Gamma_u, self.__Gamma_N, initial_guess_z)
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            self.__prediction_horizon, alpha, self.__A, self.__B, self.__Q, self.__R, self.__P).algorithm()
        CP = core_cp.ChambollePock(
            epsilon, initial_guess_z, initial_guess_eta,
            self.__prediction_horizon, initial_state, L, L_z, L_adj, alpha, self.__A, self.__B,
            self.__R, P_seq, R_tilde_seq, K_seq, A_bar_seq, self.__Gamma_x, self.__Gamma_u,
            self.__Gamma_N, self.__C_t, self.__C_N)
        CP.CP_for_ocp()
        self.__residuals_cache = CP.get_residuals_cache
        self.__z = CP.get_z
        self.__status = CP.get_status
        return self

    # SuperMann --------------------------------------------------------------------------------------------------------

    def CP_SupperMann(self, epsilon, initial_state, initial_guess_z, initial_guess_eta, memory_num, c0, c1, q, beta,
                      sigma, lambda_):
        L, L_z, L_adj, alpha = core_cp.make_alpha(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x,
                                                      self.__Gamma_u, self.__Gamma_N, initial_guess_z)
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            self.__prediction_horizon, alpha, self.__A, self.__B, self.__Q, self.__R, self.__P).algorithm()
        CP_sup = core_cp.ChambollePock(
            epsilon, initial_guess_z, initial_guess_eta,
            self.__prediction_horizon, initial_state, L, L_z, L_adj, alpha, self.__A, self.__B,
            self.__R, P_seq, R_tilde_seq, K_seq, A_bar_seq, self.__Gamma_x, self.__Gamma_u,
            self.__Gamma_N, self.__C_t, self.__C_N)
        CP_sup.CP_SuperMann(memory_num, c0, c1, q, beta, sigma, lambda_)
        self.__residuals_cache = CP_sup.get_residuals_cache
        self.__z = CP_sup.get_z
        self.__status = CP_sup.get_status
        return self

    # Chambolle-Pock algorithm scaling for Optimal Control Problems ----------------------------------------------------

    def CP_scaling(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        L, L_z, L_adj, alpha = core_cp.make_alpha(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x,
                                                      self.__Gamma_u, self.__Gamma_N, initial_guess_z)
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            self.__prediction_horizon, alpha, self.__A, self.__B, self.__Q, self.__R, self.__P).algorithm()
        CP_scaling = core_cp.ChambollePock(
            epsilon, initial_guess_z, initial_guess_eta,
            self.__prediction_horizon, initial_state, L, L_z, L_adj, alpha, self.__A, self.__B,
            self.__R, P_seq, R_tilde_seq, K_seq, A_bar_seq, self.__Gamma_x, self.__Gamma_u,
            self.__Gamma_N, self.__C_t, self.__C_N)
        CP_scaling.CP_scaling_for_ocp(self.__scaling_factor)
        self.__residuals_cache = CP_scaling.get_residuals_cache
        self.__z = CP_scaling.get_z
        self.__status = CP_scaling.get_status
        return self

    # ADMM for Optimal Control Problems --------------------------------------------------------------------------------

    def ADMM(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        L, L_z, L_adj, alpha = core_admm.make_alpha(
            self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x, self.__Gamma_u, self.__Gamma_N,
            initial_guess_z)
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            self.__prediction_horizon, alpha, self.__A, self.__B, self.__Q, self.__R, self.__P).algorithm()
        ADMM = core_admm.ADMM(epsilon, initial_guess_z, initial_guess_eta, self.__prediction_horizon, initial_state,
                              L, L_z, L_adj, alpha,
                              self.__A, self.__B, self.__R, P_seq, R_tilde_seq, K_seq,
                              A_bar_seq, self.__Gamma_x, self.__Gamma_N, self.__C_t, self.__C_N)
        ADMM.ADMM_for_ocp()
        self.__z = ADMM.get_z
        self.__status = ADMM.get_status
        self.__residuals_cache = ADMM.get_residuals_cache
        return self

    # ADMM scaling for Optimal Control Problems ------------------------------------------------------------------------

    def ADMM_scaling(self, epsilon, initial_state, initial_guess_z, initial_guess_eta):
        L, L_z, L_adj, alpha = core_admm.make_alpha(
            self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x, self.__Gamma_u, self.__Gamma_N,
            initial_guess_z)
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.ProximalOfflinePart(
            self.__prediction_horizon, alpha, self.__A, self.__B, self.__Q, self.__R, self.__P).algorithm()
        ADMM_scaling = core_admm.ADMM(epsilon, initial_guess_z, initial_guess_eta, self.__prediction_horizon,
                                      initial_state,
                                      L, L_z, L_adj, alpha,
                                      self.__A, self.__B, self.__R, P_seq, R_tilde_seq, K_seq,
                                      A_bar_seq, self.__Gamma_x, self.__Gamma_N, self.__C_t, self.__C_N)
        ADMM_scaling.ADMM_scaling_for_ocp(self.__scaling_factor)
        self.__z = ADMM_scaling.get_z
        self.__status = ADMM_scaling.get_status

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
