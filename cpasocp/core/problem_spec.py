import cpasocp.core.dynamics as core_dynamics
import cpasocp.core.costs as core_costs
import cpasocp.core.constraints as core_constraints
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
        self.__A = [None]
        self.__B = [None]
        self.__dynamics = [None]
        self.__Q = [None]
        self.__R = [None]
        self.__P = [None]
        self.__list_of_stage_cost = [None] * self.__prediction_horizon
        self.__terminal_cost = [None]
        self.__P = [None]
        self.__Gamma_x = [None]
        self.__Gamma_u = [None]
        self.__Gamma_N = [None]
        self.__C_t = [None]
        self.__C_N = [None]
        self.__constraints = [None]
        self.__z = [None]
        self.__eta = [None]

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

    def with_constraints(self, stage_state_constraints, stage_control_constraints, terminal_state_constraints,
                         stage_ncc_sets_constraints, terminal_ncc_set_constraints):
        self.__Gamma_x = stage_state_constraints
        self.__Gamma_u = stage_control_constraints
        self.__Gamma_N = terminal_state_constraints
        self.__C_t = stage_ncc_sets_constraints
        self.__C_N = terminal_ncc_set_constraints
        self.__constraints = core_constraints.Constraints(stage_state_constraints, stage_control_constraints,
                                                          terminal_state_constraints, stage_ncc_sets_constraints,
                                                          terminal_ncc_set_constraints)
        return self

    # Chambolle-Pock algorithm for Optimal Control Problems -----------------------------------------------------------

    def chambolle_pock_algorithm(self, proximal_lambda, epsilon, initial_state, initial_guess_z,
                                 initial_guess_eta):
        P_seq, R_tilde_seq, K_seq, A_bar_seq = core_offline.proximal_of_h_offline_part(self.__prediction_horizon,
                                                                                       proximal_lambda, self.__A,
                                                                                       self.__B, self.__Q, self.__R,
                                                                                       self.__P)
        Phi = core_offline.ProximalOfflinePart(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x,
                                               self.__Gamma_u, self.__Gamma_N).make_Phi()
        Phi_z = Phi * initial_guess_z
        Phi_star = core_offline.ProximalOfflinePart(self.__prediction_horizon, self.__A, self.__B, self.__Gamma_x,
                                                    self.__Gamma_u, self.__Gamma_N).make_Phi_star()
        self.__z, self.__eta = core_cpa.chambolle_pock_algorithm_for_ocp(epsilon, initial_guess_z, initial_guess_eta,
                                                                         Phi, Phi_z, Phi_star,
                                                                         self.__prediction_horizon, initial_state,
                                                                         self.__A, self.__B, self.__R, P_seq,
                                                                         R_tilde_seq, K_seq, A_bar_seq)
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
