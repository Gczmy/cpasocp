import numpy as np

import cpasocp.core.dynamics as core_dynamics
import cpasocp.core.costs as core_costs
import cpasocp.core.constraints as core_constraints


class CPASOCP:
    """
    Chambolle-Pock algorithm solving optimal control problem creation and storage
    """

    def __init__(self, prediction_horizon):
        """
        :param prediction_horizon: prediction horizon (N) of dynamic system
        """
        self.__prediction_horizon = prediction_horizon
        self.__dynamics = [None]
        self.__list_of_stage_cost = [None] * self.__prediction_horizon
        self.__terminal_cost = [None]
        self.__constraints = [None]

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

    # Dynamics ---------------------------------------------------------------------------------------------------------

    def with_dynamics(self, state_dynamics, control_dynamics):
        self.__dynamics = core_dynamics.QuadraticDynamics(state_dynamics, control_dynamics)
        return self

    # Costs ------------------------------------------------------------------------------------------------------------

    def with_cost(self, cost_type, stage_state_weight, control_weight, terminal_state_weight):
        if cost_type == "Quadratic":
            for i in range(self.__prediction_horizon):
                self.__list_of_stage_cost[i] = core_costs.QuadraticStage(stage_state_weight, control_weight)
            self.__terminal_cost = core_costs.QuadraticTerminal(terminal_state_weight)
            return self
        else:
            raise ValueError("cost type '%s' not supported" % cost_type)

    # Constraints ------------------------------------------------------------------------------------------------------

    def with_constraints(self, stage_state_constraints, stage_control_constraints, terminal_state_constraints,
                         stage_ncc_sets_constraints, terminal_ncc_set_constraints):
        self.__constraints = core_constraints.Constraints(stage_state_constraints, stage_control_constraints,
                                                          terminal_state_constraints, stage_ncc_sets_constraints,
                                                          terminal_ncc_set_constraints)
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
