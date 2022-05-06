import numpy as np


class QuadraticStage:
    """
    A quadratic cost item for any stage
    """

    def __init__(self, stage_state_weight, control_weight):
        """
        :param stage_state_weight: stage state cost matrix (Q)
        :param control_weight: input cost matrix or scalar (R)
        """
        if stage_state_weight.shape[0] != stage_state_weight.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__stage_state_weight = stage_state_weight
        self.__control_weight = control_weight
        self.__most_recent_cost_value = None

    def get_cost_value(self, state, control):
        """For calculating stage cost"""
        if state.shape[0] != self.__stage_state_weight.shape[0]:
            raise ValueError("quadratic cost input stage state dimension does not match state weight matrix")
        if isinstance(self.__control_weight, np.ndarray):
            if control.shape[0] != self.__control_weight.shape[0]:
                raise ValueError("quadratic cost input control dimension does not match control weight matrix")
            self.__most_recent_cost_value = 0.5 * (state.T @ self.__stage_state_weight @ state
                                                   + control.T @ self.__control_weight @ control)
        elif isinstance(self.__control_weight, int):
            self.__most_recent_cost_value = 0.5 * (state.T @ self.__stage_state_weight @ state
                                                   + control.T @ control * self.__control_weight)
        else:
            raise ValueError("control weights type '%s' not supported" % type(self.__control_weight).__name__)
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def stage_state_weights(self):
        return self.__stage_state_weight

    @property
    def control_weights(self):
        return self.__control_weight

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"


class QuadraticTerminal:
    """
    A quadratic cost item for terminal
    """

    def __init__(self, terminal_state_weight):
        """
        :param terminal_state_weight: terminal state cost matrix (P)
        """
        if terminal_state_weight.shape[0] != terminal_state_weight.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__terminal_state_weight = terminal_state_weight
        self.__most_recent_cost_value = None

    def get_cost_value(self, state):
        """For calculating terminal cost"""
        if state.shape[0] != self.__terminal_state_weight.shape[0]:
            raise ValueError("quadratic cost input terminal state dimension does not match state weight matrix")
        self.__most_recent_cost_value = 0.5 * state.T @ self.__terminal_state_weight @ state
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def terminal_state_weights(self):
        return self.__terminal_state_weight

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"
