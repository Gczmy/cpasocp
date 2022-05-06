import numpy as np


class QuadraticStage:
    """
    A quadratic cost item for any stage
    """

    def __init__(self, stage_state_weights, control_weights):
        """
        :param stage_state_weights: stage state cost matrix (Q)
        :param control_weights: input cost matrix or scalar (R)
        """
        if stage_state_weights.shape[0] != stage_state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__stage_state_weights = stage_state_weights
        self.__control_weights = control_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state, control):
        """For calculating stage cost"""
        if state.shape[0] != self.__stage_state_weights.shape[0]:
            raise ValueError("quadratic cost input stage state dimension does not match state weight matrix")
        if isinstance(self.__control_weights, np.ndarray):
            if control.shape[0] != self.__control_weights.shape[0]:
                raise ValueError("quadratic cost input control dimension does not match control weight matrix")
            self.__most_recent_cost_value = 0.5 * (state.T @ self.__stage_state_weights @ state
                                                   + control.T @ self.__control_weights @ control)
        elif isinstance(self.__control_weights, int):
            self.__most_recent_cost_value = 0.5 * (state.T @ self.__stage_state_weights @ state
                                                   + control.T @ control * self.__control_weights)
        else:
            raise ValueError("control weights type '%s' not supported" % type(self.__control_weights).__name__)
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def stage_state_weights(self):
        return self.__stage_state_weights

    @property
    def control_weights(self):
        return self.__control_weights

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

    def __init__(self, terminal_state_weights):
        """
        :param terminal_state_weights: terminal state cost matrix (P)
        """
        if terminal_state_weights.shape[0] != terminal_state_weights.shape[1]:
            raise ValueError("quadratic cost state weight matrix is not square")
        else:
            self.__terminal_state_weights = terminal_state_weights
        self.__most_recent_cost_value = None

    def get_cost_value(self, state):
        """For calculating terminal cost"""
        if state.shape[0] != self.__terminal_state_weights.shape[0]:
            raise ValueError("quadratic cost input terminal state dimension does not match state weight matrix")
        self.__most_recent_cost_value = 0.5 * state.T @ self.__terminal_state_weights @ state
        return self.__most_recent_cost_value[0, 0]

    # GETTERS
    @property
    def terminal_state_weights(self):
        return self.__terminal_state_weights

    @property
    def most_recent_cost_value(self):
        return self.__most_recent_cost_value[0, 0]

    def __str__(self):
        return f"Cost item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Cost item; type: {type(self).__name__}"
