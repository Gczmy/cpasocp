class QuadraticDynamics:
    """
    A pair of state (A) and control (B) dynamics matrices
    """

    def __init__(self, prediction_horizon, state_dynamics, control_dynamics):
        """
        :param prediction_horizon: prediction horizon (N) of dynamic system
        :param state_dynamics: matrix A, describing the state dynamics
        :param control_dynamics: matrix B, describing control dynamics
        """
        # check if state and control matrices have same number of rows
        if state_dynamics.shape[0] != control_dynamics.shape[0]:
            raise ValueError("Dynamics matrices rows are different sizes")
        self.__prediction_horizon = prediction_horizon
        self.__state_dynamics = state_dynamics
        self.__control_dynamics = control_dynamics

    # GETTERS
    @property
    def state_dynamics(self):
        return self.__state_dynamics

    @property
    def control_dynamics(self):
        return self.__control_dynamics

    def __str__(self):
        return f"Dynamics item; type: {type(self).__name__}"

    def __repr__(self):
        return f"Dynamics item; type: {type(self).__name__}"
