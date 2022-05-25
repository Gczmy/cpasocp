class Constraints:
    """
    State/Control Constraints and Terminal Constraints
    """

    def __init__(self, stage_state, stage_control, terminal_state, stage_sets, terminal_set):
        """
        :param stage_state: matrix (Gamma_x), describing the state constraints
        :param stage_control: matrix (Gamma_u), describing control constraints
        :param terminal_state: matrix (Gamma_N), describing terminal constraints
        :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
        state-control constraints
        :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
        """
        # check if state and control matrices have same number of rows
        if stage_state.shape[0] != stage_control.shape[0]:
            raise ValueError("Constraints matrices rows are different sizes")
        self.__Gamma_x = stage_state
        self.__Gamma_u = stage_control
        self.__Gamma_N = terminal_state
        self.__C_t = stage_sets
        self.__C_N = terminal_set

    @property
    def stage_state(self):
        return self.__Gamma_x

    @property
    def stage_control(self):
        return self.__Gamma_u

    @property
    def terminal_state(self):
        return self.__Gamma_N

    @property
    def stage_sets(self):
        return self.__C_t

    @property
    def terminal_set(self):
        return self.__C_N

    def __str__(self):
        return f"Constraints item; stage sets type: {type(self.__C_t).__name__}, " \
               f"terminal set type: {type(self.__C_N).__name__}"

    def __repr__(self):
        return f"Constraints item; stage sets type: {type(self.__C_t).__name__}, " \
               f"terminal set type: {type(self.__C_N).__name__}"
