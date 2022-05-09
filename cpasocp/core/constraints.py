class Constraints:
    """
    State/Control Constraints and Terminal Constraints
    """

    def __init__(self, stage_state_constraints, stage_control_constraints, terminal_state_constraints,
                 stage_ncc_sets_constraints, terminal_ncc_set_constraints):
        """
        :param stage_state_constraints: matrix Gamma_x, describing the state constraints
        :param stage_control_constraints: matrix Gamma_u, describing control constraints
        :param terminal_state_constraints: matrix Gamma_N, describing terminal constraints
        :param stage_ncc_sets_constraints: nonempty convex closed sets C_t, describing state-control constraints
        :param terminal_ncc_set_constraints: nonempty convex closed set C_N, describing terminal constraints
        """
        # check if state and control matrices have same number of rows
        if stage_state_constraints.shape[0] != stage_control_constraints.shape[0]:
            raise ValueError("Constraints matrices rows are different sizes")
        self.__Gamma_x = stage_state_constraints
        self.__Gamma_u = stage_control_constraints
        self.__Gamma_N = terminal_state_constraints
        self.__C_t = stage_ncc_sets_constraints
        self.__C_N = terminal_ncc_set_constraints

    @property
    def stage_state_constraints(self):
        return self.__Gamma_x

    @property
    def stage_control_constraints(self):
        return self.__Gamma_u

    @property
    def terminal_state_constraints(self):
        return self.__Gamma_N

    @property
    def stage_ncc_sets(self):
        return self.__C_t

    @property
    def terminal_ncc_set(self):
        return self.__C_N

    def __str__(self):
        return f"Constraints item; stage sets type: {type(self.__C_t).__name__}, " \
               f"terminal set type: {type(self.__C_N).__name__}"

    def __repr__(self):
        return f"Constraints item; stage sets type: {type(self.__C_t).__name__}, " \
               f"terminal set type: {type(self.__C_N).__name__}"
