class Constraints:
    """
    State/Control Constraints and Terminal Constraints
    """

    def __init__(self, constraints_type, stage_sets, terminal_set):
        """
        :param constraints_type: constraints type, e.g., Rectangle
        :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
        state-control constraintsn
        :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
        """
        self.__constraints_type = constraints_type
        self.__C_t = stage_sets
        self.__C_N = terminal_set

    @property
    def stage_sets(self):
        return self.__C_t

    @property
    def terminal_set(self):
        return self.__C_N

    def __str__(self):
        return f"Constraints item; Constraints type: {self.__constraints_type}, " \
               f"stage sets type: {type(self.__C_t).__name__}, " \
               f"terminal set type: {type(self.__C_N).__name__}"

    def __repr__(self):
        return f"Constraints item; stage sets type: {type(self.__C_t).__name__}, " \
               f"terminal set type: {type(self.__C_N).__name__}"
