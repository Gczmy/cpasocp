import numpy as np
import cpasocp.core.sets as core_sets


class Constraints:
    """
    State/Control Constraints and Terminal Constraints
    """

    def __init__(self, constraints_type, state_dynamics, control_dynamics, stage_sets, terminal_set):
        """
        :param constraints_type: constraints type, e.g., Rectangle
        :param stage_sets: nonempty convex closed sets (C) which is the Cartesian product of sets (C_t), describing
        state-control constraintsn
        :param terminal_set: nonempty convex closed set (C_N), describing terminal constraints
        """
        self.__constraints_type = constraints_type
        self.__A = state_dynamics
        self.__B = control_dynamics
        self.__Gamma_x = None
        self.__Gamma_u = None
        self.__Gamma_N = None
        self.__C_t = stage_sets
        self.__C_N = terminal_set
        self.make_gamma_matrix()

    def make_gamma_matrix(self):
        """generate Gamma matrix depends on constraints type"""
        if self.__constraints_type is None:
            self.__constraints_type = 'No constraints'
            self.__C_t = core_sets.Real()
            self.__C_N = core_sets.Real()
        if self.__C_t is None:
            self.__C_t = core_sets.Real()
        if self.__C_N is None:
            self.__C_N = core_sets.Real()
        if self.__constraints_type == 'No constraints' or self.__constraints_type == 'Real':
            if type(self.__C_t).__name__ == 'Cartesian':
                pass
            elif type(self.__C_t).__name__ == 'Real':
                pass
            else:
                raise ValueError("stage sets are not Real!")
            if type(self.__C_N).__name__ == 'Real':
                pass
            else:
                raise ValueError("terminal set is not Real!")
            if self.__A is None or self.__B is None:
                pass
            else:
                n_x = self.__A.shape[1]
                n_u = self.__B.shape[1]
                self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
                self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
                self.__Gamma_N = np.eye(n_x)
        elif self.__constraints_type == 'Rectangle':
            if type(self.__C_t).__name__ == 'Cartesian':
                pass
            elif type(self.__C_t).__name__ == 'Rectangle':
                pass
            else:
                raise ValueError("stage sets are not Rectangle!")
            if type(self.__C_N).__name__ == 'Rectangle':
                pass
            else:
                raise ValueError("terminal set is not Rectangle!")
            if self.__A is None or self.__B is None:
                pass
            else:
                n_x = self.__A.shape[1]
                n_u = self.__B.shape[1]
                self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
                self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
                self.__Gamma_N = np.eye(n_x)
        else:
            raise ValueError("Constraints type is not support!")
        return self.__Gamma_x, self.__Gamma_u, self.__Gamma_N

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
