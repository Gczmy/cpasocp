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
        self.make_gamma_matrix_scaling()

    def make_gamma_matrix_scaling(self):
        """generate Gamma matrix depends on constraints type and scaling"""
        n_x = self.__A.shape[1]
        n_u = self.__B.shape[1]
        if self.__constraints_type == 'No constraints' or self.__constraints_type == 'Real':
            if type(self.__C_t).__name__ == 'Cartesian':
                pass
            elif type(self.__C_t).__name__ == 'Real':
                pass
            else:
                raise ValueError("stage sets are not Real while constraints_type is %s" % self.__constraints_type)
            if type(self.__C_N).__name__ == 'Real':
                pass
            else:
                raise ValueError("terminal set is not Real!")
            self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
            self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
            self.__Gamma_N = np.eye(n_x)

        elif self.__constraints_type == 'Rectangle':
            if type(self.__C_t).__name__ == 'Cartesian':
                pass
            elif type(self.__C_t).__name__ == 'Rectangle':
                pass
            else:
                raise ValueError("stage sets are not Rectangle while constraints_type is %s" % self.__constraints_type)
            if type(self.__C_N).__name__ == 'Rectangle':
                pass
            else:
                raise ValueError("terminal set is not Rectangle while constraints_type is %s" % self.__constraints_type)
            num_sets = self.__C_t.num_sets
            stage_rect_min = self.__C_t.rect_min
            stage_rect_max = self.__C_t.rect_max
            terminal_rect_min = self.__C_N.rect_min
            terminal_rect_max = self.__C_N.rect_max
            if isinstance(stage_rect_min[0], list) and isinstance(stage_rect_max[0], list):
                # scaling Gamma_x and Gamma_u matrix
                x_part_scaling = np.eye(n_x)
                for i in range(n_x):
                    x_scaling_factor = stage_rect_max[0][i] - stage_rect_min[0][i]
                    x_part_scaling[i, :] = x_part_scaling[i, :] / x_scaling_factor * 10
                u_part_scaling = np.eye(n_u)
                for i in range(n_u):
                    u_scaling_factor = stage_rect_max[0][i + n_x] - stage_rect_min[0][i + n_x]
                    u_part_scaling[i, :] = u_part_scaling[i, :] / u_scaling_factor * 10
                N_part_scaling = np.eye(n_x)
                for i in range(n_x):
                    N_scaling_factor = terminal_rect_max[i] - terminal_rect_min[i]
                    N_part_scaling[i, :] = N_part_scaling[i, :] / N_scaling_factor * 10
                self.__Gamma_x = np.vstack((x_part_scaling, np.zeros((n_u, n_x))))
                self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), u_part_scaling))

                # reconstruct stage_sets
                rectangle = core_sets.Rectangle([-5] * (n_x + n_u), [5] * (n_x + n_u))
                stage_sets_list = [rectangle] * num_sets
                self.__C_t = core_sets.Cartesian(stage_sets_list)
            else:
                self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
                self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))

            if isinstance(terminal_rect_min, list) and isinstance(terminal_rect_max, list):
                # scaling Gamma_N matrix
                N_part_scaling = np.eye(n_x)
                for i in range(n_x):
                    N_scaling_factor = terminal_rect_max[i] - terminal_rect_min[i]
                    N_part_scaling[i, :] = N_part_scaling[i, :] / N_scaling_factor * 10
                self.__Gamma_N = N_part_scaling

                # reconstruct terminal set
                self.__C_N = core_sets.Rectangle([-5] * (n_x + n_u), [5] * (n_x + n_u))
            else:
                self.__Gamma_N = np.eye(n_x)
        else:
            raise ValueError("Constraints type is not support ('No constraints' or 'Real or 'Rectangle')!")

    @property
    def Gamma_x(self):
        return self.__Gamma_x

    @property
    def Gamma_u(self):
        return self.__Gamma_u

    @property
    def Gamma_N(self):
        return self.__Gamma_N

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
