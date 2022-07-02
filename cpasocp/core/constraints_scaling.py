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
        self.__scaling_factor = None
        self.make_gamma_matrix_scaling()

    def make_gamma_matrix_scaling(self):
        """generate Gamma matrix depends on constraints type and scaling"""
        if self.__constraints_type is None:
            self.__constraints_type = 'No constraints'
            self.__C_t = core_sets.Real()
            self.__C_N = core_sets.Real()
        if self.__C_t is None:
            self.__C_t = core_sets.Real()
        if self.__C_N is None:
            self.__C_N = core_sets.Real()

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
            self.__Gamma_x = np.vstack((np.eye(n_x), np.zeros((n_u, n_x))))
            self.__Gamma_u = np.vstack((np.zeros((n_x, n_u)), np.eye(n_u)))
            self.__Gamma_N = np.eye(n_x)

            N = self.__C_t.num_sets
            stage_rect_min = self.__C_t.rect_min
            stage_rect_max = self.__C_t.rect_max
            terminal_rect_min = self.__C_N.rect_min
            terminal_rect_max = self.__C_N.rect_max
            n_z = N * (n_x + n_u) + n_x

            # make scaling factor
            x_scaling_factor = np.zeros(n_x)
            u_scaling_factor = np.zeros(n_u)

            if isinstance(stage_rect_min[0], list) and isinstance(stage_rect_max[0], list):
                for i in range(n_x):
                    x_scaling_factor[i] = stage_rect_max[0][i] - stage_rect_min[0][i]
                for i in range(n_u):
                    u_scaling_factor[i] = stage_rect_max[0][i + n_x] - stage_rect_min[0][i + n_x]
            elif (isinstance(stage_rect_min[0], list) is False) and isinstance(stage_rect_max[0], list):
                for i in range(n_x):
                    x_scaling_factor[i] = stage_rect_max[0][i] - stage_rect_min[0]
                for i in range(n_u):
                    u_scaling_factor[i] = stage_rect_max[0][i + n_x] - stage_rect_min[0]
            elif isinstance(stage_rect_min[0], list) and (isinstance(stage_rect_max[0], list) is False):
                for i in range(n_x):
                    x_scaling_factor[i] = stage_rect_max[0] - stage_rect_min[0][i]
                for i in range(n_u):
                    u_scaling_factor[i] = stage_rect_max[0] - stage_rect_min[0][i + n_x]
            else:
                for i in range(n_x):
                    x_scaling_factor[i] = stage_rect_max[0] - stage_rect_min[0]
                for i in range(n_u):
                    u_scaling_factor[i] = stage_rect_max[0] - stage_rect_min[0]

            N_scaling_factor = np.zeros(n_x)
            if isinstance(terminal_rect_min, list) and isinstance(terminal_rect_max, list):
                for i in range(n_x):
                    N_scaling_factor[i] = terminal_rect_max[i] - terminal_rect_min[i]
            elif (isinstance(terminal_rect_min, list) is False) and isinstance(terminal_rect_max, list):
                for i in range(n_x):
                    N_scaling_factor[i] = terminal_rect_max[i] - terminal_rect_min
            elif isinstance(terminal_rect_min, list) and (isinstance(terminal_rect_max, list) is False):
                for i in range(n_x):
                    N_scaling_factor[i] = terminal_rect_max - terminal_rect_min[i]
            else:
                for i in range(n_x):
                    N_scaling_factor[i] = terminal_rect_max - terminal_rect_min
            scaling_factor = np.zeros(n_z)
            for i in range(N):
                x_index = i * (n_x + n_u)
                u_index = i * (n_x + n_u) + n_x
                scaling_factor[x_index: x_index + n_x] = x_scaling_factor[:]
                scaling_factor[u_index: u_index + n_u] = u_scaling_factor[:]
            N_index = N * (n_x + n_u)
            scaling_factor[N_index: N_index + n_x] = N_scaling_factor[:]
            self.__scaling_factor = np.reshape(scaling_factor, (n_z, 1))

            # reconstruct stage_sets
            rectangle = core_sets.Rectangle(-0.5, 0.5)
            stage_sets_list = [rectangle] * N
            self.__C_t = core_sets.Cartesian(stage_sets_list)
            # reconstruct terminal set
            self.__C_N = core_sets.Rectangle(-0.5, 0.5)
        else:
            raise ValueError("Constraints type is not support ('No constraints' or 'Real or 'Rectangle')!")

    @property
    def scaling_factor(self):
        return self.__scaling_factor

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
