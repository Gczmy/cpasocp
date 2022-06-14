import numpy as np


def _check_dimension(set_type, set_dimension, vector):
    """
    Function for checking set dimensions against given vector

    If dimensions match, return vector size.
    If dimensions do not match, raise error.
    """
    vector_dimension = vector.size
    if set_dimension is None:
        set_dimension = vector_dimension
    if set_dimension != vector_dimension:
        raise ValueError('%s set dimension error: set dimension = %d, input vector dimension = %d'
                         % (set_type, set_dimension, vector_dimension))
    else:
        return vector_dimension


def _check_rectangle_min_max(rect_min, rect_max, vector):
    """
    Function for Rectangle set min and max

    If min > max, raise error.
    """
    dimension = vector.size
    if isinstance(rect_min, list) and isinstance(rect_max, list):
        for i in range(dimension):
            if rect_min[i] <= rect_max[i]:
                pass
            else:
                raise ValueError('Rectangle set min max error: rect_min[%d] can not > rect_max[%d]' % (i, i))
    elif (isinstance(rect_min, list) is False) and isinstance(rect_max, list):
        for i in range(dimension):
            if rect_min <= rect_max[i]:
                pass
            else:
                raise ValueError('Rectangle set min max error: rect_min can not > rect_max[%d]' % i)
    elif isinstance(rect_min, list) and (isinstance(rect_max, list) is False):
        for i in range(dimension):
            if rect_min[i] <= rect_max:
                pass
            else:
                raise ValueError('Rectangle set min max error: rect_min[%d] can not > rect_max' % i)
    else:
        if rect_min <= rect_max:
            pass
        else:
            raise ValueError('Rectangle set min max error: rect_min can not > rect_max')


class Rectangle:
    """
    A set of rectangle of dimension n
    """
    def __init__(self, rect_min, rect_max, dimension=None):
        """
        :param rect_min: list or scalar, Rectangle min for v0,...,vn
        :param rect_max: list or scalar, Rectangle max for v0,...,vn
        :param dimension: scalar, dimension of input vector
        """
        self.__rect_min = rect_min
        self.__rect_max = rect_max
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        _check_rectangle_min_max(self.__rect_min, self.__rect_max, vector)
        self.__shape = vector.shape
        projection = np.empty(self.__shape)
        if isinstance(self.__rect_min, list) and isinstance(self.__rect_max, list):
            for i in range(self.__dimension):
                projection[i] = max(self.__rect_min[i], vector[i])
                projection[i] = min(self.__rect_max[i], projection[i])
        elif (isinstance(self.__rect_min, list) is False) and isinstance(self.__rect_max, list):
            for i in range(self.__dimension):
                projection[i] = max(self.__rect_min, vector[i])
                projection[i] = min(self.__rect_max[i], projection[i])
        elif isinstance(self.__rect_min, list) and (isinstance(self.__rect_max, list) is False):
            for i in range(self.__dimension):
                projection[i] = max(self.__rect_min[i], vector[i])
                projection[i] = min(self.__rect_max, projection[i])
        else:
            for i in range(self.__dimension):
                projection[i] = max(self.__rect_min, vector[i])
                projection[i] = min(self.__rect_max, projection[i])
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Set dimension"""
        return self.__dimension

    @property
    def rect_min(self):
        """Rectangle min"""
        return self.__rect_min

    @property
    def rect_max(self):
        """Rectangle max"""
        return self.__rect_max


class Ball:
    """
    A set of ball of dimension n
    """

    def __init__(self, radius=1, dimension=None):
        self.__radius = radius
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = np.empty(self.__shape)
        for i in range(self.__dimension):
            projection = min((self.__radius / np.linalg.norm(vector)), 1) * vector
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Set dimension"""
        return self.__dimension


class Real:
    """
    A cone of reals of dimension n (R^n)
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = vector.copy()
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class Zero:
    """
    A zero cone ({0})
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = np.zeros(self.__dimension).reshape(self.__shape)
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class NonnegativeOrthant:
    """
    A nonnegative orthant cone of dimension n (R^n_+)
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        projection = np.empty(self.__shape)
        for i in range(self.__dimension):
            projection[i] = max(0, vector[i])
        return projection

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class SecondOrderCone:
    """
    A second order cone (N^n_2)
    """

    def __init__(self, dimension=None):
        self.__dimension = dimension
        self.__shape = None

    def project(self, vector):
        self.__dimension = _check_dimension(type(self), self.__dimension, vector)
        self.__shape = vector.shape
        last_part = vector[-1].reshape(1, 1)
        first_part = vector[0:-1]
        two_norm_of_first_part = np.linalg.norm(first_part)
        if two_norm_of_first_part <= last_part:
            projection = vector.copy()
            return projection
        elif two_norm_of_first_part <= -last_part:
            projection = np.zeros(shape=self.__shape)
            return projection
        else:
            projection_of_last_part = (two_norm_of_first_part + last_part) / 2
            projection_of_first_part = projection_of_last_part * (first_part/two_norm_of_first_part)
            projection = np.concatenate((projection_of_first_part,
                                         projection_of_last_part)).reshape(self.__shape)
            return projection

    # GETTERS
    @property
    def dimension(self):
        """Cone dimension"""
        return self.__dimension


class Cartesian:
    """
    The Cartesian product of sets (set x set x ...)
    """

    def __init__(self, sets):
        """
        :param sets: ordered list of sets
        """
        self.__sets = sets
        self.__num_sets = len(sets)
        self.__dimension = 0
        for i in self.__sets:
            if i.dimension is None:
                self.__dimension = None
                break
            else:
                self.__dimension += i.dimension
        self.__dimensions = [None] * self.__num_sets

    def project(self, list_of_vectors):
        projection_list = []
        for i in range(self.__num_sets):
            self.__dimensions[i] = _check_dimension(type(self.__sets[i]),
                                                    self.__sets[i].dimension,
                                                    list_of_vectors[i])
            projection_list.append(self.__sets[i].project(list_of_vectors[i]))
        projection = projection_list[0]
        for i in range(1, self.__num_sets):
            projection = np.vstack((projection, projection_list[i]))
        self.__dimension = sum(self.__dimensions)
        return projection

    # GETTERS
    @property
    def types(self):
        """Cartesian product of sets type"""
        product = type(self.__sets[0]).__name__
        for i in self.__sets[1:]:
            product = product + " x " + type(i).__name__
        return product

    @property
    def dimension(self):
        """set dimension"""
        return self.__dimension

    @property
    def dimensions(self):
        """List of the dimensions of each set"""
        return self.__dimensions

    @property
    def num_sets(self):
        """Number of sets that make up Cartesian set"""
        return self.__num_sets

    @property
    def rect_min(self):
        """list of rectangle_min"""
        rect_min = [None] * self.__num_sets
        for i in range(self.__num_sets):
            if type(self.__sets[i]).__name__ == 'Rectangle':
                rect_min[i] = self.__sets[i].rect_min
            else:
                raise ValueError('%s set type error: sets[%d] is not Rectangle' % (type(self.__sets[i].__name__), i))
        return rect_min

    @property
    def rect_max(self):
        """list of rectangle_max"""
        rect_max = [None] * self.__num_sets
        for i in range(self.__num_sets):
            if type(self.__sets[i]).__name__ == 'Rectangle':
                rect_max[i] = self.__sets[i].rect_max
            else:
                raise ValueError('%s set type error: sets[%d] is not Rectangle' % (type(self.__sets[i].__name__), i))
        return rect_max
