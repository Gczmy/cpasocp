import unittest
import cpasocp.core.sets as core_sets
import numpy as np


class TestSets(unittest.TestCase):
    __rectangle = core_sets.Rectangle(rect_min=-2, rect_max=2)
    __ball = core_sets.Ball()
    __cartesian = core_sets.Cartesian([__ball, __ball, __ball, __ball])
    __num_samples = 100
    __sample_multiplier = 10
    __set_dimension = 20
    __num_test_repeats = 100

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_dimension_check(self):
        # set size equals vector size
        _ = core_sets._check_dimension("Real", 5, np.ones(5))

    def test_rectangle_project(self):
        # create set
        set_type = "Rectangle"
        rectangle = TestSets.__rectangle

        # create vector for projection
        vector = np.array(TestSets.__sample_multiplier * np.random.rand(TestSets.__set_dimension))\
            .reshape((TestSets.__set_dimension, 1))

        # create points for test
        samples = [None] * TestSets.__num_samples
        for i in range(TestSets.__num_samples):
            samples[i] = np.random.randint(-2, 2, 20)  # rectangle samples

        # test rectangle set
        self.assertEqual(set_type, type(rectangle).__name__)
        projection = rectangle.project(vector)

        for i in range(TestSets.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestSets.__set_dimension,))
                                     - projection.reshape((TestSets.__set_dimension,)),
                                     samples[i].reshape((TestSets.__set_dimension,))
                                     - projection.reshape((TestSets.__set_dimension,))) <= 0)

    def test_ball_project(self):
        # create set
        set_type = "Ball"
        ball = TestSets.__ball

        # create vector for projection
        vector = np.array(TestSets.__sample_multiplier * np.random.rand(TestSets.__set_dimension))\
            .reshape((TestSets.__set_dimension, 1))

        # create points for test
        samples = [None] * TestSets.__num_samples
        for i in range(TestSets.__num_samples):
            samples[i] = np.random.randint(-1, 1, 20)  # ball samples

        # test ball set
        self.assertEqual(set_type, type(ball).__name__)
        projection = ball.project(vector)
        for i in range(TestSets.__num_samples):
            self.assertTrue(np.inner(vector.reshape((TestSets.__set_dimension,))
                                     - projection.reshape((TestSets.__set_dimension,)),
                                     samples[i].reshape((TestSets.__set_dimension,))
                                     - projection.reshape((TestSets.__set_dimension,))) <= 0)

    def test_cartesian_project(self):
        # create set
        set_type = "Cartesian"
        sets_type = "Ball x Ball x Ball x Ball"
        cartesian = TestSets.__cartesian

        # create points for projection
        num_cones = cartesian.num_sets
        vector = [None] * num_cones
        samples = []
        for i in range(num_cones):
            samples.append([None] * TestSets.__num_samples)
            vector[i] = np.array(TestSets.__sample_multiplier * np.random.rand(TestSets.__set_dimension)) \
                .reshape((TestSets.__set_dimension, 1))

        # create set samples
        for i in range(TestSets.__num_samples):
            samples[0][i] = np.random.randint(-1, 1, TestSets.__set_dimension)  # ball samples
            samples[1][i] = np.random.randint(-1, 1, TestSets.__set_dimension)  # ball samples
            samples[2][i] = np.random.randint(-1, 1, TestSets.__set_dimension)  # ball samples
            samples[3][i] = np.random.randint(-1, 1, TestSets.__set_dimension)  # ball samples

        # test cartesian
        self.assertEqual(set_type, type(cartesian).__name__)
        self.assertEqual(sets_type, cartesian.types)
        projection = cartesian.project([vector[0], vector[1], vector[2], vector[3]])

        for i in range(num_cones):
            for j in range(TestSets.__num_samples):
                self.assertTrue(np.inner((vector[i].reshape((TestSets.__set_dimension,))
                                          - projection[i*vector[i].shape[0]: (i+1)*vector[i].shape[0]].
                                          reshape((TestSets.__set_dimension,))),
                                         (samples[i][j].reshape((TestSets.__set_dimension,))
                                          - projection[i*vector[i].shape[0]: (i+1)*vector[i].shape[0]].
                                          reshape((TestSets.__set_dimension,)))) <= 0)


if __name__ == '__main__':
    unittest.main()
