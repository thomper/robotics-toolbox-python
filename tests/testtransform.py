import unittest
from numpy import testing
from robotools.transform import *


class TestRotation(unittest.TestCase):
    def test_rotation_2d(self):
        testing.assert_array_almost_equal(rot2(0.3), np.array([[0.9553, -0.2955],
                                                               [0.2955, 0.9553]]),
                                          decimal=4)

        testing.assert_array_almost_equal(trot2(0.3), np.array([[0.9553, -0.2955, 0],
                                                                [0.2955, 0.9553, 0],
                                                                [0, 0, 1]]),
                                          decimal=4)

    def test_rotation_3d(self):
        testing.assert_array_almost_equal(rotx(0), np.eye(3), decimal=9)
        testing.assert_array_almost_equal(roty(0), np.eye(3), decimal=9)
        testing.assert_array_almost_equal(rotz(0), np.eye(3), decimal=9)

        testing.assert_array_almost_equal(rotx(np.pi / 2), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), decimal=9)
        testing.assert_array_almost_equal(roty(np.pi / 2), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), decimal=9)
        testing.assert_array_almost_equal(rotz(np.pi / 2), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), decimal=9)

        testing.assert_array_almost_equal(trotx(0), np.eye(4), decimal=9)
        testing.assert_array_almost_equal(troty(0), np.eye(4), decimal=9)
        testing.assert_array_almost_equal(trotz(0), np.eye(4), decimal=9)

        testing.assert_array_almost_equal(trotx(np.pi / 2), np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), decimal=9)
        testing.assert_array_almost_equal(troty(np.pi / 2), np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]), decimal=9)
        testing.assert_array_almost_equal(trotz(np.pi / 2), np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), decimal=9)

        testing.assert_array_almost_equal(rotx(np.pi / 2), rotx(90, 'deg'), decimal=9)
        testing.assert_array_almost_equal(roty(np.pi / 2), roty(90, 'deg'), decimal=9)
        testing.assert_array_almost_equal(rotz(np.pi / 2), rotz(90, 'deg'), decimal=9)

        testing.assert_array_almost_equal(trotx(np.pi / 2), trotx(90, 'deg'), decimal=9)
        testing.assert_array_almost_equal(troty(np.pi / 2), troty(90, 'deg'), decimal=9)
        testing.assert_array_almost_equal(trotz(np.pi / 2), trotz(90, 'deg'), decimal=9)

    def test_trotx(self):
        testing.assert_array_almost_equal(trotx(0.1), np.array([[1, 0, 0, 0],
                                                                [0, 0.995, -0.0998, 0],
                                                                [0, 0.0998, 0.995, 0],
                                                                [0, 0, 0, 1]]),
                                          decimal=4)

    def test_troty(self):
        testing.assert_array_almost_equal(troty(0.1), np.array([[0.995, 0, 0.0998, 0],
                                                                [0, 1, 0, 0],
                                                                [-0.0998, 0, 0.995, 0],
                                                                [0, 0, 0, 1]]),
                                          decimal=4)

    def test_trotz(self):
        testing.assert_array_almost_equal(trotz(0.1), np.array([[0.995, -0.0998, 0, 0],
                                                                [0.0998, 0.995, 0, 0],
                                                                [0, 0, 1, 0],
                                                                [0, 0, 0, 1]]),
                                          decimal=4)


if __name__ == '__main__':
    unittest.main()
