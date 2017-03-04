import unittest
from numpy import testing
from robotools.transform import *


class TestRotation(unittest.TestCase):
    def test_rotation_2d(self):
        testing.assert_array_almost_equal(rot2(0.3),
                                          np.array([[0.9553, -0.2955],
                                                    [0.2955, 0.9553]]),
                                          decimal=4)

        testing.assert_array_almost_equal(trot2(0.3),
                                          np.array([[0.9553, -0.2955, 0],
                                                    [0.2955, 0.9553, 0],
                                                    [0, 0, 1]]),
                                          decimal=4)

    def test_rotation_3d(self):
        testing.assert_array_almost_equal(rotx(0), np.eye(3), decimal=9)
        testing.assert_array_almost_equal(roty(0), np.eye(3), decimal=9)
        testing.assert_array_almost_equal(rotz(0), np.eye(3), decimal=9)

        testing.assert_array_almost_equal(rotx(np.pi / 2), np.array(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]]), decimal=9)
        testing.assert_array_almost_equal(roty(np.pi / 2), np.array(
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), decimal=9)
        testing.assert_array_almost_equal(rotz(np.pi / 2), np.array(
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]]), decimal=9)

        testing.assert_array_almost_equal(trotx(0), np.eye(4), decimal=9)
        testing.assert_array_almost_equal(troty(0), np.eye(4), decimal=9)
        testing.assert_array_almost_equal(trotz(0), np.eye(4), decimal=9)

        testing.assert_array_almost_equal(trotx(np.pi / 2), np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
                                          decimal=9)
        testing.assert_array_almost_equal(troty(np.pi / 2), np.array(
                [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
                                          decimal=9)
        testing.assert_array_almost_equal(trotz(np.pi / 2), np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                                          decimal=9)

        testing.assert_array_almost_equal(rotx(np.pi / 2), rotx(90, 'deg'),
                                          decimal=9)
        testing.assert_array_almost_equal(roty(np.pi / 2), roty(90, 'deg'),
                                          decimal=9)
        testing.assert_array_almost_equal(rotz(np.pi / 2), rotz(90, 'deg'),
                                          decimal=9)

        testing.assert_array_almost_equal(trotx(np.pi / 2), trotx(90, 'deg'),
                                          decimal=9)
        testing.assert_array_almost_equal(troty(np.pi / 2), troty(90, 'deg'),
                                          decimal=9)
        testing.assert_array_almost_equal(trotz(np.pi / 2), trotz(90, 'deg'),
                                          decimal=9)

    def test_r2t_2d_single_matrix(self):
        testing.assert_array_almost_equal(r2t(np.array([[1, 2], [3, 4]])),
                                          np.array([[1, 2, 0],
                                                    [3, 4, 0],
                                                    [0, 0, 1]]),
                                          decimal=4)

    def test_r2t_2d_multiple_matrices(self):
        rot_mats = np.tile(np.array([[1, 2], [3, 4]]), (10, 1, 1))
        transforms = r2t(rot_mats)
        self.assertEqual(transforms.shape, (10, 3, 3))
        testing.assert_array_almost_equal(transforms[1], np.array([[1, 2, 0],
                                                                   [3, 4, 0],
                                                                   [0, 0, 1]]),
                                          decimal=4)

    def test_r2t_3d_single_matrix(self):
        testing.assert_array_almost_equal(r2t(np.array([[1, 2, 3],
                                                        [4, 5, 6],
                                                        [7, 8, 9]])),
                                          np.array([[1, 2, 3, 0],
                                                    [4, 5, 6, 0],
                                                    [7, 8, 9, 0],
                                                    [0, 0, 0, 1]]),
                                          decimal=4)

    def test_r2t_3d_multiple_matrices(self):
        rot_mats = np.tile(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                           (10, 1, 1))
        transforms = r2t(rot_mats)
        self.assertEqual(transforms.shape, (10, 4, 4))
        testing.assert_array_almost_equal(transforms[4],
                                          np.array([[1, 2, 3, 0],
                                                    [4, 5, 6, 0],
                                                    [7, 8, 9, 0],
                                                    [0, 0, 0, 1]]),
                                          decimal=4)

    def test_trotx(self):
        testing.assert_array_almost_equal(trotx(0.1), np.array([[1, 0, 0, 0],
                                                                [0, 0.995,
                                                                 -0.0998, 0],
                                                                [0, 0.0998,
                                                                 0.995, 0],
                                                                [0, 0, 0, 1]]),
                                          decimal=4)

    def test_troty(self):
        testing.assert_array_almost_equal(troty(0.1),
                                          np.array([[0.995, 0, 0.0998, 0],
                                                    [0, 1, 0, 0],
                                                    [-0.0998, 0, 0.995, 0],
                                                    [0, 0, 0, 1]]),
                                          decimal=4)

    def test_trotz(self):
        testing.assert_array_almost_equal(trotz(0.1),
                                          np.array([[0.995, -0.0998, 0, 0],
                                                    [0.0998, 0.995, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]]),
                                          decimal=4)


class TestRollPitchYawConversion(unittest.TestCase):
    def test_rpy2r_native_sequence_rad(self):
        testing.assert_array_almost_equal(rpy2r((0.1, 0.2, 0.3)),
                                          np.array([[0.9363, -0.2896, 0.1987],
                                                    [0.3130, 0.9447, -0.0978],
                                                    [-0.1593, 0.1538, 0.9752]]),
                                          decimal=4)

    def test_rpy2r_native_sequence_deg(self):
        testing.assert_array_almost_equal(rpy2r((0.1, 0.2, 0.3), units='deg'),
                                          np.array([[1, -0.0052, 0.0035],
                                                    [0.0052, 1, -0.0017],
                                                    [-0.0035, 0.0018, 1]]),
                                          decimal=4)

    def test_rpy2r_native_sequence_zyx(self):
        testing.assert_array_almost_equal(rpy2r((0.1, 0.2, 0.3),
                                                axis_order='zyx'),
                                          np.array([[0.9752, -0.0370, 0.2184],
                                                    [0.0978, 0.9564, -0.2751],
                                                    [-0.1987, 0.2896, 0.9363]]),
                                          decimal=4)

    def test_rpy2r_single_row_numpy_array_rad(self):
        testing.assert_array_almost_equal(rpy2r(np.array([0.1, 0.2, 0.3])),
                                          np.array([[0.9363, -0.2896, 0.1987],
                                                    [0.3130, 0.9447, -0.0978],
                                                    [-0.1593, 0.1538, 0.9752]]),
                                          decimal=4)

    def test_rpy2r_single_row_numpy_array_deg(self):
        testing.assert_array_almost_equal(rpy2r(np.array([0.1, 0.2, 0.3]),
                                                units='deg'),
                                          np.array([[1, -0.0052, 0.0035],
                                                    [0.0052, 1, -0.0017],
                                                    [-0.0035, 0.0018, 1]]),
                                          decimal=4)

    def test_rpy2r_single_row_numpy_array_zyx(self):
        testing.assert_array_almost_equal(rpy2r(np.array([0.1, 0.2, 0.3]),
                                                axis_order='zyx'),
                                          np.array([[0.9752, -0.0370, 0.2184],
                                                    [0.0978, 0.9564, -0.2751],
                                                    [-0.1987, 0.2896, 0.9363]]),
                                          decimal=4)

    def test_rpy2r_single_row_zeros_numpy_array(self):
        testing.assert_array_almost_equal(rpy2r(np.zeros(3)), np.eye(3),
                                          decimal=4)

    def test_rpy2r_trajectory(self):
        rpy_mat = np.tile(np.array([0.1, 0.2, 0.3]), 3).reshape(3, 3)
        testing.assert_array_almost_equal(rpy2r(rpy_mat)[1],
                                          np.array([[0.9363, -0.2896, 0.1987],
                                                    [0.3130, 0.9447, -0.0978],
                                                    [-0.1593, 0.1538, 0.9752]]),
                                          decimal=4)


if __name__ == '__main__':
    unittest.main()
