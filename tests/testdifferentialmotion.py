import unittest
from numpy import testing
from robotools.transform import *


class TestJacobian(unittest.TestCase):
    def test_rpy2jac_scalar_input(self):
        with self.assertRaises(ValueError):
            rpy2jac(1)

    def test_rpy2jac_native_sequence(self):
        testing.assert_array_almost_equal(rpy2jac((0.1, 0.2, 0.3)),
                                          np.array([[1, 0, 0.1987],
                                                    [0, 0.9950, -0.0978],
                                                    [0, 0.0998, 0.9752]]),
                                          decimal=4)

    def test_rpy2jac_single_matrix(self):
        testing.assert_array_almost_equal(rpy2jac(np.array([0.1, 0.2, 0.3])),
                                          np.array([[1, 0, 0.1987],
                                                    [0, 0.9950, -0.0978],
                                                    [0, 0.0998, 0.9752]]),
                                          decimal=4)

    def test_rpy2jac_single_matrix_zeros(self):
        testing.assert_array_almost_equal(rpy2jac(np.zeros(3)),
                                          np.eye(3),
                                          decimal=4)


if __name__ == '__main__':
    unittest.main()
