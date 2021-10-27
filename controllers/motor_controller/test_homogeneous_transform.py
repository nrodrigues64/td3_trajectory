import homogeneous_transform as ht
import unittest
import numpy as np
import numpy.testing
from math import pi

rtol = 1e-6
atol = 1e-6


class TestRotX(unittest.TestCase):
    def test_rot_x_0(self):
        received = ht.rot_x(0)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_x_halfpi(self):
        received = ht.rot_x(pi/2)
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, -1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)
        # self.assertTrue(np.allclose(received, expected),np.testing.assert_allclose)


class TestRotY(unittest.TestCase):
    def test_rot_y_0(self):
        received = ht.rot_y(0)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_y_halfpi(self):
        received = ht.rot_y(pi/2)
        expected = np.array([[0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [-1, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)


class TestRotZ(unittest.TestCase):
    def test_rot_z_0(self):
        received = ht.rot_z(0)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_z_halfpi(self):
        received = ht.rot_z(pi/2)
        expected = np.array([[0, -1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)


class TestTranslation(unittest.TestCase):
    def test_translation_0(self):
        received = ht.translation(np.array([0, 0, 0]))
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_translation_other(self):
        vec = np.array([1, -2, 0])
        received = ht.translation(vec)
        expected = np.eye(4, dtype=np.double)
        for i in range(3):
            expected[i, 3] = vec[i]
        np.testing.assert_allclose(received, expected, rtol, atol)


class TestInvertTransform(unittest.TestCase):
    def test_invert_transform_eye(self):
        arg = np.eye(4, dtype=np.double)
        received = ht.invert_transform(arg)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_rot(self):
        alpha = np.pi/3
        T = ht.rot_x(alpha)
        received = ht.invert_transform(T)
        expected = ht.rot_x(-alpha)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_translation(self):
        vec = np.array([1, 0, -2])
        T = ht.translation(vec)
        received = ht.invert_transform(T)
        expected = ht.translation(-vec)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_mixed(self):
        a1 = np.pi/3
        v1 = np.array([1, 0, -2])
        a2 = -np.pi/4
        v2 = np.array([-1, 2, -5])
        T = ht.rot_x(a1) @ ht.translation(v1) @ ht.rot_y(a2) @ ht.translation(v2)
        received = ht.invert_transform(T)
        expected = ht.translation(-v2) @ ht.rot_y(-a2) @ ht.translation(-v1) @ ht.rot_x(-a1)
        np.testing.assert_allclose(received, expected, rtol, atol)


class TestDRotX(unittest.TestCase):
    def test_d_rot_x_0(self):
        received = ht.d_rot_x(0)
        expected = np.zeros((4, 4), dtype=np.double)
        expected[1, 2] = -1
        expected[2, 1] = 1
        np.testing.assert_allclose(received, expected, rtol, atol)


class TestDRotY(unittest.TestCase):
    def test_d_rot_y_0(self):
        received = ht.d_rot_y(0)
        expected = np.zeros((4, 4), dtype=np.double)
        expected[0, 2] = 1
        expected[2, 0] = -1
        np.testing.assert_allclose(received, expected, rtol, atol)


class TestDRotZ(unittest.TestCase):
    def test_d_rot_z_0(self):
        received = ht.d_rot_z(0)
        expected = np.zeros((4, 4), dtype=np.double)
        expected[0, 1] = -1
        expected[1, 0] = 1
        np.testing.assert_allclose(received, expected, rtol, atol)
