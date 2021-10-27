from robots import cosineLaw, RobotRT, RobotRRR, LegRobot

import unittest
import numpy as np
import numpy.testing

rtol = 1e-6
atol = 1e-3
atol_rad = np.pi * 10**-3


class TestCosineLaw(unittest.TestCase):
    def test_cosine_law_too_close(self):
        solutions = cosineLaw(0, 0.1, 0.4, 0.6)
        np.testing.assert_equal(len(solutions), 0)

    def test_cosine_law_too_far(self):
        solutions = cosineLaw(0, 1.1, 0.4, 0.6)
        np.testing.assert_equal(len(solutions), 0)

    def test_cosine_law_limit(self):
        solutions = cosineLaw(1.0 - 1e-15, 0, 0.4, 0.6)
        self.assertTrue(len(solutions) > 0)
        expected = np.array([0.0, 0.0])
        for s in solutions:
            np.testing.assert_allclose(s, expected, rtol, atol)


def iterativeTest(robot, initial_pos, target, method, max_steps):
    # Nb iterations can be used to ensure long-term convergence
    joints = robot.computeMGI(initial_pos, target, method, max_steps=max_steps, seed=44203)
    final_pos = robot.computeMGD(joints)
    print(f'Target: {target}')
    print(f'Final pos: {final_pos}')
    print(f'Final joints: {joints}')
    # Numerical issues can be quite important here
    special_atol = 0.005
    np.testing.assert_allclose(target, final_pos, rtol, special_atol)


class TestRobotRT(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.model = RobotRT()

    def test_rt_operational_limits(self):
        D = np.sqrt((0.2+0.25)**2 + 0.275**2)
        expected = [[-D, D], [-D, D]]
        received = self.model.getOperationalDimensionLimits()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rt_jacobian_config0(self):
        received = self.model.computeJacobian(np.array([0, 0]))
        expected = np.array([[0.275, 0.2], [1.0, 0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rt_jacobian_config1(self):
        # config: [pi/2, 0]
        received = self.model.computeJacobian(np.array([np.pi/2, 0]))
        expected = np.array([[-0.2, 0.275], [0.0, 1.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rt_jacobian_config2(self):
        # config: [0, 0.1]
        received = self.model.computeJacobian(np.array([0, 0.1]))
        expected = np.array([[0.275, 0.3], [1.0, 0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rt_analytical_mgi_config0(self):
        nb_sol, sol = self.model.analyticalMGI(np.array([0.2, -0.275]))
        expected_sol = np.array([0, 0])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_robot_rt_jac_inverse_long0(self):
        iterativeTest(self.model, np.array([0, 0]), np.array([0.25, 0.3]), "jacobianInverse", 50000)

    def test_robot_rt_jac_inverse_long1(self):
        iterativeTest(self.model, np.array([0, 0]), np.array([-0.2, 0.25]), "jacobianInverse", 50000)

    def test_robot_rt_jac_inverse_short(self):
        iterativeTest(self.model, np.array([0, 0]), np.array([0.25, 0.3]), "jacobianInverse", 500)

    def test_robot_rt_jac_transposed_config0(self):
        iterativeTest(self.model, np.array([0, 0]), np.array([0.25, 0.3]), "jacobianTransposed", 10)

    def test_robot_rt_jac_transposed_config1(self):
        # NOTE: if initial pos is [0,0], then the optimization will be stuck
        iterativeTest(self.model, np.array([0, 0.1]), np.array([-0.2, 0.275]), "jacobianTransposed", 10)

    def test_robot_rt_analytical_mgi_config1(self):
        # config: [0, 0.2]
        nb_sol, sol = self.model.analyticalMGI(np.array([0.4, -0.275]))
        expected_sol = np.array([0, 0.2])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_robot_rt_analytical_mgi_config2(self):
        # config: [np.pi/2, 0.1]
        nb_sol, sol = self.model.analyticalMGI(np.array([0.275, 0.3]))
        expected_sol = np.array([np.pi/2, 0.1])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_robot_rt_analytical_mgi_too_far(self):
        # unreachable config
        nb_sol, sol = self.model.analyticalMGI(np.array([0.46, -0.275]))
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)


class TestRRRRobot(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.model = RobotRRR()

    def test_rrr_operational_limits(self):
        D1 = 0.4 + 0.325
        D2 = 0.5 + D1
        Z = 1.025
        expected = [[-D2, D2], [-D2, D2], [Z-D1, Z+D1]]
        received = self.model.getOperationalDimensionLimits()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rrr_jacobian_config0(self):
        received = self.model.computeJacobian(np.array([0, 0, 0]))
        expected = np.array([[-1.225, 0.0, 0.0], [0.0, 0.0, 0.725], [0.0, 0.0, 0.325]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rrr_jacobian_config1(self):
        # config: [pi/2, 0, 0]
        received = self.model.computeJacobian(np.array([np.pi/2, 0, 0]))
        expected = np.array([[0.0, -1.225, 0.0], [0.0, 0.0, 0.725], [0.0, 0.0, 0.325]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rrr_jacobian_config2(self):
        # config: [0, pi/2, 0]
        received = self.model.computeJacobian(np.array([0, np.pi/2, 0]))
        expected = np.array([[-0.5, 0.0, 0.0], [0.0, -0.725, 0.0], [0.0, -0.325, 0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rrr_jacobian_config3(self):
        # config: [0, 0, pi/2]
        received = self.model.computeJacobian(np.array([0, 0, np.pi/2]))
        expected = np.array([[-0.9, 0.0, 0.0], [0.0, -0.325, 0.4], [0.0, -0.325, 0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_robot_rrr_jac_inverse_long1(self):
        iterativeTest(self.model, np.array([0, 0.1, 0]), np.array([0.0, 0.7, 1.025]), "jacobianInverse", 50000)

    def test_robot_rrr_jac_inverse_long2(self):
        iterativeTest(self.model, np.array([0, 0.2, 0]), np.array([0.3, -0.5, 1.2]), "jacobianInverse", 50000)

    def test_robot_rrr_jac_inverse_singularity(self):
        iterativeTest(self.model, np.array([0, 0, 0]), np.array([0.0, -0.7, 1.025]), "jacobianInverse", 50000)

    def test_robot_rrr_jac_transposed_config1(self):
        iterativeTest(self.model, np.array([0, 0, 0.1]), np.array([0.0, 0.575, 1.025]), "jacobianTransposed", 100)

    def test_robot_rrr_jac_transposed_config2(self):
        iterativeTest(self.model, np.array([0, 0.2, 0]), np.array([0.6, -0.2, 1.2]), "jacobianTransposed", 50)

    def test_robot_rrr_jac_transposed_singularity(self):
        iterativeTest(self.model, np.array([0.1, 0, 0.5]), np.array([0.0, -0.7, 1.025]), "jacobianTransposed", 50)

    def test_robot_rrr_analytical_mgi_config0(self):
        # Adding a tiny offset to make 'sure' that target is considered as
        # reachable despite floating point issues
        operational_pos = np.array([0.0, 1.01-10**-14, 1.01])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 2)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_robot_rrr_analytical_mgi_config1(self):
        # Adding a tiny offset to make 'sure' that target is considered as
        # reachable despite floating point issues
        operational_pos = np.array([-0.4, 0.0, 1.01 + 0.3 + 0.31-10**-14])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 2)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_robot_rrr_analytical_mgi_config2(self):
        # No need for offset here, not at the border of the reachable space
        operational_pos = np.array([0.71, 0.0, 1.01 + 0.3])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 2)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_robot_rrr_analytical_mgi_config3(self):
        # Classic exemple near center, 4 solutions expected
        # No need for offset here, not at the border of the reachable space
        operational_pos = np.array([0.05, 0.05, 1.05])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 4)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_robot_rrr_analytical_mgi_singularity(self):
        # Exemple at center, depending on floating points approximations, answer
        # might be -1 or 4
        operational_pos = np.array([0.0, 0.0, 1.05])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_equal(nb_sol, -1)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_robot_rrr_analytical_mgi_unreachable1(self):
        # target too far
        operational_pos = np.array([0.0, 1.25, 1.025])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

    def test_robot_rrr_analytical_mgi_unreachable2(self):
        # target too far
        operational_pos = np.array([0.0, 0.05, 2.0])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

    def test_robot_rrr_analytical_mgi_unreachable3(self):
        # target too close to 'q1' position
        operational_pos = np.array([0.0, 0.5, 1.025])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)


class TestLegRobot(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.model = LegRobot()

    def test_leg_operational_limits(self):
        D1 = np.sqrt((0.3 + 0.3 + 0.225)**2 + 0.05**2)
        D2 = np.sqrt((0.5 + 0.3 + 0.3 + 0.225)**2 + 0.05**2)
        Z = 1.025
        expected = [[-D2, D2], [-D2, D2], [Z-D1, Z+D1], [-1, 1]]
        received = self.model.getOperationalDimensionLimits()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_compute_mgd_config0(self):
        joints = np.array([0, 0, 0, 0])
        expected = np.array([0.05, 1.325, 1.025, 0.0])
        received = self.model.computeMGD(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_compute_mgd_config1(self):
        joints = np.array([0, 0, 0, -np.pi/2])
        expected = np.array([0.05, 1.1, 0.80, -1.0])
        received = self.model.computeMGD(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_compute_mgd_config2(self):
        joints = np.array([-np.pi/2, np.pi/2, -np.pi/2, np.pi/2])
        expected = np.array([0.8, -0.05, 1.55, 1.0])
        received = self.model.computeMGD(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_jacobian_config0(self):
        joints = np.array([0, 0, 0, 0])
        expected = np.array(
            [
                [-1.325, 0, 0, 0],
                [0.05, 0.0, 0.0, 0.0],
                [0.0, 0.825, 0.525, 0.225],
                [0.0, 1.0, 1.0, 1.0]
            ])
        received = self.model.computeJacobian(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_jacobian_config1(self):
        # Initial direction: x+
        # At q3, starts pointing down
        joints = np.array([-np.pi/2, 0, -np.pi/2, 0])
        expected = np.array(
            [
                [0.05, 0.525, 0.525, 0.225],
                [0.8, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
        received = self.model.computeJacobian(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_jac_inverse_long0(self):
        iterativeTest(self.model,
                      np.array([0, 0.1, 0, 0]),
                      np.array([0.0, 0.7, 0.8, -1.0]),
                      "jacobianInverse", 5000)

    def test_leg_robot_jac_inverse_long1(self):
        iterativeTest(self.model,
                      np.array([0, 0.1, 0, 0]),
                      np.array([0.3, 0.5, 1.2, 0.5]),
                      "jacobianInverse", 5000)

    def test_leg_robot_jac_inverse_short(self):
        iterativeTest(self.model,
                      np.array([0, 0.1, 0, 0]),
                      np.array([0.0, 0.7, 0.8, -1.0]),
                      "jacobianInverse", 5000)

    def test_leg_robot_jac_transposed_long0(self):
        iterativeTest(self.model,
                      np.array([0, 0.1, 0, 0]),
                      np.array([0.0, 0.7, 0.6, -1.0]),
                      "jacobianTransposed", 50)

    def test_leg_robot_jac_transposed_long1(self):
        iterativeTest(self.model,
                      np.array([0, 0.1, 0, 0]),
                      np.array([0.3, 0.5, 1.2, 0.5]),
                      "jacobianTransposed", 50)

    def test_leg_robot_analytical_mgi_config0(self):
        # Adding small offset to ensure we're in a reachable position
        operational_pos = np.array([0.05, 1.325 - 1e-5, 1.025, 0.0])
        expected_sol = np.array([0, 0, 0, 0])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols > 0, True)
        # Note: Due to the small offset and the fact that we are close to a singularity, tolerance is increased
        special_atol = 0.02
        np.testing.assert_allclose(received_sol, expected_sol, rtol, special_atol)

    def test_leg_robot_analytical_mgi_config1(self):
        operational_pos = np.array([0.2, 0.5, 0.9, -0.5])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        # Here we have 2 solutions for q123 and for each of those we have two
        # solutions for q12
        np.testing.assert_equal(nb_sols, 4)
        backward_pos = self.model.computeMGD(received_sol)
        np.testing.assert_allclose(backward_pos, operational_pos, rtol, atol)

    def test_leg_robot_analytical_mgi_config2(self):
        operational_pos = np.array([0.0, 0.2, 0.8, -0.9])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        # Here, there is one option where config can be reached with the non-obvious
        # position for first angle
        np.testing.assert_equal(nb_sols, 6)
        backward_pos = self.model.computeMGD(received_sol)
        # Note: numerical stability after MGD(MGI(o)) is quite low
        special_atol = 0.01
        np.testing.assert_allclose(backward_pos, operational_pos, rtol, special_atol)

    def test_leg_robot_analytical_mgi_invalid_unreachable1(self):
        # Due to the link_offset, position can't be reached
        operational_pos = np.array([0.0, 0.01, 0.8, -0.9])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols, 0)

    def test_leg_robot_analytical_mgi_invalid_unreachable2(self):
        # A position clearly too far to be reached
        operational_pos = np.array([0.0, 1.4, 1.0, 0.0])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols, 0)

    def test_leg_robot_analytical_mgi_invalid_unreachable3(self):
        # Due to the link_offset, position can't be reached
        operational_pos = np.array([0.0, 1.1, 1.0, -1.0])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols, 0)
