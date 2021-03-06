import numpy as np
from abc import abstractmethod
import homogeneous_transform as ht
import math
from scipy import optimize


def cosineLaw(x, y, L1, L2):
    """
    Parameters
    ----------
    x : double
    y : double
    L1: double
    L2: double

    Returns
    -------
    solutions : list((double,double))
        The list of couples (alpha, beta) that allows to reach the provided
        target
    """
    solutions = []
    dist = math.sqrt(x**2+y**2)
    if (dist < abs(L1 - L2)) or dist > (L1 + L2):
        return solutions
    phi = math.atan2(y, x)
    alpha = math.acos((L1**2+dist**2-L2**2) / (2*L1*dist))
    beta = math.acos((L1**2+L2**2-dist**2) / (2*L1*L2))
    solutions.append(np.array([phi+alpha, beta - np.pi]))
    tol = 1e-9  # Only consider 1 solution if alpha is too small
    if abs(alpha) > tol:
        solutions.append(np.array([phi-alpha, np.pi - beta]))
    return solutions


class RobotModel:
    def getNbJoints(self):
        """
        Returns
        -------
        length : int
            The number of joints for the robot
        """
        return len(self.getJointsNames())

    def getMotorsNames(self):
        """
        Returns
        -------
        motor_names : string[]
            The list of names for the motors
        """
        return [name + "_motor" for name in self.getJointsNames()]

    def getSensorsNames(self):
        """
        Returns
        -------
        motor_names : string[]
            The list of names for the motors
        """
        return [name + "_sensor" for name in self.getJointsNames()]

    @abstractmethod
    def getJointsNames(self):
        """
        Returns
        -------
        joint_names : string[]
            The list of names given to robot joints
        """

    @abstractmethod
    def getJointsLimits(self):
        """
        Returns
        -------
        np.array
            The values limits for the robot joints, each row is a different
            joint, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getOperationalDimensionNames(self):
        """
        Returns
        -------
        joint_names : string array
            The list of names of the operational dimensions
        """

    @abstractmethod
    def getOperationalDimensionLimits(self):
        """
        Returns
        -------
        limits : np.array(x,2)
            The values limits for the operational dimensions, each row is a
            different dimension, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getBaseFromToolTransform(self, joints):
        """
        Parameters
        ----------
        joints_position : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The transformation matrix from base to tool
        """

    @abstractmethod
    def computeMGD(self, q):
        """
        Parameters
        ----------
        q : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The coordinate of the effectors in the operational space
        """

    @abstractmethod
    def computeJacobian(self, joints):
        """
        Parameters
        ----------
        joints : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The jacobian of the robot for given joints values
        """

    @abstractmethod
    def analyticalMGI(self, target):
        """
        Parameters
        ----------
        joints : np.arraynd shape(n,)
            The current values of the joints of the robot in joint space
        target : np.arraynd shape(m,)
            The target in operational space

        Returns
        -------
        nb_solutions : int
            The number of solutions for the given target, -1 if there is an
            infinity of solutions
        joint_pos : np.ndarray shape(X,) or None
            One of the joint configuration which allows to reach the provided
            target. If no solution is available, returns None.
        """

    def computeMGI(self, joints, target, method, max_steps=50, seed=None):
        """
        Parameters
        ----------
        joints : np.ndarray shape(n,)
            The current position of joints in angular space
        target : np.ndarray shape(m,)
            The target in operational space
        method : str
            The method used to compute MGI, available choices:
            - analyticalMGI
            - jacobianInverse
            - jacobianTransposed
        seed : None or int
            The seed used for inner random components if needed
        """
        if method == "analyticalMGI":
            nb_sols, sol = self.analyticalMGI(target)
            return sol
        elif method == "jacobianInverse":
            return self.solveJacInverse(joints, target, max_steps=max_steps, seed=seed)
        elif method == "jacobianTransposed":
            return self.solveJacTransposed(joints, target, max_epochs=max_steps, seed=seed)
        raise RuntimeError("Unknown method: " + method)

    def solveJacInverse(self, joints, target, max_steps=500, tol=1e-6, seed=None):
        """
        Parameters
        ----------
        joints: np.ndarray shape(n,)
            The initial position for the search in angular space
        target: np.ndarray shape(n,)
            The wished target for the tool in operational space
        max_steps: int
            The maximal number of steps allowed
        seed: None or int
            Since the method comport some random part, the seed can be specified
            to obtain reproductible results.
        """
        max_step_size = 0.05
        rng = np.random.default_rng(seed)
        for i in range(max_steps):
            pos = self.computeMGD(joints)
            error = target - pos
            if np.linalg.norm(error) < tol:
                break
            try:
                J_inv = np.linalg.inv(self.computeJacobian(joints))
                step = J_inv @ error
                step_size = np.linalg.norm(step)
                if step_size > max_step_size:
                    step = step / step_size * max_step_size
                joints = joints + step
            except np.linalg.LinAlgError:
                noise_level = 1e-1
                random_offset = rng.uniform(-noise_level, noise_level, joints.shape[0])
                print(f'LinAlgError: randomizing by {random_offset}')
                joints = joints + random_offset
        return joints

    def solveJacTransposed(self, joints, target, max_epochs=10, max_iterations=500, seed=None):
        limits = self.getJointsLimits()

        def cost_func(x):
            return np.linalg.norm(self.computeMGD(x) - target, 2)

        def jac_func(x):
            return - 2 * (self.computeJacobian(x).transpose() @ (target - self.computeMGD(x)))
        tol_cost = 10**-4
        tol_joints = 10 ** -3
        min_improvement = tol_cost * 10**-2
        last_joints = None
        last_cost = None
        cost = cost_func(joints)
        rng = np.random.default_rng(seed)
        epoch = 0
        while epoch < max_epochs and cost > tol_cost:
            print(f'Epoch {epoch:3d}:\n\tjoints: {joints}\n\tcost: {cost_func(joints):.5f}')
            # If change of joints was low, add noise
            if last_joints is not None:
                joint_diff = np.linalg.norm(last_joints-joints)
                cost_diff = cost - last_cost
                if joint_diff < tol_joints and cost_diff < min_improvement:
                    noise_level = 1e-1
                    random_offset = rng.uniform(-noise_level, noise_level, joints.shape[0])
                    joints = joints + random_offset
                    print('randomizing joints')
            res = optimize.minimize(cost_func, joints,
                                    jac=jac_func,
                                    bounds=optimize.Bounds(limits[:, 0], limits[:, 1]),
                                    method="SLSQP",
                                    options={"maxiter": max_iterations})
            epoch += 1
            last_joints = joints
            last_cost = cost
            joints = res.x
            cost = cost_func(joints)
        return joints


class RobotRT(RobotModel):
    """
    Model a robot with a 2 degrees of freedom: 1 rotation and 1 translation

    The operational space of the robot is 2 dimensional because it can only move inside a plane
    """
    def __init__(self):
        self.W = 0.05
        self.L0 = 1.0
        self.L1 = 0.2
        self.L2 = 0.25 + self.W/2  # Distance including the offset
        self.max_q1 = 0.25
        self.T_0_1 = ht.translation([0, 0, self.L0+self.W/2])
        self.T_1_2 = ht.translation([self.L1, 0, 0])
        self.T_2_E = ht.translation([0.0, -self.L2, 0]) @ ht.rot_z(np.pi)

    def getJointsNames(self):
        return ["q1", "q2"]

    def getJointsLimits(self):
        return np.array([[-np.pi, np.pi], [0, 0.55]], dtype=np.double)

    def getOperationalDimensionNames(self):
        return ["x", "y"]

    def getOperationalDimensionLimits(self):
        max_dist = np.sqrt((self.L1+self.max_q1)**2 + self.L2**2)
        return np.array([[-1, 1],  [-1, 1]]) * max_dist

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.translation(joints[1] * np.array([1, 0, 0]))
        return T_0_1 @ T_1_2 @ self.T_2_E

    def computeMGD(self, q):
        tool_pos = self.getBaseFromToolTransform(q) @ np.array([0, 0, 0, 1])
        return tool_pos[:2]

    def analyticalMGI(self, target):
        dist = np.linalg.norm(target)
        min_dist = np.sqrt(self.L1**2 + self.L2**2)
        max_dist = np.sqrt((self.L1 + self.max_q1)**2 + self.L2**2)
        if dist < min_dist or dist > max_dist:
            return 0, None
        # Using basic geometry to get distance of joint q1
        q1 = np.sqrt(dist**2 - self.L2**2) - self.L1
        dir_to_target = math.atan2(target[1], target[0])
        dir_offset = math.atan2(self.L2, self.L1+q1)
        q0 = dir_to_target + dir_offset
        return 1, np.array([q0, q1])

    def computeJacobian(self, joints):
        J = np.zeros((2, 2), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        J[:, 0] = (self.T_0_1 @ ht.d_rot_z(joints[0]) @ self.T_1_2 @
                   ht.translation(joints[1] * np.array([1, 0, 0])) @ self.T_2_E)[:2, 3]
        J[:, 1] = (self.T_0_1 @ ht.rot_z(joints[0]) @ self.T_1_2 @
                   ht.d_translation(np.array([1, 0, 0])) @ self.T_2_E)[:2, 3]
        return J


class RobotRRR(RobotModel):
    """
    Model a robot with 3 degrees of freedom along different axis
    """
    def __init__(self):
        self.W = 0.05
        self.L0 = 1.0 + self.W/2
        self.L1 = 0.5
        self.L2 = 0.4
        self.L3 = 0.3 + self.W/2
        self.T_0_1 = ht.translation([0, 0, self.L0])
        self.T_1_2 = ht.translation([0, self.L1, 0])
        self.T_2_3 = ht.translation([0.0, self.L2, 0])
        self.T_3_E = ht.translation([0.0, self.L3, 0])

    def getJointsNames(self):
        return ["q1", "q2", "q3"]

    def getJointsLimits(self):
        return np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], dtype=np.double)

    def getOperationalDimensionNames(self):
        return ["x", "y", "z"]

    def getOperationalDimensionLimits(self):
        max_xy = self.L1 + self.L2 + self.L3
        min_z = self.L0 - self.L2 - self.L3
        max_z = self.L0 + self.L2 + self.L3
        return np.array([[-max_xy, max_xy], [-max_xy, max_xy], [min_z, max_z]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, q):
        tool_pos = self.getBaseFromToolTransform(q) @ np.array([0, 0, 0, 1])
        return tool_pos[:3]

    def analyticalMGI(self, target):
        # When X and Y of target are 'almost' zero, there is an infinity of solutions
        tol = 1e-9  # Only consider 1 solution if alpha is too small
        singularity = np.linalg.norm(target[:2]) < tol
        # First: use q0 to align target along y-axis:
        # - There's 2 potential solutions:
        theta = 0
        if not singularity:
            theta = math.atan2(target[1], target[0]) - np.pi/2
        solutions = []
        for q0 in [theta, theta + np.pi]:
            target_in_0 = np.zeros(4, dtype=np.double)
            target_in_0[:3] = target
            target_in_0[3] = 1
            # Put target in the proper referential:
            # only 2 rotations and 2 translations remaining
            target_in_2a = (ht.invert_transform(self.T_1_2) @ ht.rot_z(-q0) @
                            ht.invert_transform(self.T_0_1)  @ target_in_0)
            for q12 in cosineLaw(target_in_2a[1], target_in_2a[2], self.L2, self.L3):
                solutions.append(np.array([q0, q12[0], q12[1]]))
        if len(solutions) == 0:
            return 0, None
        if singularity:
            return -1, solutions[0]
        return len(solutions), solutions[0]

    def computeJacobian(self, joints):
        J = np.zeros((3, 3), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        J[:, 0] = (self.T_0_1 @ ht.d_rot_z(joints[0]) @ self.T_1_2 @
                   ht.rot_x(joints[1]) @ self.T_2_3 @ ht.rot_x(joints[2]) @
                   self.T_3_E)[:3, 3]
        J[:, 1] = (self.T_0_1 @ ht.rot_z(joints[0]) @ self.T_1_2 @
                   ht.d_rot_x(joints[1]) @ self.T_2_3 @ ht.rot_x(joints[2]) @
                   self.T_3_E)[:3, 3]
        J[:, 2] = (self.T_0_1 @ ht.rot_z(joints[0]) @ self.T_1_2 @
                   ht.rot_x(joints[1]) @ self.T_2_3 @ ht.d_rot_x(joints[2]) @
                   self.T_3_E)[:3, 3]
        return J


class LegRobot(RobotModel):
    """
    Model of a simple robot leg with 4 degrees of freedom
    """
    def __init__(self):
        self.W = 0.05
        self.L0 = 1.0 + self.W/2
        self.L1 = 0.5
        self.L2 = 0.3
        self.L3 = 0.3
        self.L4 = 0.2 + self.W/2
        self.T_0_1 = ht.translation([0, 0, self.L0])
        self.T_1_2 = ht.translation([self.W, self.L1, 0])
        self.T_2_3 = ht.translation([-self.W, self.L2, 0])
        self.T_3_4 = ht.translation([self.W, self.L3, 0])
        self.T_4_E = ht.translation([0, self.L4, 0])

    def getJointsNames(self):
        return ["q1", "q2", "q3", "q4"]

    def getJointsLimits(self):
        angle_lim = np.array([-np.pi, np.pi])
        L = np.zeros((4, 2))
        for d in range(4):
            L[d, :] = angle_lim
        return L

    def getOperationalDimensionNames(self):
        return ["x", "y", "z", "r32"]

    def getOperationalDimensionLimits(self):
        xy_max = math.sqrt((self.L1 + self.L2 + self.L3 + self.L4)**2 + self.W**2)
        z_offset = math.sqrt((self.L2 + self.L3 + self.L4)**2 + self.W**2)
        z_min = self.L0 - z_offset
        z_max = self.L0 + z_offset
        return np.array([[-xy_max, xy_max], [-xy_max, xy_max], [z_min, z_max], [-1, 1]])

    def getBaseFromToolTransform(self, joints):
        return (self.T_0_1 @ ht.rot_z(joints[0]) @
                self.T_1_2 @ ht.rot_x(joints[1]) @
                self.T_2_3 @ ht.rot_x(joints[2]) @
                self.T_3_4 @ ht.rot_x(joints[3]) @
                self.T_4_E)

    def extractMGD(self, T):
        """
        T : np.arraynd shape(4,4)
           An homogeneous transformation matrix
        """
        return np.append(T[:3, 3], T[2, 1])

    def computeMGD(self, joints):
        return self.extractMGD(self.getBaseFromToolTransform(joints))

    def analyticalMGI(self, target):
        solutions = []
        # Due to the link offset (W), elements near 'z-axis' are unreachable
        XY_norm = np.linalg.norm(target[:2])
        if XY_norm < self.W:
            return 0, None
        # q0 is the only element which can 'align' the direction of the tool
        # with respect to X,Y. Due to the link offset, it is more complex than
        # doing only atan2(Y,X)
        alpha = math.atan2(target[1], target[0]) - np.pi/2
        beta = math.atan2(self.W, XY_norm)
        # By symetry we have two solutions, note beta sign changing
        for q0 in [alpha + beta, np.pi + alpha - beta]:
            # In referential post q1, target should be in [0.02, Y_in_1, Z_in_1]
            target_pos_in_q0 = np.concatenate((target[:3], [1]))
            target_pos_in_q1 = ht.rot_z(-q0) @ ht.invert_transform(self.T_0_1) @ target_pos_in_q0
            Y_in_1 = target_pos_in_q1[1]
            Z_in_1 = target_pos_in_q1[2]
            # Now we have aligned the elements, we also know that:
            # sin(q1+q2+q3) = target[3] (aka r_3,2)
            alpha = math.asin(target[3])
            for q123 in [alpha, np.pi - alpha]:
                # Target origin of 3 in basis 1 is determined by q123
                Y3_in_1 = Y_in_1 - math.cos(q123) * self.L4
                Z3_in_1 = Z_in_1 - math.sin(q123) * self.L4
                q12_solutions = cosineLaw(Y3_in_1 - self.L1, Z3_in_1, self.L2, self.L3)
                for q12 in q12_solutions:
                    q3 = q123 - q12[0] - q12[1]
                    solutions.append([q0, q12[0], q12[1], q3])
        nb_sols = len(solutions)
        if nb_sols == 0:
            return 0, None
        return nb_sols, solutions[0]

    def computeJacobian(self, joints):
        J = np.zeros((4, 4), dtype=np.double)
        J[:, 0] = self.extractMGD(self.T_0_1 @ ht.d_rot_z(joints[0]) @
                                  self.T_1_2 @ ht.rot_x(joints[1]) @
                                  self.T_2_3 @ ht.rot_x(joints[2]) @
                                  self.T_3_4 @ ht.rot_x(joints[3]) @
                                  self.T_4_E)
        J[:, 1] = self.extractMGD(self.T_0_1 @ ht.rot_z(joints[0]) @
                                  self.T_1_2 @ ht.d_rot_x(joints[1]) @
                                  self.T_2_3 @ ht.rot_x(joints[2]) @
                                  self.T_3_4 @ ht.rot_x(joints[3]) @
                                  self.T_4_E)
        J[:, 2] = self.extractMGD(self.T_0_1 @ ht.rot_z(joints[0]) @
                                  self.T_1_2 @ ht.rot_x(joints[1]) @
                                  self.T_2_3 @ ht.d_rot_x(joints[2]) @
                                  self.T_3_4 @ ht.rot_x(joints[3]) @
                                  self.T_4_E)
        J[:, 3] = self.extractMGD(self.T_0_1 @ ht.rot_z(joints[0]) @
                                  self.T_1_2 @ ht.rot_x(joints[1]) @
                                  self.T_2_3 @ ht.rot_x(joints[2]) @
                                  self.T_3_4 @ ht.d_rot_x(joints[3]) @
                                  self.T_4_E)
        return J


def getRobotModel(robot_name):
    robot = None
    if robot_name == "RobotRT":
        robot = RobotRT()
    elif robot_name == "RobotRRR":
        robot = RobotRRR()
    elif robot_name == "LegRobot":
        robot = LegRobot()
    else:
        raise RuntimeError("Unknown robot name: '" + robot_name + "'")
    return robot
