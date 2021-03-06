#!/usr/bin/env python3

import numpy as np
import json
import math
import argparse
from abc import abstractmethod
import traceback

import robots


def buildTrajectory(type_name, start, knots, parameters=None):
    if type_name == "ConstantSpline":
        return ConstantSpline(knots, start)
    if type_name == "LinearSpline":
        return LinearSpline(knots, start)
    if type_name == "CubicZeroDerivativeSpline":
        return CubicZeroDerivativeSpline(knots, start)
    if type_name == "CubicWideStencilSpline":
        return CubicWideStencilSpline(knots, start)
    if type_name == "CubicCustomDerivativeSpline":
        return CubicCustomDerivativeSpline(knots, start)
    if type_name == "NaturalCubicSpline":
        return NaturalCubicSpline(knots, start)
    if type_name == "PeriodicCubicSpline":
        return PeriodicCubicSpline(knots, start)
    if type_name == "TrapezoidalVelocity":
        if parameters is None:
            raise RuntimeError("Parameters can't be None for TrapezoidalVelocity")
        return TrapezoidalVelocity(knots, parameters["vel_max"], parameters["acc_max"], start)
    raise RuntimeError("Unknown type: {:}".format(type_name))


def buildTrajectoryFromDictionary(dic):
    return buildTrajectory(dic["type_name"], dic["start"], np.array(dic["knots"]), dic.get("parameters"))


def buildRobotTrajectoryFromDictionary(dic):
    model = robots.getRobotModel(dic["model_name"])
    return RobotTrajectory(model, np.array(dic["targets"]), dic["trajectory_type"],
                           dic["target_space"], dic["planification_space"],
                           dic["start"], dic.get("parameters"))


def buildRobotTrajectoryFromFile(path):
    with open(path) as f:
        return buildRobotTrajectoryFromDictionary(json.load(f))


class Trajectory:
    """
    Describe a one dimension trajectory. Provides access to the value
    for any degree of the derivative

    Parameters
    ----------
    start: float
        The time at which the trajectory starts
    end: float or None
        The time at which the trajectory ends, or None if the trajectory never
        ends
    """
    def __init__(self, start=0):
        """
        The child class is responsible for setting the attribute end, if the
        trajectory is not periodic
        """
        self.start = start
        self.end = None

    @abstractmethod
    def getVal(self, t, d):
        """
        Computes the value of the derivative of order d at time t.

        Notes:
        - If $t$ is before (resp. after) the start (resp. the end) of the
         trajectory, returns:
          - The position at the start (resp. end) of the trajectory if d=0
          - 0 for any other value of $d$

        Parameters
        ----------
        t : float
            The time at which the position is requested
        d : int >= 0
            Order of the derivative. 0 to access position, 1 for speed,
            2 for acc, etc...

        Returns
        -------
        x : float
            The value of derivative of degree d at time t.
        """

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end


class Spline(Trajectory):
    """
    Attributes
    ----------
    knots : np.ndarray shape (N,2+)
        The list of timing for all the N via points:
        - Column 0 represents time points
        - Column 1 represents the position
        - Additional columns might be used to specify other elements
          (e.g derivative)
    coeffs : np.ndarray shape(N-1,K+1)
        A list of n-1 polynomials of degree $K$: The polynomial at slice $i$ is
        defined as follows: $S_i(t) = \\sum_{j=0}^{k}coeffs[i,j] * (t-t_i)^(k-j)$
    """

    def __init__(self, knots, start=0):
        super().__init__(start)
        self.knots = knots
        self.n = len(knots)
        self.coeffs = np.zeros((self.n-1, 4))
        self.end = self.knots[self.n-1, 0] + start
        self.updatePolynomials()

    @abstractmethod
    def updatePolynomials(self):
        """
        Updates the polynomials based on the knots and the interpolation method
        """

    def getDegree(self):
        """
        Returns
        -------
        d : int
            The degree of the polynomials used in this spline
        """
        raise NotImplementedError()

    def getPolynomial(self, t):
        """
        Parameters
        ----------
        t : float
           The time at which the polynomial is requested

        Returns
        -------
        adjusted_t : float
            Normalized time for the slice considered
        p : np.ndarray shape(k+1,)
            The coefficients of the polynomial at time t, see coeffs
        """

        t = t-self.start

        for k in range(self.n-1):
            if self.knots[k, 0] < t and self.knots[k+1, 0] > t:
                adjusted_t = t-self.knots[k, 0]
                p = self.coeffs[k]
                return adjusted_t , p
        return 0,0

    def getVal(self, t, d=0):
        if t <= self.start:
            if d == 0:
                return self.knots[0, 1]
            return 0
        elif t >= self.end:
            if d == 0:
                return self.knots[self.n-1, 1]
            return 0

        adjusted_t,p = self.getPolynomial(t)
        p1 = p.copy()
        
        for _ in range(d):
            p1[0] = p1[1]
            p1[1] = p1[2]*2
            p1[2] = p1[3]*3
            p1[3] = 0
        
        value = p1[0] + adjusted_t * p1[1] + adjusted_t**2 * p1[2] + adjusted_t**3 * p1[3]
        return value



class ConstantSpline(Spline):
    def updatePolynomials(self):
        for i in range(self.n-1):
            self.coeffs[i, 0] = self.knots[i, 1]
            self.coeffs[i, 1] = 0
            self.coeffs[i, 2] = 0
            self.coeffs[i, 3] = 0


class LinearSpline(Spline):
    def updatePolynomials(self):
        for i in range(self.n-1):
            x0 = self.knots[i, 1]
            x1 = self.knots[i+1, 1]
            t0 = self.knots[i, 0]
            t1 = self.knots[i+1, 0]
            
            self.coeffs[i, 0] = x0
            self.coeffs[i, 1] = (x1 - x0) / (t1 - t0)
            self.coeffs[i, 2] = 0
            self.coeffs[i, 3] = 0



class CubicZeroDerivativeSpline(Spline):
    """
    Update polynomials ensuring derivative is 0 at every knot.
    """

    def updatePolynomials(self):
        for i in range(self.n-1):
            x0 = self.knots[i, 1]
            x1 = self.knots[i+1, 1]
            t0 = self.knots[i, 0]
            t1 = self.knots[i+1, 0]
            
            delta_t = t1-t0
            A = np.array([
                [0, 0, 0, 1],
                [delta_t**3, delta_t**2, delta_t, 1],
                [0, 0, 1, 0],
                [3*delta_t**2, 2*delta_t, 1, 0]
            ])

            B = np.array([x0, x1, 0, 0])

            solutions = np.linalg.solve(A, B)

            self.coeffs[i, 0] = solutions[3]
            self.coeffs[i, 1] = solutions[2]
            self.coeffs[i, 2] = solutions[1]
            self.coeffs[i, 3] = solutions[0]
            # On a invers?? l'ordre des solutions car on pensait que c'??tait
            # a + bx + cx^2 + dx^3 alors que c'??tait ax^3 + bx^2 + cx + d



class CubicWideStencilSpline(Spline):
    """
    Update polynomials based on a larger neighborhood following the method 1
    described in http://www.math.univ-metz.fr/~croisil/M1-0809/2.pdf
    """

    def updatePolynomials(self):
        assert self.n >= 4

        for i in range(self.n-1):
            x,t = None, None
            if i == 0:                    
                x = self.knots[0:4, 1]
                t = self.knots[0:4, 0]
            elif i == self.n - 2:
                x = self.knots[i-2:i+2, 1]
                t = self.knots[i-2:i+2, 0]
            else:
                x = self.knots[i-1:i+3, 1]
                t = self.knots[i-1:i+3, 0]
            
            t = t - self.knots[i, 0]

            A = np.array([
                [t[0]**3, t[0]**2, t[0], 1],
                [t[1]**3, t[1]**2, t[1], 1],
                [t[2]**3, t[2]**2, t[2], 1],
                [t[3]**3, t[3]**2, t[3], 1],
            ])
            B = x

            solutions = np.linalg.solve(A, B)

            self.coeffs[i, 0] = solutions[3]
            self.coeffs[i, 1] = solutions[2]
            self.coeffs[i, 2] = solutions[1]
            self.coeffs[i, 3] = solutions[0]



class CubicCustomDerivativeSpline(Spline):
    """
    For this splines, user is requested to specify the velocity at every knot.
    Therefore, knots is of shape (N,3)
    """
    def updatePolynomials(self):
        assert self.knots.shape[1] >= 3

        for i in range(self.n-1):
            x0 = self.knots[i, 1]
            x1 = self.knots[i+1, 1]
            t0 = self.knots[i, 0]
            t1 = self.knots[i+1, 0]
            v0 = self.knots[i, 2]
            v1 = self.knots[i+1, 2]
            
            delta_t = t1-t0
            A = np.array([
                [0, 0, 0, 1],
                [delta_t**3, delta_t**2, delta_t, 1],
                [0, 0, 1, 0],
                [3*delta_t**2, 2*delta_t, 1, 0]
            ])

            B = np.array([x0, x1, v0, v1])

            solutions = np.linalg.solve(A, B)

            self.coeffs[i, 0] = solutions[3]
            self.coeffs[i, 1] = solutions[2]
            self.coeffs[i, 2] = solutions[1]
            self.coeffs[i, 3] = solutions[0]


class NaturalCubicSpline(Spline):
    def updatePolynomials(self):
        
        A = np.zeros((4 * (self.n-1), 4 * (self.n-1)))
        B = np.zeros(4 * (self.n-1))

        for i in range(self.n-1):
            x0 = self.knots[i, 1]
            x1 = self.knots[i+1, 1]
            t0 = self.knots[i, 0]
            t1 = self.knots[i+1, 0]
            
            delta_t = t1-t0

            start = 4*i
            end = start + 4

            A[start, start:end] = [0, 0, 0, 1]
            A[start+1, start:end] = [delta_t**3, delta_t**2, delta_t, 1]
            if i < self.n-2:
                A[start+2, start:end] = [3*delta_t**2, 2*delta_t, 1, 0]
                A[start+2, end:end+4] = [0, 0, -1, 0]
                A[start+3, start:end] = [6*delta_t, 2, 0, 0]
                A[start+3, end:end+4] = [0, -2, 0, 0]

            B[4*i] = x0
            B[4*i+1] = x1
        
        # contraintes suppl??mentaires
        A[-2, 1] = 2
        delta_t = self.knots[-1,0] - self.knots[-2,0]
        A[-1, -4:-2] = [6*delta_t, 2]

        coeffs = np.linalg.solve(A,B)

        for i in range(self.n-1):
            self.coeffs[i, 0] = coeffs[4*i+3]
            self.coeffs[i, 1] = coeffs[4*i+2]
            self.coeffs[i, 2] = coeffs[4*i+1]
            self.coeffs[i, 3] = coeffs[4*i]


class PeriodicCubicSpline(Spline):
    """
    Describe global splines where position, 1st order derivative and second
    derivative are always equal on both sides of a knot. This i
    """
    def updatePolynomials(self):
        A = np.zeros((4 * (self.n-1), 4 * (self.n-1)))
        B = np.zeros(4 * (self.n-1))

        for i in range(self.n-1):
            x0 = self.knots[i, 1]
            x1 = self.knots[i+1, 1]
            t0 = self.knots[i, 0]
            t1 = self.knots[i+1, 0]
            
            delta_t = t1-t0

            start = 4*i
            end = start + 4

            A[start, start:end] = [0, 0, 0, 1]
            A[start+1, start:end] = [delta_t**3, delta_t**2, delta_t, 1]
            if i < self.n-2:
                A[start+2, start:end] = [3*delta_t**2, 2*delta_t, 1, 0]
                A[start+2, end:end+4] = [0, 0, -1, 0]
                A[start+3, start:end] = [6*delta_t, 2, 0, 0]
                A[start+3, end:end+4] = [0, -2, 0, 0]

            B[4*i] = x0
            B[4*i+1] = x1
        
        # contraintes suppl??mentaires
        delta_t = self.knots[-1,0] - self.knots[-2,0]
        A[-2, 1] = 2
        A[-2, -4:-2] = [-6*delta_t, -2]
        A[-1, 2] = 1
        A[-1, -4:] = [-3*delta_t**2, -2*delta_t, -1, 0]

        coeffs = np.linalg.solve(A,B)

        for i in range(self.n-1):
            self.coeffs[i, 0] = coeffs[4*i+3]
            self.coeffs[i, 1] = coeffs[4*i+2]
            self.coeffs[i, 2] = coeffs[4*i+1]
            self.coeffs[i, 3] = coeffs[4*i]


    def getVal(self, t, d=0):
        D = self.end - self.start
        return super().getVal(self.start + (t-self.start)%D, d)


class TrapezoidalVelocity(Trajectory):
    def __init__(self, knots, vMax, accMax, start):
        super().__init__(start)
        self.x_src = knots[0]
        self.x_end = knots[1]
        self.D = self.x_end - self.x_src
        self.vMax = vMax
        self.accMax = accMax

        if abs(self.D) > (vMax*vMax) / accMax:
            self.Tacc = vMax / accMax
        else:
            self.Tacc = np.sqrt(abs(self.D) / accMax)
        
        self.Dacc = (self.accMax * self.Tacc * self.Tacc) / 2

        self.end = self.start + 2*self.Tacc + (abs(self.D) - 2*self.Dacc) / vMax

    def getVal(self, t, d):
        if d < 0 or d > 2:
            return 0
        
        if t < self.start:
            if d == 0: return self.x_src
            return 0
        if t > self.end:
            if d == 0: return self.x_end
            return 0

        D_sign = np.sign(self.D)
        T = self.end - self.start

        if t <= self.Tacc:
            if d == 0: return self.x_src + D_sign * (self.accMax*t*t)/2
            if d == 1: return D_sign * self.accMax * t
            return D_sign * self.accMax

        elif t > T - self.Tacc:
            if d == 0: return self.x_end - D_sign * (self.accMax * (T-t) * (T-t))/2
            if d == 1: return D_sign * self.accMax * (T-t)
            return - D_sign * self.accMax

        else:
            if d == 0: return self.x_src + D_sign * (self.Dacc + self.vMax * (t-self.Tacc))
            if d == 1: return D_sign * self.vMax
            return 0
        


class RobotTrajectory:
    """
    Represents a multi-dimensional trajectory for a robot.

    Attributes
    ----------
    model : control.RobotModel
        The model used for the robot
    planification_space : str
        Two space in which trajectories are planified: 'operational' or 'joint'
    trajectories : list(Trajectory)
        One trajectory per dimension of the planification space
    """

    supported_spaces = ["operational", "joint"]

    def __init__(self, model, targets, trajectory_type,
                 target_space, planification_space,
                 start=0, parameters=None):
        """
        model : robots.RobotModel
            The model of the robot concerned by this trajectory
        targets : np.ndarray shape(m,n) or shape(m,n+1)
            The multi-dimensional knots for the trajectories. One row concerns one
            target. Each column concern one of the dimension of the target space.
            For trajectories with specified time points (e.g. splines), the first
            column indicates time point.
        target_space : str
            The space in which targets are provided: 'operational' or 'joint'
        trajectory_type : str
            The name of the class to be used with trajectory
        planification_space : str
            The space in which trajectories are defined: 'operational' or 'joint'
        start : float
            The start of the trajectories [s]
        parameters : dictionary or None
            A dictionary containing extra-parameters for trajectories
        """
        
        assert (target_space in self.supported_spaces) and (planification_space in self.supported_spaces)

        self.model = model
        self.planification_space = planification_space
        self.trajectories = []
        self.start = start
        self.end = start

        target_len, target_dim = targets.shape

        # Pour savoir si les cibles contiennent l'information de temps,
        # on r??cup??re le nombre de dimensions dans lesquelles op??re le robot
        self.nb_dim = len(model.getOperationalDimensionNames())
        if target_space == "joint":
            self.nb_dim = model.getNbJoints()     
        time_info = False
        if target_dim == self.nb_dim+1:
            time_info = True

        if target_space != self.planification_space:
            for i in range(target_len):
                begin = 1 if time_info else 0
                target = targets[i, begin:]
                if planification_space == "joint":
                    targets[i, begin:] = self.model.analyticalMGI(target)[1]
                else:
                    targets[i, begin:] = self.model.computeMGD(target)
        
        # Pour chaque dimension, on calcule la trajectoire qu'on ajoute ?? la liste des trajectoires
        # Si n??cessaire, on met aussi ?? jour self.end
        for i in range(self.nb_dim):
            if time_info:
                knots = np.zeros((target_len, 2))
                knots[:, 0] = targets[:, 0]
                knots[:, 1] = targets[:, i+1]
            else:            
                knots = targets[:, i]
                
            traj = buildTrajectory(trajectory_type, start, knots, parameters)
            self.trajectories.append(traj)
            if traj.getEnd() > self.end:
                self.end = traj.getEnd()
        


    def getVal(self, t, dim, degree, space):
        """
        Parameters
        ----------
        t : float
            The time at which the value is requested
        dim : int
            The dimension index
        degree : int
            The degree of the derivative requested (0 means position)
        space : str
            The space in which the value is requested: 'operational' or 'joint'

        Returns
        -------
        v : float or None
            The value of derivative of order degree at time t on dimension dim
            of the chosen space, None if computation is not implemented or fails
        """

        if space == self.planification_space:
            return self.trajectories[dim].getVal(t, degree)
        
        value = None
        if space == "operational":            
            if degree == 0: value = self.getOperationalTarget(t)
            elif degree == 1: value = self.getOperationalVelocity(t)
            elif degree == 2: value = self.getOperationalAcc(t)
        elif space == "joint":
            if degree == 0: value = self.getJointTarget(t)
            elif degree == 1: value = self.getJointVelocity(t)
            elif degree == 2: value = self.getJointAcc(t)
        
        if value is not None:
            return value[dim]
        return None



    def getPlanificationVal(self, t, degree):
        value = []
        for i in range(len(self.trajectories)):
            value.append(self.trajectories[i].getVal(t, degree))
        return value

    def getOperationalTarget(self, t):
        value = self.getPlanificationVal(t, 0)
        if self.planification_space == "operational":
            return value
        return self.model.computeMGD(value)

    def getJointTarget(self, t):
        value = self.getPlanificationVal(t, 0)
        if self.planification_space == "joint":
            return value
        return self.model.analyticalMGI(value)[1]

    def getOperationalVelocity(self, t):
        value = self.getPlanificationVal(t, 1)
        if self.planification_space == "operational":
            return value
        return self.model.computeJacobian(self.getJointTarget(t)) @ value

    def getJointVelocity(self, t):
        value = self.getPlanificationVal(t, 1)
        if self.planification_space == "joint":
            return value        
        return np.linalg.pinv(self.model.computeJacobian(self.getJointTarget(t))) @ value

    def getOperationalAcc(self, t):
        value = self.getPlanificationVal(t, 2)
        if self.planification_space == "operational":
            return value
        return self.model.computeMGD(value)

    def getJointAcc(self, t):
        value = self.getPlanificationVal(t, 2)
        if self.planification_space == "joint":
            return value
        return self.model.analyticalMGI(value)[1]

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--robot", help="Consider robot trajectories and not 1D trajectories",
                        action="store_true")
    parser.add_argument("--degrees",
                        type=lambda s: np.array([int(item) for item in s.split(',')]),
                        default=[0, 1, 2],
                        help="The degrees of derivative to plot")
    parser.add_argument("trajectories", nargs="+", type=argparse.FileType('r'))
    args = parser.parse_args()
    trajectories = {}
    tmax = 0
    tmin = 10**10
    for t in args.trajectories:
        try:
            if args.robot:
                trajectories[t.name] = buildRobotTrajectoryFromDictionary(json.load(t))
            else:
                trajectories[t.name] = buildTrajectoryFromDictionary(json.load(t))
            tmax = max(tmax, trajectories[t.name].getEnd())
            tmin = min(tmin, trajectories[t.name].getStart())
        except KeyError:
            print("Error while building trajectory from file {:}:\n{:}".format(t.name, traceback.format_exc()))
            exit()
    order_names = ["position", "velocity", "acceleration", "jerk"]
    print("source,t,order,variable,value")
    for source_name, trajectory in trajectories.items():
        for t in np.arange(tmin - args.margin, tmax + args.margin, args.dt):
            for degree in args.degrees:
                order_name = order_names[degree]
                if (args.robot):
                    space_dims = {
                        "joint": trajectory.model.getJointsNames(),
                        "operational": trajectory.model.getOperationalDimensionNames()
                    }
                    for space, dim_names in space_dims.items():
                        for dim in range(len(dim_names)):
                            v = trajectory.getVal(t, dim, degree, space)
                            if v is not None:
                                print("{:}, {:}, {:}, {:}, {:}".format(source_name, t, order_name, dim_names[dim], v))
                else:
                    v = trajectory.getVal(t, degree)
                    print("{:}, {:}, {:}, {:}, {:}".format(source_name, t, order_name, "x", v))
