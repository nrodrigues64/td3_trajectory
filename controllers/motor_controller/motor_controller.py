"""
motor_controller controller

A simple controller to set the target of a robot
"""

import csv
import os
from controller import Robot
import json
import numpy as np

import robots
import trajectories

ALLOWED_MODES = ['computeMGD', 'analyticalMGI', 'jacobianInverse', 'jacobianTransposed']
DEFAULT_CONTROL_MODE = 'analyticalMGI'  # The mode used for the simulation see ALLOWED_MODES
DEFAULT_ROBOT = 'RobotRRR'
DEFAULT_TRAJECTORY_PATH = "robot_trajectories/rrr_periodic_joint.json"

# Setting parameters from environment variables
control_mode = os.environ.get('CONTROL_MODE', DEFAULT_CONTROL_MODE)
robot_name = os.environ.get('ROBOT_NAME', DEFAULT_ROBOT)
trajectory_path = os.environ.get('TRAJECTORY_PATH', DEFAULT_TRAJECTORY_PATH)

trajectory = None
with open(trajectory_path) as f:
    trajectory = trajectories.buildRobotTrajectoryFromDictionary(json.load(f))

model = robots.getRobotModel(robot_name)
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initializing motors
motors = []
for name in model.getMotorsNames():
    motors.append(robot.getDevice(name))

# Initializing sensors
sensors = []
for name in model.getSensorsNames():
    sensors.append(robot.getDevice(name))
    sensors[-1].enable(timestep)

with open('robot_data.csv', 'w') as output_file:
    logger = csv.DictWriter(output_file, ["t", "variable", "order", "source", "value"])
    logger.writeheader()

    t = 0.0  # [s]
    last_op_pos = None
    last_q = None
    while robot.step(timestep) != -1:
        nb_joints = model.getNbJoints()
        q = np.zeros(nb_joints)
        target_op_pos = None
        target_q = trajectory.getJointTarget(t)
        target_op = trajectory.getOperationalTarget(t)
        target_q_vel = trajectory.getJointVelocity(t)
        target_op_vel = trajectory.getOperationalVelocity(t)
        target_q_acc = trajectory.getJointAcc(t)
        target_op_acc = trajectory.getOperationalAcc(t)
        # Setting motors and writing joints target + measurements
        for i in range(nb_joints):
            if target_q is not None:
                motors[i].setPosition(target_q[i])
            q[i] = sensors[i].getValue()
        # Measuring and writing operational target
        op_pos = model.computeMGD(q)
        # Simple measure of average speed with respect to last pos, note that it 'works' only because the sensor has
        # infinite resolution
        op_vel = None if last_op_pos is None else (op_pos-last_op_pos) * 1000.0 / timestep
        q_vel = None if last_q is None else (q-last_q) * 1000.0 / timestep
        log_values = {
            "operational": {
                "sensors": {0: op_pos, 1: op_vel},
                "target": {0: target_op, 1: target_op_vel, 2: target_op_acc}
            },
            "joint": {
                "sensors": {0: q, 1: q_vel},
                "target": {0: target_q, 1: target_q_vel, 2: target_q_acc}
            },
        }
        joints_names = model.getJointsNames()
        op_dim_names = model.getOperationalDimensionNames()
        for space, prop1 in log_values.items():
            names = joints_names if space == 'joint' else op_dim_names
            for source, prop2 in prop1.items():
                for order, value in prop2.items():
                    if value is None:
                        continue
                    for idx in range(len(names)):
                        name = names[idx]
                        logger.writerow({"t": t, "variable": name, "order": order, "source": source, "value": value[idx]})
        t += timestep / 10**3
        last_op_pos = op_pos
        last_q = q
