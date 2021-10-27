"""
supervisor controller

spawn a robot and controls the simulation
"""

import csv
import os
from controller import Supervisor, Node

DEFAULT_ROBOT = 'RobotRRR'
DEFAULT_DURATION = 10.0    # [s]
DEFAULT_AUTO_EXIT = 0      # Simulation is killed when duration is reached [0: off, 1: on]
DEFAULT_AUTO_STOP = 1      # Simulation is paused when duration is reached [0: off, 1: on]


def spawn_robot(supervisor, proto_name, def_name, robot_name):
    node = supervisor.getRoot().getField('children')
    import_msg = f'DEF {def_name} {proto_name} {{name "{robot_name}"}}'
    node.importMFNodeFromString(-1, import_msg)
    print(f'Spawning robot {robot_name} with "{import_msg}"')
    return supervisor.getFromDef(def_name)


def find_tooltip(solid):
    name_field = solid.getField('name')
    if name_field:
        if name_field.getSFString() == "tooltip":
            return solid
    children = solid.getProtoField('children') if solid.isProto() else solid.getField('children')
    tooltip = None
    for i in range(children.getCount()):
        if tooltip is not None:
            break
        child = children.getMFNode(i)
        if child.getType() in [Node.ROBOT, Node.SOLID, Node.GROUP, Node.TRANSFORM, Node.ACCELEROMETER, Node.CAMERA, Node.GYRO,
                               Node.TOUCH_SENSOR]:
            tooltip = find_tooltip(child)
            continue
        if child.getType() in [Node.HINGE_JOINT, Node.HINGE_2_JOINT, Node.SLIDER_JOINT, Node.BALL_JOINT]:
            endPoint = child.getProtoField('endPoint') if child.isProto() else child.getField('endPoint')
            solid = endPoint.getSFNode()
            if solid.getType() == Node.NO_NODE or solid.getType() == Node.SOLID_REFERENCE:
                continue
            tooltip = find_tooltip(solid)
    return tooltip


def logData(tooltip, logger):
    tool_pos = tooltip.getPosition()
    tool_vel = tooltip.getVelocity()[:3]  # Only using cartesian velocity
    for name, idx in {"x": 0, "y": 1, "z": 2}.items():
        logger.writerow({"t": t, "variable": name, "order": 0, "source": "supervisor", "value": tool_pos[idx]})
        logger.writerow({"t": t, "variable": name, "order": 1, "source": "supervisor", "value": tool_vel[idx]})


# Setting parameters from environment variables
robot_name = os.environ.get('ROBOT_NAME', DEFAULT_ROBOT)
sim_duration = float(os.environ.get('SIM_DURATION', DEFAULT_DURATION))
auto_exit = int(os.environ.get('SIM_AUTO_EXIT', DEFAULT_AUTO_EXIT)) != 0
auto_stop = int(os.environ.get('SIM_AUTO_STOP', DEFAULT_AUTO_STOP)) != 0

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
t = 0

robot = spawn_robot(supervisor, robot_name, 'ROBOT', "robot")
simulator_tooltip = find_tooltip(robot)
if simulator_tooltip is None:
    raise RuntimeError("Tooltip not found in robot")

with open('simulator_data.csv', 'w') as output_file:
    logger = csv.DictWriter(output_file, ["t", "variable", "order", "source", "value"])
    logger.writeheader()
    while supervisor.step(timestep) != -1:
        logData(simulator_tooltip, logger)
        # Handling simulation duration
        if t >= sim_duration:
            if auto_exit:
                supervisor.simulationQuit(0)
                break
            elif auto_stop:
                supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
                auto_stop = False  # Only pausing the simulation once.
        t += timestep / 1000.0
