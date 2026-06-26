import time
from Core import Action, ActionSpace
from Robot import BaseRobot

robot = BaseRobot.from_config("Config/piper.yaml")
robot.connect()
robot.enable()

q = list(robot.observe().joint_positions)

# open both grippers
q[6] = 0.06
q[13] = 0.06
for _ in range(30):
    robot.act(Action(ActionSpace.JOINT_POSITION, q))
    time.sleep(1 / 30)

print("opened:", [round(x, 4) for x in robot.observe().joint_positions])

# close both grippers
q = list(robot.observe().joint_positions)
q[6] = 0.0
q[13] = 0.0
for _ in range(30):
    robot.act(Action(ActionSpace.JOINT_POSITION, q))
    time.sleep(1 / 30)

print("closed:", [round(x, 4) for x in robot.observe().joint_positions])
robot.disconnect()