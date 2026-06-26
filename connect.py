from Robot import BaseRobot

robot = BaseRobot.from_config("Config/piper.yaml")
print("robot:", robot.robot_type, robot.name, "dof=", robot.dof)
robot.connect()
robot.enable()
state = robot.observe()
print("connected:", robot.is_connected())
print("operational:", robot.is_operational())
print("q14:", [round(x, 4) for x in state.joint_positions])
robot.disconnect()