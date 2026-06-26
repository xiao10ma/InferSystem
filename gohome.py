from Robot import BaseRobot

robot = BaseRobot.from_config("Config/piper.yaml")
robot.connect()
robot.enable()
print("before:", [round(x, 4) for x in robot.observe().joint_positions])
ok = robot.go_home(timeout_s=30)
print("go_home ok:", ok)
print("after:", [round(x, 4) for x in robot.observe().joint_positions])
robot.disconnect()