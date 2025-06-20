import json
import time
from lerobot.common.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.common.motors.feetech import FeetechMotorsBus

# Load calibration data
with open(
    "/Users/pieterbecking/.cache/huggingface/lerobot/calibration/robots/so100_follower/dof-follower.json", "r"
) as f:
    calib_data = json.load(f)
    calibration = {k: MotorCalibration(**v) for k, v in calib_data.items()}

# Create motors dictionary
motors = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

# Initialize bus with calibration
bus = FeetechMotorsBus(port="/dev/tty.usbmodem5A680097171", motors=motors, calibration=calibration)

try:
    print("Connecting to bus...")
    bus.connect()
    print("Connected successfully")

    print("\nReading current positions...")
    positions = bus.sync_read("Present_Position")
    print(f"Current positions: {positions}")

    print("\nTrying to write a small movement...")
    # Try to move each motor slightly from its current position
    new_positions = {k: v + 5 for k, v in positions.items()}
    bus.sync_write("Goal_Position", new_positions)
    print("Movement command sent successfully")

    # Wait a moment and read positions again
    time.sleep(1)
    positions = bus.sync_read("Present_Position")
    print(f"\nNew positions after movement: {positions}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
finally:
    if "bus" in locals() and bus.is_connected:
        print("\nDisconnecting from bus...")
        bus.disconnect()
        print("Disconnected successfully")
