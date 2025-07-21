# robot_control_server.py
from flask import Flask, request, jsonify
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lerobot.common.robots import Robot, make_robot_from_config
from lerobot.common.robots.so100_follower import SO100FollowerConfig

app = Flask(__name__)

# Create a robot configuration - updated to match your calibrated robot
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem5A680097171",  # Same port as calibrated
    id="left_follower"  # Same ID as used by dual_teleoperation.py (matches calibration file)
)
robot = make_robot_from_config(robot_config)

@app.route("/execute", methods=["POST"])
def execute():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    action = data.get("action")

    try:
        if not robot.is_connected:
            robot.connect()
            
        if action == "move_to":
            position = data.get("position", [0, 0, 0, 0, 0, 0])  # 6 joint positions
            # Convert position to robot action format
            action_dict = {}
            motor_names = list(robot.action_features.keys())
            for i, pos in enumerate(position[:len(motor_names)]):
                action_dict[motor_names[i]] = float(pos)
            robot.send_action(action_dict)
        elif action == "grasp":
            # Close gripper
            action_dict = {"gripper.pos": 0.0}  # Fully closed
            robot.send_action(action_dict)
        elif action == "release":
            # Open gripper
            action_dict = {"gripper.pos": 100.0}  # Fully open
            robot.send_action(action_dict)
        else:
            return jsonify({"error": "Unknown action"}), 400
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
