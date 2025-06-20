# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .configuration_gamepad import GamepadTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    The joystick positions determine the velocity of movement rather than absolute position.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.gamepad = None
        # Current joint positions
        self.current_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
        }

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn't reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        if self.gamepad is None:
            raise RuntimeError("Gamepad is not initialized. Did you call connect()?")

        self.gamepad.update()

        # Get velocity commands from the controller
        shoulder_pan_vel, shoulder_lift_vel, elbow_flex_vel, wrist_flex_vel, wrist_roll_vel = (
            self.gamepad.get_deltas()
        )

        # Scale factor for velocity (adjust these values to tune the control sensitivity)
        VELOCITY_SCALE = 0.3  # This makes the movement more manageable

        # Update positions based on velocity
        self.current_positions["shoulder_pan"] += -shoulder_pan_vel * VELOCITY_SCALE
        self.current_positions["shoulder_lift"] += shoulder_lift_vel * VELOCITY_SCALE
        self.current_positions["elbow_flex"] += elbow_flex_vel * VELOCITY_SCALE
        self.current_positions["wrist_flex"] += wrist_flex_vel * VELOCITY_SCALE
        self.current_positions["wrist_roll"] += wrist_roll_vel * VELOCITY_SCALE

        # Create action dictionary with updated positions
        action_dict = {
            "shoulder_pan.pos": self.current_positions["shoulder_pan"],
            "shoulder_lift.pos": self.current_positions["shoulder_lift"],
            "elbow_flex.pos": self.current_positions["elbow_flex"],
            "wrist_flex.pos": self.current_positions["wrist_flex"],
            "wrist_roll.pos": self.current_positions["wrist_roll"],
        }

        # Handle gripper control
        if self.config.use_gripper:
            if self.gamepad.open_gripper_command:
                action_dict["gripper.pos"] = 100.0  # Fully open
            elif self.gamepad.close_gripper_command:
                action_dict["gripper.pos"] = 0.0  # Fully closed
            else:
                action_dict["gripper.pos"] = 50.0  # Stay in current position

        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # Reset current positions
        self.current_positions = {k: 0.0 for k in self.current_positions}

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
