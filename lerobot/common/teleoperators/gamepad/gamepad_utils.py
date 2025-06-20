#!/usr/bin/env python

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

import logging


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def should_quit(self):
        """Return True if the user has requested to quit."""
        return not self.running

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """Return True if ESC was pressed."""
        return self.key_states["quit"]

    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized gamepad: {self.joystick.get_name()}")

        print("Gamepad controls:")
        print("  Left stick: shoulder pan velocity (left/right), shoulder lift velocity (up/down)")
        print("  Right stick: elbow flex velocity (up/down), wrist roll velocity (left/right)")
        print("  D-pad up/down: wrist flex velocity")
        print("  L2/R2: gripper")
        print("  Note: Stick position determines movement speed - center to stop")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = "success"
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = "failure"
                # X button (0) for rerecord
                elif event.button == 0:
                    self.episode_end_status = "rerecord_episode"

                # RB button (6) for closing gripper
                elif event.button == 6:
                    self.close_gripper_command = True

                # LT button (7) for opening gripper
                elif event.button == 7:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [0, 2, 3]:
                    self.episode_end_status = None

                elif event.button == 6:
                    self.close_gripper_command = False

                elif event.button == 7:
                    self.open_gripper_command = False

            # Check for RB button (typically button 5) for intervention flag
            if self.joystick.get_button(5):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        if not self.joystick:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            shoulder_pan = self.joystick.get_axis(0)  # Left/Right on left stick
            shoulder_lift = self.joystick.get_axis(1)  # Up/Down on left stick

            # Right stick Y for elbow flex (axis 3) and X for wrist roll (axis 2)
            elbow_flex = -self.joystick.get_axis(3)  # Up/Down on right stick
            wrist_roll = -2.0 * self.joystick.get_axis(2)  # Left/Right on right stick (inverted and 2x speed)

            # Get wrist flex from D-pad (up/down)
            wrist_flex = 0.0
            if self.joystick.get_hat(0)[1] > 0:  # D-pad up
                wrist_flex = 1.0
            elif self.joystick.get_hat(0)[1] < 0:  # D-pad down
                wrist_flex = -1.0

            # Apply deadzone to joystick inputs
            shoulder_pan = 0 if abs(shoulder_pan) < self.deadzone else shoulder_pan
            shoulder_lift = 0 if abs(shoulder_lift) < self.deadzone else shoulder_lift
            elbow_flex = 0 if abs(elbow_flex) < self.deadzone else elbow_flex
            wrist_roll = 0 if abs(wrist_roll) < self.deadzone else wrist_roll

            # Return the joint movements
            return shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}
        self.quit_requested = False
        self.save_requested = False

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            # Check for known controller names and also "Wireless Controller" which is how PS4 controllers often appear
            if any(
                controller in device_name
                for controller in ["Logitech", "Xbox", "PS4", "PS5", "Wireless Controller"]
            ):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left stick: shoulder pan velocity (left/right), shoulder lift velocity (up/down)")
            logging.info("  Right stick: elbow flex velocity (up/down), wrist roll velocity (left/right)")
            logging.info("  D-pad up/down: wrist flex velocity")
            logging.info("  L2/R2: gripper")
            logging.info("  Note: Stick position determines movement speed - center to stop")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - handle both PS4 and other controllers
            if data and len(data) >= 8:
                if (
                    self.device_info is not None and self.device_info.get("vendor_id") == 0x054C
                ):  # Sony PS4 controller
                    # PS4 specific data format
                    # Left stick X and Y (0-255)
                    self.left_x = (data[1] - 128) / 128.0
                    self.left_y = (data[2] - 128) / 128.0
                    # Right stick X and Y (0-255)
                    self.right_x = (data[3] - 128) / 128.0
                    self.right_y = (data[4] - 128) / 128.0

                    # Apply deadzone
                    self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                    self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                    self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                    self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                    # PS4 buttons are in byte 5
                    buttons = data[5]

                    # PS4 specific button mapping
                    # Triangle (bit 7), Circle (bit 6), X (bit 5), Square (bit 4)
                    if buttons & (1 << 7):  # Triangle - success
                        self.episode_end_status = "success"
                    elif buttons & (1 << 5):  # X - failure
                        self.episode_end_status = "failure"
                    elif buttons & (1 << 4):  # Square - rerecord
                        self.episode_end_status = "rerecord_episode"
                    else:
                        self.episode_end_status = None

                    # L2/R2 triggers are in byte 8/9 (0-255)
                    self.close_gripper_command = data[8] > 128  # L2
                    self.open_gripper_command = data[9] > 128  # R2

                    # R1 for intervention (byte 6, bit 5)
                    self.intervention_flag = bool(buttons & (1 << 5))

                else:
                    # Original code for other controllers
                    # Normalize joystick values from 0-255 to -1.0-1.0
                    self.left_x = (data[1] - 128) / 128.0
                    self.left_y = (data[2] - 128) / 128.0
                    self.right_x = (data[3] - 128) / 128.0
                    self.right_y = (data[4] - 128) / 128.0

                    # Apply deadzone
                    self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                    self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                    self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                    self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                    # Parse button states (byte 5 in the Logitech RumblePad 2)
                    buttons = data[5]

                    # Check if RB is pressed then the intervention flag should be set
                    self.intervention_flag = data[6] in [2, 6, 10, 14]

                    # Check if RT is pressed
                    self.open_gripper_command = data[6] in [8, 10, 12]

                    # Check if LT is pressed
                    self.close_gripper_command = data[6] in [4, 6, 12]

                    # Check if Y/Triangle button (bit 7) is pressed for saving
                    # Check if X/Square button (bit 5) is pressed for failure
                    # Check if A/Cross button (bit 4) is pressed for rerecording
                    if buttons & 1 << 7:
                        self.episode_end_status = "success"
                    elif buttons & 1 << 5:
                        self.episode_end_status = "failure"
                    elif buttons & 1 << 4:
                        self.episode_end_status = "rerecord_episode"
                    else:
                        self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Return the joint movements
        # For HID controller, we need to adapt the controls similarly
        shoulder_pan = -self.left_x  # shoulder pan from left stick X
        shoulder_lift = -self.left_y  # shoulder lift from left stick Y
        elbow_flex = -self.right_y  # elbow flex from right stick Y
        wrist_roll = -2.0 * self.right_x  # wrist roll from right stick X (inverted and 2x speed)

        # Get wrist flex from D-pad (up/down)
        wrist_flex = 0.0
        if self.buttons.get("top_pad_pressed", False):  # D-pad up
            wrist_flex = 1.0
        elif self.buttons.get("bottom_pad_pressed", False):  # D-pad down
            wrist_flex = -1.0

        return shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested
