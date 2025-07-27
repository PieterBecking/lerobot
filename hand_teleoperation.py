#!/usr/bin/env python3

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Hand Tracking Teleoperation System

This script provides computer vision-based teleoperation using hand tracking to control robot arms.
Uses MediaPipe Hands for real-time hand detection and maps hand movements to robot joint commands.

Phase 1: Core hand tracking with visual feedback
Phase 2: Hand-to-robot mapping  
Phase 3: Dual arm integration with existing robots

Usage:
    # Test hand tracking only (Phase 1)
    python hand_teleoperation.py --test-tracking --camera-index 0
    
    # Run single arm teleoperation (Phase 2)
    python hand_teleoperation.py --teleop --single-arm --ports-config dual_so100_ports.json
    
    # Run dual arm teleoperation (Phase 3) 
    python hand_teleoperation.py --teleop --dual-arm --ports-config dual_so100_ports.json

Example:
    python hand_teleoperation.py --test-tracking --camera-index 0 --fps 30
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from lerobot.common.teleoperators.config import TeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.common.utils.utils import init_logging
from lerobot.common.robots import make_robot_from_config  
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig


@dataclass
class HandTrackingConfig(TeleoperatorConfig):
    """Configuration for hand tracking teleoperator."""
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    confidence_threshold: float = 0.5
    tracking_confidence: float = 0.5
    max_num_hands: int = 2
    model_complexity: int = 1  # 0 (lite), 1 (full)
    min_detection_confidence: float = 0.5
    
    # Robot mapping parameters
    workspace_x_range: Tuple[float, float] = (0.2, 0.8)  # Hand X position range (normalized)
    workspace_y_range: Tuple[float, float] = (0.2, 0.8)  # Hand Y position range (normalized)
    workspace_z_range: Tuple[float, float] = (0.1, 0.5)  # Hand Z depth range (normalized)
    
    # Robot joint ranges (typical Dynamixel values)
    joint_min: float = 512.0    # Minimum joint position
    joint_max: float = 3583.0   # Maximum joint position
    joint_center: float = 2048.0  # Center joint position
    gripper_open: float = 100.0   # Gripper open position (SO100: 100 = open)
    gripper_closed: float = 0.0   # Gripper closed position (SO100: 0 = closed)
    
    # Movement smoothing and safety
    smoothing_factor: float = 0.9  # Higher smoothing for joints safety
    gripper_smoothing_factor: float = 0.3  # Lower smoothing for responsive gripper
    max_joint_change_per_frame: float = 50.0  # Maximum change per frame for safety
    max_gripper_change_per_frame: float = 30.0  # Higher limit for faster gripper response


class HandTrackingTeleoperator(Teleoperator):
    """
    Hand tracking based teleoperator using MediaPipe Hands.
    
    Detects hand landmarks and maps them to robot joint commands.
    Supports both single and dual hand tracking.
    """
    
    config_class = HandTrackingConfig
    name = "hand_tracking"
    
    def __init__(self, config: HandTrackingConfig):
        super().__init__(config)
        self.config = config
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand tracking model
        self.hands = None
        
        # Camera setup
        self.camera = None
        self.is_camera_connected = False
        
        # Hand tracking state
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.left_hand_center = None
        self.right_hand_center = None
        self.left_hand_openness = 0.0  # Changed from boolean to float (0.0 = closed, 1.0 = open)
        self.right_hand_openness = 0.0  # Changed from boolean to float (0.0 = closed, 1.0 = open)
        self.left_hand_depth = 0.0
        self.right_hand_depth = 0.0
        self.left_wrist_angle = 0.0
        self.right_wrist_angle = 0.0
        
        # Smoothed robot actions for stability - will be initialized with current robot position
        self.smoothed_action = None  # For single arm mode compatibility
        self.left_smoothed_action = None  # For dual arm mode - left arm
        self.right_smoothed_action = None  # For dual arm mode - right arm
        self.current_robot_position = None
        
        # Frame info
        self.frame_width = config.camera_width
        self.frame_height = config.camera_height
        
        logging.info(f"Initialized {self} with camera index {config.camera_index}")

    @property
    def action_features(self) -> dict:
        """Actions for a 6-DOF robot arm: 5 joints + gripper."""
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float, 
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        """No feedback features for hand tracking."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if camera and MediaPipe are connected."""
        return self.is_camera_connected and self.hands is not None

    @property
    def is_calibrated(self) -> bool:
        """Hand tracking doesn't require calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Connect to camera and initialize MediaPipe."""
        if self.is_connected:
            raise Exception(f"{self} is already connected")
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence,
            model_complexity=self.config.model_complexity
        )
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.config.camera_index)
        if not self.camera.isOpened():
            raise Exception(f"Failed to open camera {self.config.camera_index}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = actual_width
        self.frame_height = actual_height
        
        self.is_camera_connected = True
        logging.info(f"{self} connected - Camera: {actual_width}x{actual_height}")

    def calibrate(self) -> None:
        """No calibration needed for hand tracking."""
        pass

    def get_robot_status(self) -> dict:
        """Get current robot position and safety status for debugging."""
        if self.current_robot_position is None:
            return {"status": "not_initialized", "positions": None}
        
        return {
            "status": "initialized",
            "current_positions": self.current_robot_position.copy(),
            "smoothed_positions": self.smoothed_action.copy() if self.smoothed_action else None,
            "safety_limits": {
                "max_joint_change": self.config.max_joint_change_per_frame,
                "max_gripper_change": self.config.max_gripper_change_per_frame,
                "joint_smoothing_factor": self.config.smoothing_factor,
                "gripper_smoothing_factor": self.config.gripper_smoothing_factor
            }
        }

    def configure(self) -> None:
        """No configuration needed for hand tracking."""
        pass

    def initialize_with_robot_position(self, robot):
        """
        SAFETY: Initialize smoothed actions with current robot position.
        This prevents sudden movements when starting teleoperation.
        """
        try:
            # Get current robot position
            current_obs = robot.get_observation()
            self.current_robot_position = {
                key: value for key, value in current_obs.items() 
                if key.endswith('.pos') and not key.startswith('camera')
            }
            
            # Initialize smoothed action with current position
            self.smoothed_action = self.current_robot_position.copy()
            
            logging.info(f"Initialized hand tracker with current robot position: {self.current_robot_position}")
            print(f"✅ SAFETY: Hand tracker initialized with current robot position")
            
        except Exception as e:
            logging.error(f"Failed to read current robot position: {e}")
            # Fallback to center positions if we can't read current position
            self.smoothed_action = {
                "shoulder_pan.pos": self.config.joint_center,
                "shoulder_lift.pos": self.config.joint_center,
                "elbow_flex.pos": self.config.joint_center,
                "wrist_flex.pos": self.config.joint_center,
                "wrist_roll.pos": self.config.joint_center,
                "gripper.pos": self.config.gripper_open,
            }
            print(f"⚠️  Could not read robot position, using fallback center positions")

    def initialize_dual_arm_with_robot_positions(self, left_robot, right_robot):
        """
        SAFETY: Initialize smoothed actions for dual arm mode with current robot positions.
        This prevents sudden movements when starting teleoperation.
        """
        try:
            # Get current left robot position
            left_obs = left_robot.get_observation()
            left_position = {
                key: value for key, value in left_obs.items() 
                if key.endswith('.pos') and not key.startswith('camera')
            }
            
            # Get current right robot position  
            right_obs = right_robot.get_observation()
            right_position = {
                key: value for key, value in right_obs.items() 
                if key.endswith('.pos') and not key.startswith('camera')
            }
            
            # Initialize separate smoothed actions
            self.left_smoothed_action = left_position.copy()
            self.right_smoothed_action = right_position.copy()
            
            logging.info(f"Initialized hand tracker with LEFT robot position: {left_position}")
            logging.info(f"Initialized hand tracker with RIGHT robot position: {right_position}")
            print(f"✅ SAFETY: Hand tracker initialized with current positions for both arms")
            
        except Exception as e:
            logging.error(f"Failed to read current robot positions: {e}")
            # Fallback to center positions if we can't read current positions
            fallback_position = {
                "shoulder_pan.pos": self.config.joint_center,
                "shoulder_lift.pos": self.config.joint_center,
                "elbow_flex.pos": self.config.joint_center,
                "wrist_flex.pos": self.config.joint_center,
                "wrist_roll.pos": self.config.joint_center,
                "gripper.pos": self.config.gripper_open,
            }
            self.left_smoothed_action = fallback_position.copy()
            self.right_smoothed_action = fallback_position.copy()
            print(f"⚠️  Could not read robot positions, using fallback center positions for both arms")

    def _detect_hands(self, frame: np.ndarray) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Detect hands in the frame and return left/right hand landmarks.
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Tuple of (left_hand_landmarks, right_hand_landmarks)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        if self.hands is None:
            return None, None
        results = self.hands.process(rgb_frame)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                if hand_label == "Left":
                    left_hand = hand_landmarks
                elif hand_label == "Right":
                    right_hand = hand_landmarks
        
        return left_hand, right_hand

    def _calculate_hand_center(self, landmarks) -> Tuple[Optional[float], Optional[float]]:
        """Calculate the center point of the hand."""
        if landmarks is None:
            return None, None
        
        # Use wrist (landmark 0) as reference point
        wrist = landmarks.landmark[0]
        center_x = wrist.x * self.frame_width
        center_y = wrist.y * self.frame_height
        
        return center_x, center_y

    def _calculate_hand_depth(self, landmarks) -> float:
        """Calculate hand depth/distance from camera using hand size."""
        if landmarks is None:
            return 0.0
        
        # Use distance between wrist and middle finger tip as depth indicator
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        # Calculate Euclidean distance (normalized)
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        dz = middle_tip.z - wrist.z if hasattr(middle_tip, 'z') else 0.0
        
        hand_size = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Larger hand size = closer to camera, smaller = further away
        # Invert and normalize (typical range: 0.1-0.3 → 0.0-1.0)
        depth = max(0.0, min(1.0, (0.3 - hand_size) / 0.2))
        
        return depth

    def _calculate_wrist_angle(self, landmarks) -> float:
        """Calculate wrist rotation angle from hand landmarks."""
        if landmarks is None:
            return 0.0
        
        # Use wrist to middle finger base vector for orientation
        wrist = landmarks.landmark[0]
        middle_base = landmarks.landmark[9]
        
        # Calculate angle in radians, then convert to normalized value
        dx = middle_base.x - wrist.x
        dy = middle_base.y - wrist.y
        
        angle_rad = np.arctan2(dy, dx)
        # Normalize to 0-1 range
        angle_normalized = (angle_rad + np.pi) / (2 * np.pi)
        
        return angle_normalized

    def _normalize_to_joint_range(self, normalized_value: float) -> float:
        """Convert normalized value (0-1) to robot joint range."""
        return (self.config.joint_min + 
                normalized_value * (self.config.joint_max - self.config.joint_min))

    def _map_hand_to_robot_action(self, use_left_hand: bool = False, arm_side: str = "single") -> dict[str, float]:
        """
        Map hand tracking data to robot joint commands.
        
        SAFETY MODE: Currently only controls gripper (open/close).
        All other joints stay at current position for safety testing.
        
        Args:
            use_left_hand: If True, use left hand data; otherwise use right hand
            arm_side: "single", "left", or "right" - determines which smoothed action to use
            
        Returns:
            Dictionary of robot joint commands
        """
        # Select appropriate smoothed action based on arm side
        if arm_side == "left":
            smoothed_action = self.left_smoothed_action
        elif arm_side == "right":
            smoothed_action = self.right_smoothed_action
        else:  # "single" or fallback
            smoothed_action = self.smoothed_action
        
        if smoothed_action is None:
            # Emergency fallback - should not happen if properly initialized
            logging.warning("Smoothed action not initialized, using center positions")
            current_positions = {
                "shoulder_pan.pos": self.config.joint_center,
                "shoulder_lift.pos": self.config.joint_center,
                "elbow_flex.pos": self.config.joint_center,
                "wrist_flex.pos": self.config.joint_center,
                "wrist_roll.pos": self.config.joint_center,
                "gripper.pos": self.config.gripper_open,
            }
        else:
            current_positions = smoothed_action.copy()
        
        # Select hand data
        if use_left_hand:
            hand_openness = self.left_hand_openness
        else:
            hand_openness = self.right_hand_openness
        
        # SAFETY MODE: Keep all joints at current position, only control gripper
        # Gripper: Linear interpolation based on hand openness (0.0 = closed, 1.0 = open)
        target_gripper_pos = (self.config.gripper_closed + 
                            hand_openness * (self.config.gripper_open - self.config.gripper_closed))
        
        return {
            "shoulder_pan.pos": current_positions.get("shoulder_pan.pos", self.config.joint_center),
            "shoulder_lift.pos": current_positions.get("shoulder_lift.pos", self.config.joint_center),
            "elbow_flex.pos": current_positions.get("elbow_flex.pos", self.config.joint_center),
            "wrist_flex.pos": current_positions.get("wrist_flex.pos", self.config.joint_center),
            "wrist_roll.pos": current_positions.get("wrist_roll.pos", self.config.joint_center),
            "gripper.pos": target_gripper_pos,  # CONTROLLED by hand gesture
        }

    def _smooth_action(self, new_action: dict[str, float], arm_side: str = "single") -> dict[str, float]:
        """
        Apply smoothing to robot actions for stability with safety limits.
        Uses different smoothing factors for gripper (fast response) vs joints (safety).
        
        Args:
            new_action: New action to smooth
            arm_side: "single", "left", or "right" - determines which smoothed action to use/update
        """
        # Select appropriate smoothed action based on arm side
        if arm_side == "left":
            current_smoothed = self.left_smoothed_action
        elif arm_side == "right":
            current_smoothed = self.right_smoothed_action
        else:  # "single" or fallback
            current_smoothed = self.smoothed_action
        
        if current_smoothed is None:
            # Update appropriate smoothed action and return
            if arm_side == "left":
                self.left_smoothed_action = new_action
            elif arm_side == "right":
                self.right_smoothed_action = new_action
            else:
                self.smoothed_action = new_action
            return new_action
        
        smoothed = {}
        
        for key, new_val in new_action.items():
            old_val = current_smoothed.get(key, new_val)
            
            # Use different smoothing factors for gripper vs joints
            if key == "gripper.pos":
                smoothing_factor = self.config.gripper_smoothing_factor
                max_change = self.config.max_gripper_change_per_frame
            else:
                smoothing_factor = self.config.smoothing_factor
                max_change = self.config.max_joint_change_per_frame
            
            # Calculate smoothed value with appropriate factor
            alpha = 1.0 - smoothing_factor
            smoothed_val = old_val * smoothing_factor + new_val * alpha
            
            # Apply safety limits to prevent large movements
            change = smoothed_val - old_val
            if abs(change) > max_change:
                if change > 0:
                    smoothed_val = old_val + max_change
                else:
                    smoothed_val = old_val - max_change
            
            smoothed[key] = smoothed_val
        
        # Update appropriate smoothed action
        if arm_side == "left":
            self.left_smoothed_action = smoothed
        elif arm_side == "right":
            self.right_smoothed_action = smoothed
        else:
            self.smoothed_action = smoothed
            
        return smoothed

    def _calculate_hand_openness(self, landmarks) -> float:
        """
        Calculate hand openness as a continuous value from 0.0 (closed) to 1.0 (open).
        Uses the span between thumb tip and pinky tip, normalized by hand size.
        """
        if landmarks is None:
            return 0.0
        
        # Get key landmarks
        thumb_tip = landmarks.landmark[4]    # Thumb tip
        pinky_tip = landmarks.landmark[20]   # Pinky tip
        wrist = landmarks.landmark[0]        # Wrist
        middle_tip = landmarks.landmark[12]  # Middle finger tip
        
        # Calculate hand span (thumb to pinky distance)
        span_x = abs(thumb_tip.x - pinky_tip.x)
        span_y = abs(thumb_tip.y - pinky_tip.y)
        hand_span = np.sqrt(span_x*span_x + span_y*span_y)
        
        # Calculate hand length (wrist to middle finger tip) for normalization
        length_x = middle_tip.x - wrist.x
        length_y = middle_tip.y - wrist.y
        hand_length = np.sqrt(length_x*length_x + length_y*length_y)
        
        # Avoid division by zero
        if hand_length < 0.01:
            return 0.0
        
        # Normalize span by hand length to get openness ratio
        span_ratio = hand_span / hand_length
        
        # Map typical span ratios to 0.0-1.0 range
        # Closed hand: span_ratio ≈ 0.3-0.5
        # Open hand: span_ratio ≈ 0.8-1.2
        min_ratio = 0.3  # Closed hand threshold
        max_ratio = 1.0  # Open hand threshold
        
        # Linear interpolation with clamping
        openness = (span_ratio - min_ratio) / (max_ratio - min_ratio)
        openness = max(0.0, min(1.0, openness))
        
        return openness

    def _draw_landmarks(self, frame: np.ndarray, left_hand, right_hand) -> np.ndarray:
        """Draw hand landmarks and information on the frame."""
        # Draw left hand
        if left_hand is not None:
            self.mp_drawing.draw_landmarks(
                frame, left_hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw hand center
            if self.left_hand_center and self.left_hand_center[0] is not None and self.left_hand_center[1] is not None:
                center = (int(self.left_hand_center[0]), int(self.left_hand_center[1]))
                cv2.circle(frame, center, 10, (0, 255, 0), -1)
                cv2.putText(frame, f"L: {self.left_hand_openness:.1%}", 
                           (center[0] - 30, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw right hand
        if right_hand is not None:
            self.mp_drawing.draw_landmarks(
                frame, right_hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw hand center  
            if self.right_hand_center and self.right_hand_center[0] is not None and self.right_hand_center[1] is not None:
                center = (int(self.right_hand_center[0]), int(self.right_hand_center[1]))
                cv2.circle(frame, center, 10, (0, 0, 255), -1)
                cv2.putText(frame, f"R: {self.right_hand_openness:.1%}", 
                           (center[0] - 30, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def get_action(self) -> dict[str, float]:
        """
        Get robot action from current hand tracking state.
        For Phase 1, returns dummy values. Will be implemented in Phase 2.
        """
        if not self.is_connected:
            raise Exception(f"{self} is not connected")
        
        # Capture frame
        if self.camera is None:
            raise Exception("Camera not connected")
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture frame from camera")
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        left_hand, right_hand = self._detect_hands(frame)
        
        # Update hand state
        self.left_hand_landmarks = left_hand
        self.right_hand_landmarks = right_hand
        self.left_hand_center = self._calculate_hand_center(left_hand)
        self.right_hand_center = self._calculate_hand_center(right_hand)
        self.left_hand_openness = self._calculate_hand_openness(left_hand)
        self.right_hand_openness = self._calculate_hand_openness(right_hand)
        self.left_hand_depth = self._calculate_hand_depth(left_hand)
        self.right_hand_depth = self._calculate_hand_depth(right_hand)
        self.left_wrist_angle = self._calculate_wrist_angle(left_hand)
        self.right_wrist_angle = self._calculate_wrist_angle(right_hand)
        
        # Map hand to robot action (use right hand as primary)
        raw_action = self._map_hand_to_robot_action(use_left_hand=False)
        
        # Apply smoothing for stable robot control
        smoothed_action = self._smooth_action(raw_action)
        self.smoothed_action = smoothed_action
        
        return smoothed_action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Hand tracking doesn't use feedback."""
        pass

    def disconnect(self) -> None:
        """Disconnect camera and cleanup MediaPipe."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        if self.hands is not None:
            self.hands.close()
            self.hands = None
            
        self.is_camera_connected = False
        logging.info(f"{self} disconnected")

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Get current frame with hand landmarks drawn for debugging."""
        if not self.is_connected:
            return None
        
        if self.camera is None:
            return None
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        left_hand, right_hand = self._detect_hands(frame)
        
        # Update hand state
        self.left_hand_landmarks = left_hand
        self.right_hand_landmarks = right_hand
        self.left_hand_center = self._calculate_hand_center(left_hand)
        self.right_hand_center = self._calculate_hand_center(right_hand)
        self.left_hand_openness = self._calculate_hand_openness(left_hand)
        self.right_hand_openness = self._calculate_hand_openness(right_hand)
        
        # Draw landmarks
        frame = self._draw_landmarks(frame, left_hand, right_hand)
        
        # Add frame info with robot mapping data
        info_text = f"Hands: L={left_hand is not None}, R={right_hand is not None}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show gripper control status for right hand
        if right_hand is not None and self.right_hand_center and self.right_hand_center[0] is not None:
            action = self._map_hand_to_robot_action(use_left_hand=False)
            gripper_text = f"GRIPPER: {self.right_hand_openness:.1%} open (pos={action['gripper.pos']:.1f})"
            cv2.putText(frame, gripper_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            safety_text = f"SAFETY MODE: Only gripper active, joints stay at current position"
            cv2.putText(frame, safety_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
            
            # Show safety limits
            limits_text = f"Smoothing: Joint={self.config.smoothing_factor:.1f}, Gripper={self.config.gripper_smoothing_factor:.1f} | Max change: Joint={self.config.max_joint_change_per_frame:.0f}, Gripper={self.config.max_gripper_change_per_frame:.0f}"
            cv2.putText(frame, limits_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 2)
        
        fps_text = f"Press 'q' to quit"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


def test_hand_tracking(camera_index: int = 0, fps: int = 30):
    """
    Test hand tracking with visual feedback.
    Shows camera feed with hand landmarks and gesture detection.
    """
    init_logging()
    logging.info("Starting hand tracking test...")
    
    # Create hand tracking teleoperator
    config = HandTrackingConfig(
        id="test_hand_tracker",
        camera_index=camera_index,
        camera_width=640,
        camera_height=480,
        max_num_hands=2
    )
    
    hand_tracker = HandTrackingTeleoperator(config)
    
    try:
        # Connect to camera and MediaPipe
        hand_tracker.connect()
        
        print(f"🎥 Camera connected: {hand_tracker.frame_width}x{hand_tracker.frame_height}")
        print("👋 Show your hands to the camera!")
        print("✋ Open/close your hands to test gesture detection")
        print("⌨️  Press 'q' to quit")
        
        frame_time = 1.0 / fps
        
        while True:
            start_time = time.time()
            
            # Get debug frame with landmarks
            frame = hand_tracker.get_debug_frame()
            
            if frame is not None:
                # Display frame
                cv2.imshow("Hand Tracking Test", frame)
                
                # Print hand state to console
                left_detected = hand_tracker.left_hand_landmarks is not None
                right_detected = hand_tracker.right_hand_landmarks is not None
                
                if left_detected or right_detected:
                    status = []
                    if left_detected:
                        status.append(f"LEFT: {hand_tracker.left_hand_openness:.1%} open")
                    if right_detected:
                        status.append(f"RIGHT: {hand_tracker.right_hand_openness:.1%} open")
                    print(f"👋 {' | '.join(status)}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # Control frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Error during hand tracking test: {e}")
        logging.exception("Error during hand tracking test")
    finally:
        # Cleanup
        hand_tracker.disconnect()
        cv2.destroyAllWindows()
        print("🏁 Hand tracking test finished")


def load_ports_config(config_path: str) -> dict:
    """Load the ports configuration from JSON file."""
    config_file = Path(config_path)
    
    # If not found and not absolute, try relative to package directory
    if not config_file.exists() and not config_file.is_absolute():
        package_dir = Path(__file__).parent
        # First try in configs/ports directory
        config_file = package_dir / "lerobot" / "configs" / "ports" / config_path
        # If not found there, try current directory
        if not config_file.exists():
            config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Port configuration file not found at {config_path}\n"
            "Please run the dual_teleoperation.py script with --register-ports first."
        )
    
    with open(config_file, 'r') as f:
        return json.load(f)


async def dual_arm_teleoperation(
    camera_index: int = 0,
    ports_config_path: str = "dual_so100_ports.json",
    fps: int = 30,
    duration: Optional[float] = None,
    display_data: bool = False,
    smoothing_factor: float = 0.9,
    max_joint_change: float = 50.0
):
    """
    Run dual arm teleoperation using hand tracking.
    Controls left follower arm with left hand, right follower arm with right hand.
    """
    init_logging()
    logging.info("Starting dual arm hand tracking teleoperation")
    
    # Safety notice
    print("\n" + "="*70)
    print("⚠️  DUAL ARM HAND TRACKING TELEOPERATION - SAFETY FEATURES ACTIVE")
    print("="*70)
    print("🛡️  Current robot positions will be read and maintained")
    print("🔒 Movement limits enforced to prevent sudden movements") 
    print(f"🎛️  Joint smoothing: {smoothing_factor:.1f} | Gripper smoothing: 0.3 (faster response)")
    print(f"📏 Max change/frame: Joints={max_joint_change:.1f} | Gripper=30.0")
    print("🤖 Only gripper control enabled in this safety mode")
    print("👋 LEFT hand → LEFT arm | RIGHT hand → RIGHT arm")
    print("="*70)
    
    # Load port configuration
    try:
        ports_config = load_ports_config(ports_config_path)
        print("✅ Loaded port configuration")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Verify both arms are configured
    if "left_follower" not in ports_config or "right_follower" not in ports_config:
        print("❌ Both left_follower and right_follower must be configured for dual arm mode")
        return
    
    # Create hand tracking teleoperator
    hand_config = HandTrackingConfig(
        id="hand_tracker",
        camera_index=camera_index,
        camera_width=640,
        camera_height=480,
        max_num_hands=2,  # Need both hands for dual arm
        smoothing_factor=smoothing_factor,
        max_joint_change_per_frame=max_joint_change
    )
    
    # Create robot configs for both arms
    left_robot_config = SO100FollowerConfig(
        port=ports_config["left_follower"]["port"],
        id=ports_config["left_follower"]["id"]  # Use the ID from config file
    )
    
    right_robot_config = SO100FollowerConfig(
        port=ports_config["right_follower"]["port"],
        id=ports_config["right_follower"]["id"]  # Use the ID from config file
    )
    
    # Create instances
    print("🔧 Creating hand tracker and robot instances...")
    try:
        hand_tracker = HandTrackingTeleoperator(hand_config)
        left_robot = make_robot_from_config(left_robot_config)
        right_robot = make_robot_from_config(right_robot_config)
        print("✅ Instances created successfully")
    except Exception as e:
        print(f"❌ Failed to create instances: {e}")
        logging.exception("Failed to create instances")
        return
    
    # Connect devices
    print("🔌 Connecting to devices...")
    try:
        hand_tracker.connect()
        print("   ✅ Hand tracker connected")
        
        left_robot.connect()
        print("   ✅ Left robot connected")
        
        right_robot.connect()
        print("   ✅ Right robot connected")
        
        # SAFETY: Initialize hand tracker with current robot positions for both arms
        print("🔒 SAFETY: Reading current robot positions...")
        hand_tracker.initialize_dual_arm_with_robot_positions(left_robot, right_robot)
        
        print("🎉 All devices connected successfully!")
        
    except Exception as e:
        print(f"❌ Failed to connect devices: {e}")
        logging.exception("Failed to connect devices")
        return
    
    print(f"\n🚀 Starting dual arm hand teleoperation at {fps} Hz...")
    if duration:
        print(f"⏱️  Duration: {duration} seconds")
    else:
        print("⏱️  Duration: Until interrupted (Ctrl+C)")
    
    print("\n" + "="*60)
    print("🤖 DUAL ARM HAND TRACKING TELEOPERATION ACTIVE - SAFETY MODE")
    print("="*60)
    print("🔒 SAFETY MODE: Only gripper control enabled")
    print("👋 Show BOTH HANDS to the camera")
    print("🔵 LEFT HAND → LEFT ARM gripper")
    print("🔴 RIGHT HAND → RIGHT ARM gripper")
    print("👌 OPEN hand = Gripper OPEN (100%)")
    print("✊ CLOSE hand = Gripper CLOSED (0%)")
    print("🔧 Hand openness linearly controls gripper position")
    print("⚡ Fast gripper response with reduced smoothing for precision")
    print("🔧 All other joints maintain their current position")
    print("🛡️  Smooth movements with safety limits enforced")
    print("📺 Watch the camera window for visual feedback")
    print("⌨️  Press Ctrl+C to stop teleoperation")
    print("="*60)
    
    # Teleoperation loop
    start_time = time.time()
    loop_count = 0
    frame_time = 1.0 / fps
    
    try:
        while True:
            loop_start = time.time()
            
            # Get actions from hand tracking for both hands
            left_action = hand_tracker._map_hand_to_robot_action(use_left_hand=True, arm_side="left")
            right_action = hand_tracker._map_hand_to_robot_action(use_left_hand=False, arm_side="right")
            
            # Apply smoothing for both actions (using separate smoothed states)
            left_smoothed = hand_tracker._smooth_action(left_action, arm_side="left")
            right_smoothed = hand_tracker._smooth_action(right_action, arm_side="right")
            
            # Update hand tracking state
            hand_tracker.get_action()  # This updates the hand detection
            
            # Send actions to respective robots
            left_robot.send_action(left_smoothed)
            right_robot.send_action(right_smoothed)
            
            # Show debug frame if requested
            if display_data:
                debug_frame = hand_tracker.get_debug_frame()
                if debug_frame is not None:
                    cv2.imshow("Dual Arm Hand Tracking Teleoperation", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
            loop_count += 1
            
            # Print status every 100 loops
            if loop_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_hz = loop_count / elapsed
                print(f"📊 Dual arm teleoperation: {loop_count} loops, {avg_hz:.1f} Hz avg")
                
                # Print current gripper states for debugging
                left_detected = hand_tracker.left_hand_landmarks is not None
                right_detected = hand_tracker.right_hand_landmarks is not None
                
                status_parts = []
                if left_detected:
                    status_parts.append(f"LEFT_HAND: {hand_tracker.left_hand_openness:.1%} → L_GRIPPER: {left_smoothed['gripper.pos']:.1f}")
                else:
                    status_parts.append("LEFT_HAND: No hand → L_GRIPPER: maintaining")
                    
                if right_detected:
                    status_parts.append(f"RIGHT_HAND: {hand_tracker.right_hand_openness:.1%} → R_GRIPPER: {right_smoothed['gripper.pos']:.1f}")
                else:
                    status_parts.append("RIGHT_HAND: No hand → R_GRIPPER: maintaining")
                
                print(f"👋 {' | '.join(status_parts)}")
                print(f"🔧 LEFT ARM smoothed state independent from RIGHT ARM smoothed state")
            
            # Check duration
            if duration is not None and time.time() - start_time >= duration:
                break
                
            # Control frame rate
            elapsed_time = time.time() - loop_start
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Dual arm teleoperation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during teleoperation: {e}")
        logging.exception("Error during teleoperation")
        
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        try:
            if display_data:
                cv2.destroyAllWindows()
            
            hand_tracker.disconnect()
            left_robot.disconnect()
            right_robot.disconnect()
            print("✅ All devices disconnected")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
        
        print("🏁 Dual arm hand teleoperation finished")


async def single_arm_teleoperation(
    camera_index: int = 0,
    ports_config_path: str = "dual_so100_ports.json",
    fps: int = 30,
    duration: Optional[float] = None,
    display_data: bool = False,
    smoothing_factor: float = 0.9,
    max_joint_change: float = 50.0,
    use_left_hand: bool = False,
    control_left_arm: bool = False
):
    """
    Run single arm teleoperation using hand tracking.
    
    Args:
        use_left_hand: If True, use left hand for control; otherwise use right hand
        control_left_arm: If True, control left follower arm; otherwise control right follower arm
    """
    init_logging()
    logging.info("Starting single arm hand tracking teleoperation")
    
    hand_name = "LEFT" if use_left_hand else "RIGHT"
    arm_name = "LEFT" if control_left_arm else "RIGHT"
    
    # Safety notice
    print("\n" + "="*70)
    print("⚠️  HAND TRACKING TELEOPERATION - SAFETY FEATURES ACTIVE")
    print("="*70)
    print("🛡️  Current robot position will be read and maintained")
    print("🔒 Movement limits enforced to prevent sudden movements") 
    print(f"🎛️  Joint smoothing: {smoothing_factor:.1f} | Gripper smoothing: 0.3 (faster response)")
    print(f"📏 Max change/frame: Joints={max_joint_change:.1f} | Gripper=30.0")
    print("🤖 Only gripper control enabled in this safety mode")
    print(f"👋 {hand_name} hand → {arm_name} arm")
    print("="*70)
    
    # Load port configuration
    try:
        ports_config = load_ports_config(ports_config_path)
        print("✅ Loaded port configuration")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Create hand tracking teleoperator
    hand_config = HandTrackingConfig(
        id="hand_tracker",
        camera_index=camera_index,
        camera_width=640,
        camera_height=480,
        max_num_hands=2,
        smoothing_factor=smoothing_factor,
        max_joint_change_per_frame=max_joint_change
    )
    
    # Create robot config for selected arm
    arm_key = "left_follower" if control_left_arm else "right_follower"
    if arm_key not in ports_config:
        print(f"❌ {arm_key} not found in ports configuration")
        return
        
    robot_config = SO100FollowerConfig(
        port=ports_config[arm_key]["port"],
        id=ports_config[arm_key]["id"]  # Use the ID from config file
    )
    
    # Create instances
    print("🔧 Creating hand tracker and robot instances...")
    try:
        hand_tracker = HandTrackingTeleoperator(hand_config)
        robot = make_robot_from_config(robot_config)
        print("✅ Instances created successfully")
    except Exception as e:
        print(f"❌ Failed to create instances: {e}")
        logging.exception("Failed to create instances")
        return
    
    # Connect devices
    print("🔌 Connecting to devices...")
    try:
        hand_tracker.connect()
        print("   ✅ Hand tracker connected")
        
        robot.connect()
        print("   ✅ Robot connected")
        
        # SAFETY: Initialize hand tracker with current robot position
        print("🔒 SAFETY: Reading current robot position...")
        hand_tracker.initialize_with_robot_position(robot)
        
        print("🎉 All devices connected successfully!")
        
    except Exception as e:
        print(f"❌ Failed to connect devices: {e}")
        logging.exception("Failed to connect devices")
        return
    
    print(f"\n🚀 Starting single arm hand teleoperation at {fps} Hz...")
    if duration:
        print(f"⏱️  Duration: {duration} seconds")
    else:
        print("⏱️  Duration: Until interrupted (Ctrl+C)")
    
    print("\n" + "="*60)
    print("🤖 HAND TRACKING TELEOPERATION ACTIVE - SAFETY MODE")
    print("="*60)
    print("🔒 SAFETY MODE: Only gripper control enabled")
    print(f"✋ Show your {hand_name} HAND to the camera")
    print("👌 OPEN hand = Gripper OPEN (100%)")
    print("✊ CLOSE hand = Gripper CLOSED (0%)")
    print("🔧 Hand openness linearly controls gripper position")
    print("⚡ Fast gripper response with reduced smoothing for precision")
    print("🔧 All other joints maintain their current position")
    print("🛡️  Smooth movements with safety limits enforced")
    print("📺 Watch the camera window for visual feedback")
    print("⌨️  Press Ctrl+C to stop teleoperation")
    print("="*60)
    
    # Teleoperation loop
    start_time = time.time()
    loop_count = 0
    frame_time = 1.0 / fps
    
    try:
        while True:
            loop_start = time.time()
            
            # Update hand detection
            hand_tracker.get_action()
            
            # Get action from selected hand
            action = hand_tracker._map_hand_to_robot_action(use_left_hand=use_left_hand)
            
            # Apply smoothing
            smoothed_action = hand_tracker._smooth_action(action)
            hand_tracker.smoothed_action = smoothed_action
            
            # Send action to robot
            robot.send_action(smoothed_action)
            
            # Show debug frame if requested
            if display_data:
                debug_frame = hand_tracker.get_debug_frame()
                if debug_frame is not None:
                    cv2.imshow(f"Hand Tracking Teleoperation - {hand_name} hand → {arm_name} arm", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
            loop_count += 1
            
            # Print status every 100 loops
            if loop_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_hz = loop_count / elapsed
                print(f"📊 Hand teleoperation: {loop_count} loops, {avg_hz:.1f} Hz avg")
                
                # Print current gripper state for debugging
                if use_left_hand:
                    hand_detected = hand_tracker.left_hand_landmarks is not None
                    hand_openness = hand_tracker.left_hand_openness
                else:
                    hand_detected = hand_tracker.right_hand_landmarks is not None
                    hand_openness = hand_tracker.right_hand_openness
                    
                if hand_detected:
                    print(f"👋 {hand_name} HAND: {hand_openness:.1%} open | {arm_name} GRIPPER: {smoothed_action['gripper.pos']:.1f} | SAFETY: Joints stay at current position")
                else:
                    print(f"🔍 No {hand_name.lower()} hand detected | SAFETY: All joints maintaining position")
            
            # Check duration
            if duration is not None and time.time() - start_time >= duration:
                break
                
            # Control frame rate
            elapsed_time = time.time() - loop_start
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Teleoperation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during teleoperation: {e}")
        logging.exception("Error during teleoperation")
        
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        try:
            if display_data:
                cv2.destroyAllWindows()
            
            hand_tracker.disconnect()
            robot.disconnect()
            print("✅ All devices disconnected")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
        
        print("🏁 Single arm hand teleoperation finished")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Hand Tracking Teleoperation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--test-tracking",
        action="store_true",
        help="Test hand tracking with visual feedback (Phase 1)"
    )
    
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Run robot teleoperation (Phase 2/3)"
    )
    
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to use for hand tracking"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second"
    )
    
    parser.add_argument(
        "--single-arm",
        action="store_true", 
        help="Single arm mode"
    )
    
    parser.add_argument(
        "--dual-arm",
        action="store_true",
        help="Dual arm mode - left hand controls left arm, right hand controls right arm"
    )
    
    parser.add_argument(
        "--use-left-hand",
        action="store_true",
        help="Use left hand for control (single arm mode only)"
    )
    
    parser.add_argument(
        "--control-left-arm",
        action="store_true",
        help="Control left arm instead of right arm (single arm mode only)"
    )
    
    parser.add_argument(
        "--ports-config",
        type=str,
        help="Path to robot ports configuration file"
    )
    
    parser.add_argument(
        "--display-data",
        action="store_true",
        help="Display camera feed with visual feedback during teleoperation"
    )
    
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="Movement smoothing factor (0.0=no smoothing, 1.0=maximum smoothing)"
    )
    
    parser.add_argument(
        "--max-joint-change",
        type=float,
        default=50.0,
        help="Maximum joint position change per frame for safety"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.test_tracking and not args.teleop:
        print("❌ Please specify either --test-tracking or --teleop")
        parser.print_help()
        return
    
    if args.teleop:
        if args.single_arm:
            if not args.ports_config:
                print("❌ Single arm mode requires --ports-config")
                return
            # Run single arm teleoperation
            asyncio.run(single_arm_teleoperation(
                camera_index=args.camera_index,
                ports_config_path=args.ports_config,
                fps=args.fps,
                display_data=args.display_data,
                smoothing_factor=args.smoothing,
                max_joint_change=args.max_joint_change,
                use_left_hand=args.use_left_hand,
                control_left_arm=args.control_left_arm
            ))
        elif args.dual_arm:
            if not args.ports_config:
                print("❌ Dual arm mode requires --ports-config")
                return
            # Run dual arm teleoperation
            asyncio.run(dual_arm_teleoperation(
                camera_index=args.camera_index,
                ports_config_path=args.ports_config,
                fps=args.fps,
                display_data=args.display_data,
                smoothing_factor=args.smoothing,
                max_joint_change=args.max_joint_change
            ))
        else:
            print("❌ Please specify --single-arm or --dual-arm with --teleop")
            return
    
    if args.test_tracking:
        test_hand_tracking(args.camera_index, args.fps)


if __name__ == "__main__":
    main() 