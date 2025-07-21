#!/usr/bin/env python3
"""
MCP Server for Robot Control

This server exposes tools for controlling a robot through the Model Context Protocol.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence
from mcp.server import FastMCP
from mcp.types import Tool
import uvicorn

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lerobot.common.robots import Robot, make_robot_from_config
from lerobot.common.robots.so100_follower import SO100FollowerConfig

# Configuration
POSITIONS_DIR = Path("positions")

# Create a robot configuration
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem5A680097171",  # Same port as calibrated
    id="left_follower"  # Same ID as used by dual_teleoperation.py
)
robot = make_robot_from_config(robot_config)

# Create the server instance
server = FastMCP("robot-control")

# Helper functions
def get_available_position_files():
    """Get list of available position files."""
    if not POSITIONS_DIR.exists():
        return []
    
    position_files = []
    for pos_file in POSITIONS_DIR.glob("*.json"):
        position_files.append(pos_file)
    
    return sorted(position_files)

def load_position_file(filename):
    """Load positions from a specific file."""
    filepath = POSITIONS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Position file not found: {filename}")
    
    with open(filepath, "r") as f:
        save_data = json.load(f)
    
    return save_data

def get_positions_from_robot():
    """Get current positions from robot."""
    observation = robot.get_observation()
    # Extract position data (filter out camera data)
    positions = {key: value for key, value in observation.items() if key.endswith('.pos')}
    return positions

async def smooth_move_to_positions(target_positions, steps=20, delay=0.1):
    """Smoothly move robot to target positions using interpolation."""
    current_positions = get_positions_from_robot()
    
    for step in range(steps + 1):
        # Interpolate between current and target positions
        alpha = step / steps
        intermediate_positions = {}
        
        for motor in target_positions:
            if motor in current_positions:
                current = current_positions[motor]
                target = target_positions[motor]
                intermediate_positions[motor] = current + alpha * (target - current)
        
        robot.send_action(intermediate_positions)
        await asyncio.sleep(delay)

@server.tool()
async def robot_move_to(position: list[float]) -> dict:
    """Move the robot to a specific position.
    
    Args:
        position: Array of 6 joint positions
    """
    try:
        if not robot.is_connected:
            robot.connect()
            
        if not position or len(position) != 6:
            return {"type": "text", "text": "Error: Position must be an array of exactly 6 numbers"}
            
        # Convert position to robot action format
        action_dict = {}
        motor_names = list(robot.action_features.keys())
        for i, pos in enumerate(position[:len(motor_names)]):
            action_dict[motor_names[i]] = float(pos)
        
        robot.send_action(action_dict)
        
        return {"type": "text", "text": f"Robot moved to position: {position}"}
            
    except Exception as e:
        return {"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}

@server.tool()
async def robot_grasp() -> dict:
    """Close the robot gripper to grasp an object."""
    try:
        if not robot.is_connected:
            robot.connect()
            
        # Close gripper
        action_dict = {"gripper.pos": 0.0}  # Fully closed
        robot.send_action(action_dict)
        
        return {"type": "text", "text": "Robot gripper closed"}
            
    except Exception as e:
        return {"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}

@server.tool()
async def robot_release() -> dict:
    """Open the robot gripper to release an object."""
    try:
        if not robot.is_connected:
            robot.connect()
            
        # Open gripper
        action_dict = {"gripper.pos": 100.0}  # Fully open
        robot.send_action(action_dict)
        
        return {"type": "text", "text": "Robot gripper opened"}
            
    except Exception as e:
        return {"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}

@server.tool()
async def list_saved_positions() -> dict:
    """List all available saved position files."""
    try:
        position_files = get_available_position_files()
        
        if not position_files:
            return {"type": "text", "text": "No saved position files found in positions/ directory"}
        
        file_list = []
        for pos_file in position_files:
            try:
                save_data = load_position_file(pos_file.name)
                timestamp = save_data.get("timestamp", "Unknown")
                motor_count = len(save_data.get("positions", {}))
                file_list.append(f"• {pos_file.stem} (saved: {timestamp}, {motor_count} motors)")
            except Exception as e:
                file_list.append(f"• {pos_file.stem} (error reading file: {str(e)})")
        
        result = "Available saved positions:\n" + "\n".join(file_list)
        return {"type": "text", "text": result}
        
    except Exception as e:
        return {"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}

@server.tool()
async def robot_smooth_move_to_saved(position_name: str, steps: int = 20, delay: float = 0.1) -> dict:
    """Move the robot smoothly to a saved position.
    
    Args:
        position_name: Name of the saved position file (without .json extension)
        steps: Number of interpolation steps for smooth movement (default: 20)
        delay: Delay between steps in seconds (default: 0.1)
    """
    try:
        if not robot.is_connected:
            robot.connect()
        
        # Add .json extension if not present
        filename = position_name if position_name.endswith('.json') else f"{position_name}.json"
        
        # Load the saved positions
        save_data = load_position_file(filename)
        target_positions = save_data["positions"]
        timestamp = save_data.get("timestamp", "Unknown")
        
        # Perform smooth movement
        await smooth_move_to_positions(target_positions, steps=steps, delay=delay)
        
        return {"type": "text", "text": f"Robot smoothly moved to position '{position_name}' (saved: {timestamp}) in {steps} steps"}
        
    except FileNotFoundError as e:
        return {"type": "text", "text": f"Error: Position file '{position_name}' not found. Use list_saved_positions to see available files."}
    except Exception as e:
        return {"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}

@server.tool()
async def get_current_position() -> dict:
    """Get the current position of all robot joints."""
    try:
        if not robot.is_connected:
            robot.connect()
        
        positions = get_positions_from_robot()
        
        position_list = []
        for motor, position in positions.items():
            position_list.append(f"  {motor:15s}: {position:8.2f}")
        
        result = "Current robot positions:\n" + "\n".join(position_list)
        return {"type": "text", "text": result}
        
    except Exception as e:
        return {"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}




if __name__ == "__main__":
    app = server.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=3000)
