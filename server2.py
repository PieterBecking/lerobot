#!/usr/bin/env python3
"""
MCP Server for Robot Control

This server exposes tools for controlling a robot through the Model Context Protocol.
"""

import asyncio
import json
from typing import Any, Dict, Sequence
from mcp.server import FastMCP
from mcp.types import Tool
import uvicorn

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lerobot.common.robots import Robot, make_robot_from_config
from lerobot.common.robots.so100_follower import SO100FollowerConfig

# Create a robot configuration
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem5A680097171",  # Same port as calibrated
    id="left_follower"  # Same ID as used by dual_teleoperation.py
)
robot = make_robot_from_config(robot_config)

# Create the server instance
server = FastMCP("robot-control")

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

if __name__ == "__main__":
    app = server.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=3000)
