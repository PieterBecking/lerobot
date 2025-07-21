#!/usr/bin/env python3
"""
MCP Server for Robot Control

This server exposes tools for controlling a robot through HTTP requests.
It acts as an MCP interface to the robot_control_server.py Flask application.
"""

import asyncio
import json
import httpx
from typing import Any, Sequence, Dict
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    JSONRPCError,
    Tool,
    TextContent,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
)

# Configuration
ROBOT_SERVER_URL = "https://68abb2f56964.ngrok-free.app"  # Update this with your actual URL

# Create the server instance
server = Server("robot-control")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available robot control tools."""
    return [
        Tool(
            name="robot_move_to",
            description="Move the robot to a specific position",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of 6 joint positions (in degrees or radians depending on robot configuration)",
                        "minItems": 6,
                        "maxItems": 6
                    }
                },
                "required": ["position"]
            }
        ),
        Tool(
            name="robot_grasp",
            description="Close the robot gripper to grasp an object",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="robot_release",
            description="Open the robot gripper to release an object",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="robot_status",
            description="Check if the robot server is responding",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        )
    ]

async def _make_robot_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make HTTP request to robot control server."""
    server_url = ROBOT_SERVER_URL
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/execute",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Execute a robot control tool."""
    try:
        if name == "robot_move_to":
            position = arguments.get("position")
            if not position or len(position) != 6:
                raise JSONRPCError(
                    INVALID_PARAMS,
                    "Position must be an array of exactly 6 numbers"
                )
            
            payload = {
                "action": "move_to",
                "position": position
            }
            
            result = await _make_robot_request(payload)
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Robot moved to position: {position}\nServer response: {json.dumps(result, indent=2)}"
                    )
                ]
            )
            
        elif name == "robot_grasp":
            payload = {"action": "grasp"}
            result = await _make_robot_request(payload)
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Robot gripper closed\nServer response: {json.dumps(result, indent=2)}"
                    )
                ]
            )
            
        elif name == "robot_release":
            payload = {"action": "release"}
            result = await _make_robot_request(payload)
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Robot gripper opened\nServer response: {json.dumps(result, indent=2)}"
                    )
                ]
            )
            
        elif name == "robot_status":
            try:
                server_url = ROBOT_SERVER_URL
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{server_url}/")
                    status_text = f"Robot server is reachable (HTTP {response.status_code})"
            except Exception as e:
                status_text = f"Robot server is not reachable: {str(e)}"
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=status_text
                    )
                ]
            )
            
        else:
            raise JSONRPCError(
                METHOD_NOT_FOUND,
                f"Unknown tool: {name}"
            )
    except Exception as e:
        raise JSONRPCError(
            INTERNAL_ERROR,
            f"Tool execution failed: {str(e)}"
        )

async def main():
    """Main entry point."""
    # You can update the URL here or via environment variable
    import os
    global ROBOT_SERVER_URL
    ROBOT_SERVER_URL = os.getenv("ROBOT_SERVER_URL", ROBOT_SERVER_URL)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="robot-control",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 











# {
#   "mcpServers": {
#     "robot-control": {
#       "command": "python",
#       "args": ["/root/mcp-robot-control/server.py"]
#     }
#   }
# }