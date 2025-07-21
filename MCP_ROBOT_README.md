# Robot Control MCP Server

This MCP (Model Context Protocol) server provides AI assistants with tools to control your robot through HTTP requests. It acts as a bridge between AI assistants and your `robot_control_server.py` Flask application.

## Features

The MCP server exposes the following tools:

- **robot_move_to**: Move the robot to a specific position (6 joint positions)
- **robot_grasp**: Close the robot gripper to grasp objects
- **robot_release**: Open the robot gripper to release objects
- **robot_status**: Check if the robot server is responding

## Setup

### Prerequisites

1. Make sure your `robot_control_server.py` is running and accessible
2. Python virtual environment with required dependencies installed
3. Update the ngrok URL in the configuration

### Installation

The required dependencies are already installed:
- `httpx` - for making HTTP requests
- `mcp` - Model Context Protocol library

### Configuration

1. **Update the robot server URL**: 
   - In `robot_mcp_server.py`, update the `ROBOT_SERVER_URL` variable
   - Or set the `ROBOT_SERVER_URL` environment variable

2. **For Claude Desktop integration**:
   - Add the configuration from `mcp_config.json` to your Claude Desktop settings
   - Update the path to point to your `robot_mcp_server.py` file
   - Update the ngrok URL in the environment variables

## Usage

### Running the MCP Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the MCP server
python robot_mcp_server.py
```

### Using with Claude Desktop

1. Add the MCP server configuration to Claude Desktop's settings
2. Restart Claude Desktop
3. You can now ask Claude to control your robot using natural language

Example conversations:
- "Release the robot gripper"
- "Move the robot to position [0, 45, -90, 0, 45, 0]"
- "Grasp an object with the robot"
- "Check if the robot server is responding"

### Tool Reference

#### robot_move_to
```json
{
  "position": [0, 45, -90, 0, 45, 0]
}
```
- `position`: Array of 6 joint positions (in degrees or radians depending on robot configuration)

#### robot_grasp
```json
{}
```
No parameters required. Closes the gripper.

#### robot_release
```json
{}
```
No parameters required. Opens the gripper.

#### robot_status
```json
{}
```
No parameters required. Checks server connectivity.

## Architecture

```
AI Assistant (Claude) 
    ↓ MCP Protocol
MCP Server (robot_mcp_server.py)
    ↓ HTTP POST
Flask Server (robot_control_server.py)
    ↓ Robot API
Physical Robot
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Make sure your `robot_control_server.py` is running
2. **Invalid URL**: Update the ngrok URL if it has changed
3. **Timeout errors**: Check network connectivity and server responsiveness

### Testing the Setup

You can test the robot server directly:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"action": "release"}' \
  https://your-ngrok-url.ngrok-free.app/execute
```

### Environment Variables

- `ROBOT_SERVER_URL`: Override the default robot server URL

## Security Notes

- The MCP server makes HTTP requests to your robot control server
- Ensure your ngrok tunnel is secure and not publicly exposed unless intended
- Consider adding authentication if needed for production use

## Development

To modify or extend the MCP server:

1. Add new tools in the `list_tools()` method
2. Implement corresponding handlers in `call_tool()`
3. Follow the MCP protocol specifications for tool definitions

## Example MCP Client Configuration

For Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "robot-control": {
      "command": "python",
      "args": ["/path/to/your/robot_mcp_server.py"],
      "env": {
        "ROBOT_SERVER_URL": "https://your-ngrok-url.ngrok-free.app"
      }
    }
  }
}
``` 