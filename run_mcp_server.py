#!/usr/bin/env python3
"""
Runner script that starts both the MCP server and ngrok.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

async def run_ngrok():
    """Start ngrok and return the public URL."""
    # Start ngrok on port 5050 (same as the robot control server)
    process = await asyncio.create_subprocess_exec(
        'ngrok', 'http', '5050',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for ngrok to start
    await asyncio.sleep(2)
    
    # Get the public URL from ngrok's API
    try:
        process = await asyncio.create_subprocess_exec(
            'curl', 'http://localhost:4040/api/tunnels',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        data = json.loads(stdout)
        public_url = data['tunnels'][0]['public_url']
        print(f"\nngrok public URL: {public_url}")
        
        # Update the MCP config with the new URL
        config_path = Path('mcp_config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'mcpServers' in config and 'robot-control' in config['mcpServers']:
                config['mcpServers']['robot-control']['env']['ROBOT_SERVER_URL'] = public_url
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"Updated MCP config with new URL")
        
        return public_url
    except Exception as e:
        print(f"Error getting ngrok URL: {e}")
        return None

async def run_robot_server():
    """Start the robot control server."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, 'server2.py',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    while True:
        if process.stdout:
            line = await process.stdout.readline()
            if line:
                print(f"[Server] {line.decode().strip()}")
        if process.stderr:
            line = await process.stderr.readline()
            if line:
                print(f"[Server Error] {line.decode().strip()}")
        
        if process.returncode is not None:
            break
    
    return await process.wait()

async def main():
    """Run both servers concurrently."""
    # Start ngrok first to get the URL
    ngrok_url = await run_ngrok()
    if not ngrok_url:
        print("Failed to start ngrok")
        return
    
    # Then start the robot server
    await run_robot_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # Kill any remaining ngrok processes
        subprocess.run(['pkill', 'ngrok'])
        sys.exit(0) 