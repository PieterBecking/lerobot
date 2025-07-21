#!/usr/bin/env python3
"""
Servo Position Control Script

This script demonstrates how to:
1. Read current servo positions and save them to a file
2. Load saved positions and move servos to those positions

Usage:
    python servo_position_control.py save           # Save current positions
    python servo_position_control.py load           # Move to saved positions
    python servo_position_control.py print          # Just print current positions
"""

import json
import time
import sys
import argparse
from pathlib import Path
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so100_follower import SO100FollowerConfig
from lerobot.common.teleoperators import make_teleoperator_from_config
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig

# Configuration paths
PORTS_CONFIG_DIR = Path("lerobot/configs/ports")
POSITIONS_DIR = Path("positions")
POSITIONS_FILE = "saved_positions.json"  # Fallback for load command

def get_available_configs():
    """Get list of available port configuration files."""
    if not PORTS_CONFIG_DIR.exists():
        raise FileNotFoundError(f"Ports config directory not found: {PORTS_CONFIG_DIR}")
    
    configs = []
    for config_file in PORTS_CONFIG_DIR.glob("*.json"):
        configs.append(config_file)
    
    if not configs:
        raise FileNotFoundError(f"No JSON config files found in {PORTS_CONFIG_DIR}")
    
    return sorted(configs)

def get_available_position_files():
    """Get list of available position files."""
    if not POSITIONS_DIR.exists():
        return []
    
    position_files = []
    for pos_file in POSITIONS_DIR.glob("*.json"):
        position_files.append(pos_file)
    
    return sorted(position_files)

def select_position_file():
    """Let user select which position file to use."""
    position_files = get_available_position_files()
    
    if not position_files:
        print(f"\n❌ No position files found in {POSITIONS_DIR}")
        return None
    
    print(f"\n📂 Available position files:")
    for i, pos_file in enumerate(position_files, 1):
        print(f"  {i}. {pos_file.stem}")
    
    while True:
        try:
            choice = input(f"\nSelect position file (1-{len(position_files)}): ")
            index = int(choice) - 1
            if 0 <= index < len(position_files):
                return position_files[index]
            else:
                print(f"Please enter a number between 1 and {len(position_files)}")
        except ValueError:
            print("Please enter a valid number")

def get_position_filename():
    """Ask user for a filename to save positions."""
    while True:
        filename = input("\n📝 Enter name for saved positions (without .json): ").strip()
        if not filename:
            print("Please enter a valid filename")
            continue
        
        # Clean filename (remove invalid characters)
        import re
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = POSITIONS_DIR / filename
        
        if filepath.exists():
            overwrite = input(f"⚠️  File '{filename}' already exists. Overwrite? (y/N): ")
            if overwrite.lower() != 'y':
                continue
        
        return filepath

def select_config_file():
    """Let user select which config file to use."""
    configs = get_available_configs()
    
    print("\n📁 Available port configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config.name}")
    
    while True:
        try:
            choice = input(f"\nSelect config file (1-{len(configs)}): ")
            index = int(choice) - 1
            if 0 <= index < len(configs):
                return configs[index]
            else:
                print(f"Please enter a number between 1 and {len(configs)}")
        except ValueError:
            print("Please enter a valid number")

def select_robot_from_config(config_file):
    """Let user select which robot arm from the config."""
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    print(f"\n🤖 Available robots in {config_file.name}:")
    robot_keys = list(config_data.keys())
    
    for i, (key, robot_info) in enumerate(config_data.items(), 1):
        robot_type = robot_info.get('type', 'unknown')
        robot_name = robot_info.get('name', key)
        port = robot_info.get('port', 'unknown')
        print(f"  {i}. {robot_name} ({robot_type}) - {port}")
    
    while True:
        try:
            choice = input(f"\nSelect robot (1-{len(robot_keys)}): ")
            index = int(choice) - 1
            if 0 <= index < len(robot_keys):
                selected_key = robot_keys[index]
                return selected_key, config_data[selected_key]
            else:
                print(f"Please enter a number between 1 and {len(robot_keys)}")
        except ValueError:
            print("Please enter a valid number")

def find_calibration_id(config_id, robot_type):
    """Find the correct calibration file ID based on config ID and robot type."""
    from pathlib import Path
    from lerobot.common.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS
    
    # Determine calibration directory based on robot type
    if robot_type == "so100_follower":
        calib_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "so100_follower"
    elif robot_type == "so100_leader":
        calib_dir = HF_LEROBOT_CALIBRATION / TELEOPERATORS / "so100_leader"
    else:
        return config_id  # fallback
    
    if not calib_dir.exists():
        print(f"⚠️  Calibration directory not found: {calib_dir}")
        return config_id
    
    # Get all available calibration files
    calib_files = [f.stem for f in calib_dir.glob("*.json")]
    print(f"📂 Available calibration files: {calib_files}")
    
    # Try exact match first
    if config_id in calib_files:
        print(f"✅ Found exact calibration match: {config_id}")
        return config_id
    
    # Try without "_arm" suffix
    without_arm = config_id.replace('_arm', '')
    if without_arm in calib_files:
        print(f"✅ Found calibration match: {config_id} → {without_arm}")
        return without_arm
    
    # Try fuzzy matching based on key parts
    config_parts = config_id.lower().split('_')
    best_match = None
    
    for calib_file in calib_files:
        calib_parts = calib_file.lower().split('_')
        # Check if all major parts match (left/right, leader/follower)
        if any(part in calib_parts for part in config_parts if part not in ['arm']):
            if 'left' in config_parts and 'left' in calib_parts:
                best_match = calib_file
                break
            elif 'right' in config_parts and 'right' in calib_parts:
                best_match = calib_file
                break
            elif 's7' in config_parts and 's7' in calib_parts:
                best_match = calib_file
                break
    
    if best_match:
        print(f"✅ Found calibration match: {config_id} → {best_match}")
        return best_match
    
    print(f"⚠️  No calibration file found for {config_id}, available: {calib_files}")
    print(f"🔧 Will use config ID as-is: {config_id}")
    return config_id

def create_robot_from_selection():
    """Create robot instance based on user selection."""
    config_file = select_config_file()
    robot_key, robot_info = select_robot_from_config(config_file)
    
    robot_type = robot_info['type']
    port = robot_info['port']
    config_id = robot_info['id']
    
    # Find the correct calibration ID
    calibration_id = find_calibration_id(config_id, robot_type)
    
    print(f"\n🔧 Creating {robot_info.get('name', robot_key)}...")
    print(f"📁 Config ID: {config_id} → Calibration ID: {calibration_id}")
    
    # Create appropriate config and device based on type
    if robot_type == "so100_follower":
        robot_config = SO100FollowerConfig(port=port, id=calibration_id)
        device = make_robot_from_config(robot_config)
    elif robot_type == "so100_leader":
        teleop_config = SO100LeaderConfig(port=port, id=calibration_id)
        device = make_teleoperator_from_config(teleop_config)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
    # Connect to device (it will automatically load existing calibration)
    print("🔌 Connecting to device...")
    device.connect()
    print("✅ Connected successfully!")
    
    return device, robot_key, robot_info

def get_positions_from_device(device):
    """Get current positions from robot or teleoperator."""
    if hasattr(device, 'get_observation'):
        # Robot device
        observation = device.get_observation()
        # Extract position data (filter out camera data)
        positions = {key: value for key, value in observation.items() if key.endswith('.pos')}
    elif hasattr(device, 'get_action'):
        # Teleoperator device
        positions = device.get_action()
    else:
        raise ValueError("Device does not support position reading")
    
    return positions

def send_positions_to_device(device, positions):
    """Send positions to robot or teleoperator."""
    if hasattr(device, 'send_action'):
        # Robot device
        device.send_action(positions)
    elif hasattr(device, 'bus'):
        # Teleoperator device - use low-level bus access
        # Convert from "motor.pos" format to just "motor" format
        motor_positions = {key.replace('.pos', ''): value for key, value in positions.items()}
        device.bus.sync_write("Goal_Position", motor_positions)
    else:
        raise ValueError("Device does not support position control")

def print_positions(device):
    """Read and print current servo positions."""
    print("\n📊 Reading current servo positions...")
    positions = get_positions_from_device(device)
    
    print("\nCurrent Positions:")
    for motor, position in positions.items():
        print(f"  {motor:15s}: {position:8.2f}")
    
    return positions

def save_positions(device):
    """Save current servo positions to file."""
    positions = print_positions(device)
    
    # Create positions directory if it doesn't exist
    POSITIONS_DIR.mkdir(exist_ok=True)
    
    # Get filename from user
    filepath = get_position_filename()
    
    # Add timestamp for reference
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "positions": positions
    }
    
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Positions saved to {filepath}")

def load_and_move_to_positions(device):
    """Load saved positions and move servos to those positions."""
    # First try to find position files in the positions directory
    position_file = select_position_file()
    
    # If no file selected, fall back to old behavior
    if position_file is None:
        if not Path(POSITIONS_FILE).exists():
            print(f"❌ No saved positions file found: {POSITIONS_FILE}")
            print("   Run 'python servo_position_control.py save' first")
            return
        position_file = Path(POSITIONS_FILE)
    
    with open(position_file, "r") as f:
        save_data = json.load(f)
    
    saved_positions = save_data["positions"]
    timestamp = save_data["timestamp"]
    
    print(f"\n📂 Loading positions from: {position_file.name}")
    print(f"📅 Saved on: {timestamp}")
    print("\nSaved Positions:")
    for motor, position in saved_positions.items():
        print(f"  {motor:15s}: {position:8.2f}")
    
    # Show current positions for comparison
    current_positions = get_positions_from_device(device)
    print("\nCurrent Positions:")
    for motor, position in current_positions.items():
        print(f"  {motor:15s}: {position:8.2f}")
    
    # Ask for confirmation
    response = input("\n🤖 Move servos to saved positions? (y/N): ")
    if response.lower() != 'y':
        print("Movement cancelled.")
        return
    
    print("\n🎯 Moving servos to saved positions...")
    send_positions_to_device(device, saved_positions)
    
    print("✅ Movement command sent!")
    
    # Wait and show final positions
    time.sleep(3)
    final_positions = get_positions_from_device(device)
    print("\nFinal Positions:")
    for motor, position in final_positions.items():
        print(f"  {motor:15s}: {position:8.2f}")

def smooth_move_to_positions(device, target_positions, steps=20, delay=0.1):
    """Smoothly move servos to target positions using interpolation."""
    current_positions = get_positions_from_device(device)
    
    print(f"\n🌊 Smooth movement in {steps} steps...")
    
    for step in range(steps + 1):
        # Interpolate between current and target positions
        alpha = step / steps
        intermediate_positions = {}
        
        for motor in target_positions:
            if motor in current_positions:
                current = current_positions[motor]
                target = target_positions[motor]
                intermediate_positions[motor] = current + alpha * (target - current)
        
        send_positions_to_device(device, intermediate_positions)
        time.sleep(delay)
        
        # Show progress
        if step % 5 == 0:
            print(f"  Step {step}/{steps}")
    
    print("✅ Smooth movement completed!")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["save", "load", "print", "smooth"], 
                       help="Command to execute")
    
    args = parser.parse_args()
    
    try:
        device, robot_key, robot_info = create_robot_from_selection()
        print(f"\n✅ Using {robot_info.get('name', robot_key)}")
        
        if args.command == "print":
            print_positions(device)
            
        elif args.command == "save":
            save_positions(device)
            
        elif args.command == "load":
            load_and_move_to_positions(device)
            
        elif args.command == "smooth":
            # Load saved positions and move smoothly
            position_file = select_position_file()
            
            if position_file is None:
                if not Path(POSITIONS_FILE).exists():
                    print(f"❌ No saved positions file found: {POSITIONS_FILE}")
                    return
                position_file = Path(POSITIONS_FILE)
                
            with open(position_file, "r") as f:
                save_data = json.load(f)
            
            print(f"\n📂 Using positions from: {position_file.name}")
            print(f"📅 Saved on: {save_data['timestamp']}")
            
            # Store initial position before any movement
            print("\n📍 Storing initial position...")
            initial_positions = get_positions_from_device(device)
            print("Initial Positions:")
            for motor, position in initial_positions.items():
                print(f"  {motor:15s}: {position:8.2f}")
            
            # Move to target positions smoothly
            print(f"\n🎯 Moving to saved positions...")
            smooth_move_to_positions(device, save_data["positions"])
            
            # Wait 5 seconds
            print(f"\n⏱️  Waiting 5 seconds...")
            for i in range(5, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
            
            # Move back to initial positions smoothly
            print(f"\n🔄 Returning to initial position...")
            smooth_move_to_positions(device, initial_positions)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'device' in locals() and hasattr(device, 'is_connected') and device.is_connected:
            print("\n🔌 Disconnecting from device...")
            device.disconnect()
            print("✅ Disconnected successfully")

if __name__ == "__main__":
    main() 