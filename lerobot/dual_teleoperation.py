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
Dual SO100 Teleoperation Script

This script provides two main functionalities:
1. Port registration for 4 SO100 arms (2 leaders + 2 followers)
2. Dual arm teleoperation using the registered ports

The script will guide you through disconnecting and reconnecting each arm
to automatically detect their USB ports, then save the configuration to a JSON file.
After registration, you can run dual arm teleoperation.

Usage:
    # Register ports only
    python -m lerobot.dual_teleoperation --register-ports
    
    # Run dual teleoperation (requires existing port config)
    python -m lerobot.dual_teleoperation --teleop
    
    # Register ports and then run teleoperation
    python -m lerobot.dual_teleoperation --register-ports --teleop

Example:
    python -m lerobot.dual_teleoperation --teleop --fps 30 --display-data
"""

import argparse
import asyncio
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Optional

import draccus
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so100_leader,
)
from lerobot.common.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun


@dataclass
class DualTeleoperateConfig:
    """Configuration for dual arm teleoperation."""
    # Port configuration file path
    ports_config_path: str = "dual_so100_ports.json"
    # Limit the maximum frames per second
    fps: int = 60
    # Duration of teleoperation in seconds (None = infinite)
    teleop_time_s: Optional[float] = None
    # Display all cameras and data on screen
    display_data: bool = False
    # Camera configurations (optional)
    left_cameras: Optional[dict] = None
    right_cameras: Optional[dict] = None


def find_available_ports():
    """Find all available USB ports on the system."""
    from serial.tools import list_ports  # Part of pyserial library

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_device_port(device_name: str) -> str:
    """
    Find the USB port for a specific device by having the user disconnect and reconnect it.
    
    Args:
        device_name: Human-readable name of the device (e.g., "Left Leader Arm")
        
    Returns:
        str: The USB port path for the device
    """
    print(f"\n{'='*60}")
    print(f"🔍 DETECTING PORT FOR: {device_name.upper()}")
    print(f"{'='*60}")
    
    print("📋 Current connected ports:")
    ports_before = find_available_ports()
    for i, port in enumerate(sorted(ports_before), 1):
        print(f"   {i}. {port}")
    
    print(f"\n🔌 Please DISCONNECT the USB cable from your {device_name}")
    print("   and press ENTER when done...")
    input()

    print("⏳ Detecting port change...")
    time.sleep(1.0)  # Allow time for port to be released
    
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        print(f"✅ Detected port: {port}")
        print(f"🔌 Please RECONNECT the USB cable for {device_name}")
        input("   Press ENTER when reconnected...")
        print(f"✅ {device_name} registered successfully!")
        return port
    elif len(ports_diff) == 0:
        raise OSError(f"❌ Could not detect port for {device_name}. No port change detected.")
    else:
        raise OSError(f"❌ Could not detect port for {device_name}. Multiple ports changed: {ports_diff}")


def save_config_to_json(config: dict, config_path: Path):
    """Save the port configuration to a JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to: {config_path}")


def load_existing_config(config_path: Path) -> dict:
    """Load existing configuration if it exists."""
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def print_banner():
    """Print a welcome banner."""
    print("""
🤖 DUAL SO100 TELEOPERATION SCRIPT 🤖
=====================================

This script provides:
  1. Port registration for 4 SO100 arms
  2. Dual arm teleoperation control

Arms configuration:
  - Left Leader Arm (teleoperator)
  - Right Leader Arm (teleoperator)  
  - Left Follower Arm (robot)
  - Right Follower Arm (robot)
""")


def print_summary(config: dict):
    """Print a summary of the detected configuration."""
    print(f"\n{'='*60}")
    print("🎉 PORT REGISTRATION COMPLETE!")
    print(f"{'='*60}")
    print("📋 Detected Configuration:")
    print(f"   Left Leader:    {config['left_leader']['port']}")
    print(f"   Right Leader:   {config['right_leader']['port']}")
    print(f"   Left Follower:  {config['left_follower']['port']}")
    print(f"   Right Follower: {config['right_follower']['port']}")
    print("\n✅ You can now use these ports for dual arm teleoperation!")


async def arm_teleoperation_loop(
    teleop: Teleoperator,
    robot: Robot, 
    arm_name: str,
    fps: int,
    display_data: bool = False,
    duration: Optional[float] = None
):
    """
    Async teleoperation loop for a single arm pair.
    
    Args:
        teleop: Teleoperator instance
        robot: Robot instance  
        arm_name: Name for this arm (e.g., "left", "right")
        fps: Target frames per second
        display_data: Whether to display data in rerun
        duration: Duration to run (None = infinite)
    """
    print(f"🤖 Starting {arm_name} arm teleoperation...")
    
    start_time = time.perf_counter()
    loop_count = 0
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Get action from teleoperator
            action = teleop.get_action()
            
            # Log data if display is enabled
            if display_data:
                try:
                    observation = robot.get_observation()
                    for obs_key, obs_val in observation.items():
                        if isinstance(obs_val, float):
                            rr.log(f"{arm_name}/observation_{obs_key}", rr.Scalars([obs_val]))
                        elif isinstance(obs_val, np.ndarray):
                            rr.log(f"{arm_name}/observation_{obs_key}", rr.Image(obs_val), static=True)
                except Exception as e:
                    logging.warning(f"Failed to get observation for {arm_name}: {e}")
                
                for act_key, act_val in action.items():
                    if isinstance(act_val, float):
                        rr.log(f"{arm_name}/action_{act_key}", rr.Scalars([act_val]))
            
            # Send action to robot
            robot.send_action(action)
            
            # Wait for next cycle
            dt_s = time.perf_counter() - loop_start
            if dt_s < 1 / fps:
                await asyncio.sleep(1 / fps - dt_s)
            
            loop_count += 1
            
            # Check duration
            if duration is not None and time.perf_counter() - start_time >= duration:
                break
                
            # Print status every 100 loops
            if loop_count % 100 == 0:
                elapsed = time.perf_counter() - start_time
                avg_hz = loop_count / elapsed
                print(f"📊 {arm_name.capitalize()} arm: {loop_count} loops, {avg_hz:.1f} Hz avg")
                
    except asyncio.CancelledError:
        print(f"🛑 {arm_name.capitalize()} arm teleoperation cancelled")
    except Exception as e:
        print(f"❌ Error in {arm_name} arm teleoperation: {e}")
        logging.exception(f"Error in {arm_name} arm teleoperation")
    finally:
        print(f"🏁 {arm_name.capitalize()} arm teleoperation finished")


def load_ports_config(config_path: str) -> dict:
    """Load the ports configuration from JSON file."""
    # Try as absolute path first
    config_file = Path(config_path)
    
    # If not found and not absolute, try relative to package directory
    if not config_file.exists() and not config_file.is_absolute():
        package_dir = Path(__file__).parent
        # First try in configs/ports directory
        config_file = package_dir / "configs" / "ports" / config_path
        # If not found there, try package directory (backward compatibility)
        if not config_file.exists():
            config_file = package_dir / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Port configuration file not found at {config_path} or {config_file}\n"
            "Please run with --register-ports first to create the configuration."
        )
    
    with open(config_file, 'r') as f:
        return json.load(f)


def create_arm_configs(ports_config: dict, cameras_config: Optional[dict] = None) -> tuple:
    """
    Create teleoperator and robot configurations from ports config.
    
    Args:
        ports_config: Dictionary with port configurations
        cameras_config: Optional camera configurations
        
    Returns:
        Tuple of (left_teleop_config, left_robot_config, right_teleop_config, right_robot_config)
    """
    # Left arm configurations
    left_teleop_config = SO100LeaderConfig(
        port=ports_config["left_leader"]["port"],
        id="left_leader"
    )
    
    left_robot_config = SO100FollowerConfig(
        port=ports_config["left_follower"]["port"],
        id="left_follower",
        cameras=cameras_config.get("left", {}) if cameras_config else {}
    )
    
    # Right arm configurations  
    right_teleop_config = SO100LeaderConfig(
        port=ports_config["right_leader"]["port"],
        id="right_leader"
    )
    
    right_robot_config = SO100FollowerConfig(
        port=ports_config["right_follower"]["port"], 
        id="right_follower",
        cameras=cameras_config.get("right", {}) if cameras_config else {}
    )
    
    return left_teleop_config, left_robot_config, right_teleop_config, right_robot_config


async def dual_teleoperation(cfg: DualTeleoperateConfig):
    """
    Run dual arm teleoperation using async coroutines.
    
    Args:
        cfg: Dual teleoperation configuration
    """
    init_logging()
    logging.info("Starting dual arm teleoperation")
    logging.info(pformat(asdict(cfg)))
    
    # Load port configuration
    try:
        ports_config = load_ports_config(cfg.ports_config_path)
        print("✅ Loaded port configuration")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Initialize rerun if display_data is enabled
    if cfg.display_data:
        _init_rerun(session_name="dual_teleoperation")
        print("📺 Rerun visualization initialized")
    
    # Create camera configurations
    cameras_config = {}
    if cfg.left_cameras:
        cameras_config["left"] = cfg.left_cameras
    if cfg.right_cameras:
        cameras_config["right"] = cfg.right_cameras
    
    # Create arm configurations
    left_teleop_config, left_robot_config, right_teleop_config, right_robot_config = create_arm_configs(
        ports_config, cameras_config
    )
    
    # Create teleoperator and robot instances
    print("🔧 Creating teleoperator and robot instances...")
    try:
        left_teleop = make_teleoperator_from_config(left_teleop_config)
        left_robot = make_robot_from_config(left_robot_config)
        right_teleop = make_teleoperator_from_config(right_teleop_config)
        right_robot = make_robot_from_config(right_robot_config)
        print("✅ All instances created successfully")
    except Exception as e:
        print(f"❌ Failed to create instances: {e}")
        logging.exception("Failed to create instances")
        return
    
    # Connect all devices
    print("🔌 Connecting to devices...")
    try:
        left_teleop.connect()
        print("   ✅ Left leader connected")
        
        left_robot.connect()
        print("   ✅ Left follower connected")
        
        right_teleop.connect()
        print("   ✅ Right leader connected")
        
        right_robot.connect()
        print("   ✅ Right follower connected")
        
        print("🎉 All devices connected successfully!")
        
    except Exception as e:
        print(f"❌ Failed to connect devices: {e}")
        logging.exception("Failed to connect devices")
        return
    
    print(f"\n🚀 Starting dual arm teleoperation at {cfg.fps} Hz...")
    if cfg.teleop_time_s:
        print(f"⏱️  Duration: {cfg.teleop_time_s} seconds")
    else:
        print("⏱️  Duration: Until interrupted (Ctrl+C)")
    
    print("\n" + "="*60)
    print("🤖 DUAL ARM TELEOPERATION ACTIVE")
    print("="*60)
    print("Press Ctrl+C to stop teleoperation")
    print("="*60)
    
    # Run dual arm teleoperation
    try:
        # Create async tasks for both arms
        left_task = asyncio.create_task(
            arm_teleoperation_loop(
                left_teleop, left_robot, "left", 
                cfg.fps, cfg.display_data, cfg.teleop_time_s
            )
        )
        
        right_task = asyncio.create_task(
            arm_teleoperation_loop(
                right_teleop, right_robot, "right",
                cfg.fps, cfg.display_data, cfg.teleop_time_s
            )
        )
        
        # Wait for both tasks to complete
        await asyncio.gather(left_task, right_task)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Teleoperation interrupted by user")
        # Cancel tasks
        left_task.cancel()
        right_task.cancel()
        try:
            await asyncio.gather(left_task, right_task, return_exceptions=True)
        except:
            pass
        
    except Exception as e:
        print(f"\n❌ Error during teleoperation: {e}")
        logging.exception("Error during teleoperation")
        
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        try:
            if cfg.display_data:
                rr.rerun_shutdown()
            
            left_teleop.disconnect()
            left_robot.disconnect()
            right_teleop.disconnect()  
            right_robot.disconnect()
            print("✅ All devices disconnected")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
        
        print("🏁 Dual arm teleoperation finished")


def list_and_select_port_configs() -> Optional[Path]:
    """
    List available port configurations and let user select one or create new.
    
    Returns:
        Optional[Path]: Selected or created config path, or None if cancelled
    """
    # Get all JSON files in the ports directory
    ports_dir = Path(__file__).parent / "configs" / "ports"
    ports_dir.mkdir(parents=True, exist_ok=True)
    
    config_files = list(ports_dir.glob("*.json"))
    
    if config_files:
        print("\n📄 Available port configurations:")
        for i, config_file in enumerate(config_files, 1):
            # Load and display config summary
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"\n{i}. {config_file.name}")
                for arm_name, arm_config in config.items():
                    print(f"     {arm_name}: {arm_config.get('port', 'not set')}")
            except:
                print(f"\n{i}. {config_file.name} (invalid format)")
        
        print("\n0. Create new configuration")
        
        while True:
            choice = input("\n❓ Select configuration (0-{}, or q to quit): ".format(len(config_files))).strip().lower()
            
            if choice == 'q':
                return None
            
            try:
                choice_num = int(choice)
                if choice_num == 0:
                    break  # Create new config
                elif 1 <= choice_num <= len(config_files):
                    return config_files[choice_num - 1]
            except ValueError:
                pass
            
            print("❌ Invalid choice, please try again")
    else:
        print("\n📄 No existing port configurations found.")
        if input("❓ Create new configuration? (Y/n): ").strip().lower() not in ['n', 'no']:
            pass  # Continue to create new
        else:
            return None
    
    # Get name for new configuration
    while True:
        config_name = input("\n📝 Enter name for new configuration (without .json): ").strip()
        if config_name:
            config_path = ports_dir / f"{config_name}.json"
            if config_path.exists():
                print("❌ Configuration with that name already exists!")
            else:
                return config_path
        else:
            print("❌ Please enter a valid name")
    
    return None


def register_ports():
    """Register ports for dual SO100 teleoperation."""
    print_banner()
    
    # Let user select or create configuration
    config_path = list_and_select_port_configs()
    if config_path is None:
        print("\n✋ Port registration cancelled.")
        return None
        
    # Load existing config if available
    existing_config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
            print(f"\n📄 Using existing configuration: {config_path}")
            print("   Current ports:")
            for arm_name, arm_config in existing_config.items():
                print(f"     {arm_name}: {arm_config.get('port', 'not set')}")
            
            overwrite = input("\n❓ Do you want to re-register all ports? (y/N): ").strip().lower()
            if overwrite not in ['y', 'yes']:
                print("✋ Keeping existing configuration.")
                return str(config_path)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load existing config: {e}")
    
    input("\n🚀 Press ENTER to start port registration...")
    
    # Device registration order and configuration
    devices = [
        ("left_leader", "Left Leader Arm", "so100_leader"),
        ("right_leader", "Right Leader Arm", "so100_leader"),
        ("left_follower", "Left Follower Arm", "so100_follower"),
        ("right_follower", "Right Follower Arm", "so100_follower")
    ]
    
    config = {}
    
    try:
        for device_key, device_name, device_type in devices:
            port = find_device_port(device_name)
            config[device_key] = {
                "type": device_type,
                "port": port,
                "id": f"{device_key}_arm",
                "name": device_name
            }
            time.sleep(0.5)  # Brief pause between registrations
        
        # Save configuration to JSON
        save_config_to_json(config, config_path)
        
        # Print summary
        print_summary(config)
        return str(config_path)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Registration interrupted by user.")
        if config:
            print("💾 Saving partial configuration...")
            save_config_to_json(config, config_path)
        return str(config_path) if config else None
    except Exception as e:
        print(f"\n❌ Error during registration: {e}")
        if config:
            print("💾 Saving partial configuration...")
            save_config_to_json(config, config_path)
        raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Dual SO100 Teleoperation - Port Registration and Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--register-ports", 
        action="store_true",
        help="Register USB ports for the 4 SO100 arms"
    )
    
    parser.add_argument(
        "--teleop",
        action="store_true", 
        help="Run dual arm teleoperation"
    )
    
    parser.add_argument(
        "--ports-config-path",
        type=str,
        default=None,  # Changed to None as default
        help="Path to the ports configuration JSON file (relative to configs/ports/ or absolute). If not provided, will prompt for selection."
    )
    
    parser.add_argument(
        "--fps", 
        type=int,
        default=60,
        help="Target frames per second for teleoperation"
    )
    
    parser.add_argument(
        "--teleop-time-s",
        type=float,
        default=None,
        help="Duration of teleoperation in seconds (None = infinite)"
    )
    
    parser.add_argument(
        "--display-data",
        action="store_true",
        help="Display cameras and data using rerun visualization"
    )
    
    args = parser.parse_args()
    
    # If no action specified, default to showing help
    if not args.register_ports and not args.teleop:
        parser.print_help()
        return

    # Handle port configuration selection
    config_path = None
    if args.ports_config_path:
        # If path provided, use it directly
        config_path = Path(args.ports_config_path)
    else:
        # Otherwise, show selection menu
        print_banner()
        selected_path = list_and_select_port_configs()
        if selected_path is None:
            if args.register_ports:
                print("\n✋ Port registration cancelled.")
            else:
                print("\n❌ No port configuration selected. Please either:")
                print("   1. Provide --ports-config-path")
                print("   2. Run with --register-ports to create a configuration")
            return
        config_path = selected_path
    
    # Register ports if requested
    if args.register_ports:
        try:
            config_path = register_ports()
            if not config_path:
                return
        except Exception as e:
            print(f"❌ Port registration failed: {e}")
            return
    
    # Run teleoperation if requested
    if args.teleop:
        try:
            cfg = DualTeleoperateConfig(
                ports_config_path=str(config_path),
                fps=args.fps,
                teleop_time_s=args.teleop_time_s,
                display_data=args.display_data
            )
            
            # Run async teleoperation
            asyncio.run(dual_teleoperation(cfg))
            
        except Exception as e:
            print(f"❌ Teleoperation failed: {e}")
            logging.exception("Teleoperation failed")


if __name__ == "__main__":
    main()
