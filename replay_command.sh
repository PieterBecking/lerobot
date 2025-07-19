#!/bin/bash

# Simple script to replay recorded episodes from robot_recordings/ folder
# Usage: ./replay_command.sh [task_name]

TASK_NAME=${1:-"staple_demo"}

echo "🎬 Replaying episode: $TASK_NAME"
echo "📁 Loading from: robot_recordings/$TASK_NAME/"
echo "Press Ctrl+C to stop replay at any time"
echo ""

# Check if the dataset exists
if [ ! -d "robot_recordings/$TASK_NAME" ]; then
    echo "❌ Error: Dataset 'robot_recordings/$TASK_NAME' not found!"
    echo "Make sure you have recorded this episode first using:"
    echo "   ./record_command.sh $TASK_NAME"
    exit 1
fi

# Check if the parquet file exists
PARQUET_FILE="robot_recordings/$TASK_NAME/data/chunk-000/episode_000000.parquet"
if [ ! -f "$PARQUET_FILE" ]; then
    echo "❌ Error: Episode data file not found!"
    echo "Expected: $PARQUET_FILE"
    exit 1
fi

# Python script to replay the episode
python3 << EOF
import time
import pandas as pd
from lerobot.common.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.common.utils.robot_utils import busy_wait

def replay_episode():
    # Robot setup (same as teleoperation config)
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A680106981",
        id="s7_follower_arm",
    )
    robot = SO100Follower(robot_config)
    
    # Load the parquet file
    parquet_path = "$PARQUET_FILE"
    print(f"📁 Loading: {parquet_path}")
    
    try:
        df = pd.read_parquet(parquet_path)
        print(f"📊 Loaded {len(df)} frames successfully!")
        
        if 'action' not in df.columns:
            print("❌ No action data found!")
            return
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Connect to robot
    print("🔌 Connecting to robot...")
    robot.connect()
    
    try:
        print("⏳ Starting replay in 3 seconds...")
        time.sleep(3)
        
        print("🎯 Replaying '$TASK_NAME' episode...")
        
        # Action names from dataset
        action_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
                       "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
        
        fps = 30  # Original recording fps
        
        for idx in range(len(df)):
            start_time = time.perf_counter()
            
            # Get action for this frame
            action_array = df.iloc[idx]['action']
            action = {name: float(action_array[i]) for i, name in enumerate(action_names)}
            
            print(f"Frame {idx:3d}/{len(df)}: Moving robot...")
            
            # Send action to robot
            try:
                robot.send_action(action)
            except Exception as e:
                print(f"⚠️  Robot error at frame {idx}: {e}")
                continue
            
            # Maintain timing
            dt_s = time.perf_counter() - start_time
            busy_wait(1/fps - dt_s)
            
        print("✅ Replay completed successfully!")
        
    except KeyboardInterrupt:
        print("⏹️  Replay stopped by user")
    except Exception as e:
        print(f"❌ Error during replay: {e}")
    finally:
        robot.disconnect()
        print("🔌 Robot disconnected")

if __name__ == "__main__":
    replay_episode()
EOF

echo ""
echo "🎬 Replay of '$TASK_NAME' completed!" 