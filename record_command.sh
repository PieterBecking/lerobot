#!/bin/bash

# Simple script to record episodes with SO100 leader-follower setup
# Usage: ./record_command.sh [episode_name]

EPISODE_NAME=${1:-"demo_episode"}

echo "Recording episode: $EPISODE_NAME"
echo "This will be saved locally in ~/.cache/huggingface/lerobot/"
echo "Use Ctrl+C to stop early, or Right arrow key during recording"
echo ""

python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem5A680106981 \
    --robot.id=s7_follower_arm \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem5A460843481 \
    --teleop.id=s7_leader_arm \
    --dataset.repo_id=local/$EPISODE_NAME \
    --dataset.single_task=$EPISODE_NAME \
    --dataset.num_episodes=1 \
    --dataset.root=robot_recordings/$EPISODE_NAME \
    --dataset.push_to_hub=false \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --display_data=true

echo "Recording completed! Data saved in robot_recordings/$EPISODE_NAME" 