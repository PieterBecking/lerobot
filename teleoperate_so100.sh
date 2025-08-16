#!/bin/bash

# SO-100 Teleoperation Script
# Configure the ports and settings for SO-100 leader and follower

# Default ports - modify these for your specific setup
FOLLOWER_PORT="/dev/tty.usbmodem5A680097171"
ROBOT_ID="andrej"

# Default robot and teleoperator IDs
LEADER_PORT="/dev/tty.usbmodem5A460843481"
TELEOP_ID="lex"

# Run teleoperation without display_data to avoid observation calls
python -m lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$ROBOT_ID" \
    --teleop.type=so100_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$TELEOP_ID" \
    --display_data=false