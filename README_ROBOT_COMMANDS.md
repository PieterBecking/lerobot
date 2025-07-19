# 🤖 Robot Recording & Replay Commands

Simple workflow for recording and replaying robot episodes locally.

## Quick Start

### 1. Record an Episode
```bash
./record_command.sh my_task_name
```

### 2. Replay an Episode  
```bash
./replay_command.sh my_task_name
```

## Features

✅ **Local Storage Only** - Everything saved in `robot_recordings/` folder  
✅ **No Hugging Face** - No uploads to cloud  
✅ **Simple Commands** - Just task name as argument  
✅ **Error Handling** - Validates datasets exist before replay  
✅ **Same Robot Config** - Uses your working teleoperation setup  
✅ **Auto-formatting** - Dataset names automatically prefixed with "local/"  

## Examples

```bash
# Record a stapling task
./record_command.sh staple

# Record a cup pickup task  
./record_command.sh pickup_cup

# Record object placement
./record_command.sh place_object

# Replay any recorded task
./replay_command.sh staple
./replay_command.sh pickup_cup
./replay_command.sh place_object
```

## Recording Controls

During recording:
- **Right arrow (→)**: Exit episode early
- **Left arrow (←)**: Re-record current episode  
- **Escape**: Stop recording completely
- **Ctrl+C**: Emergency stop

## Replay Controls

During replay:
- **Ctrl+C**: Stop replay immediately

## File Structure

```
robot_recordings/
├── staple/
│   ├── data/chunk-000/episode_000000.parquet
│   └── meta/
├── pickup_cup/
│   ├── data/chunk-000/episode_000000.parquet  
│   └── meta/
└── place_object/
    ├── data/chunk-000/episode_000000.parquet
    └── meta/
```

Each task gets its own directory under `robot_recordings/` containing the complete dataset structure.

## Robot Configuration

Both scripts use the same robot configuration that works with your teleoperation:

- **Follower Robot**: `/dev/tty.usbmodem5A680106981`
- **Leader Robot**: `/dev/tty.usbmodem5A460843481` 
- **Recording FPS**: 30
- **Episode Length**: 60 seconds max
- **Reset Time**: 10 seconds

## Technical Notes

- Dataset names are automatically prefixed with `local/` (e.g., `staple` becomes `local/staple`)
- This format is required by LeRobot's validation system
- Your recordings are still saved locally in `robot_recordings/[task_name]/`

Ready to create robot demonstrations! 🚀 