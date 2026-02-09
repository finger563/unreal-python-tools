# Python Tools for Unreal Engine 5 / RAMMS

Python libraries and scripts for interfacing with Unreal Engine 5, particularly the RAMMS centralized camera capture system.

## Features

- **EXR File Loading**: Load multi-channel EXR files with RGB, depth, and motion vectors
- **Depth Visualization**: Colorized depth maps with automatic range adjustment
- **Motion Vector Visualization**: HSV-encoded optical flow visualization
- **Multi-Camera Display**: View all cameras from an actor simultaneously in grid layout
- **Interactive Playback**: Play sequences with variable speed control
- **Legacy Format Support**: Also works with older raw binary format

Currently supports:
- RAMMS Centralized Camera Capture System (EXR + JSON format)
- [Camera Capture Component](https://github.com/finger563/unreal-camera-capture) (legacy raw format)
- [RTSP Display](https://github.com/finger563/unreal-rtsp-display)

## Setup

```sh
# Set up Python environment
python3 -m venv env

# Activate environment
# Linux/Mac:
source env/bin/activate
# Windows:
env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Quick Start

```sh
# View first available camera
python display_raw.py ../Ramms/Saved/CameraCaptures

# View specific camera
python display_raw.py ../Ramms/Saved/CameraCaptures --actor BP_Kinova_Gen3_C_UAID_XXX --camera Gripper_CaptureCamera

# View all cameras from an actor in grid
python display_raw.py ../Ramms/Saved/CameraCaptures --actor BP_Mebot_Ramms_C_0 --all-cameras

# Start from specific frame
python display_raw.py ../Ramms/Saved/CameraCaptures --actor ActorName --all-cameras --start-frame 100
```

### Controls
- **Space**: Play/Pause
- **N**: Next frame
- **P**: Previous frame
- **Q** or **ESC**: Quit

## Data Format

### RAMMS Centralized Format

Directory structure:
```
Saved/CameraCaptures/
├── ActorName1/
│   ├── CameraName1/
│   │   ├── frame_0000000.exr    # RGB + Depth + Motion
│   │   ├── frame_0000000.json   # Metadata
│   │   └── ...
│   └── CameraName2/
│       └── ...
└── ActorName2/
    └── ...
```

**EXR Channels:**
- R, G, B: RGB color (float 0-1, linear)
- A: Depth (float, normalized 0-1, represents 0-10000cm)
- Future: Dedicated Depth, MotionX, MotionY channels

**JSON Metadata** includes:
- Frame number, timestamp
- Camera transform (position, rotation)
- Intrinsics (focal length, principal point)
- Image dimensions

### Legacy Format

Single directory with:
- `camera_config.csv`: Camera configuration
- `CameraName_N.raw`: Raw binary image data
- `transformations.csv`: Camera poses

## Python API

```python
from data_loader import DataLoader

# Initialize
loader = DataLoader('Saved/CameraCaptures')

# Discover actors and cameras
print(loader.actors)  # Dict[actor_name, List[camera_name]]

# Load frame with all channels
frame_data = loader.load_frame_exr('ActorName', 'CameraName', frame_number=0)
# Returns dict with: rgb, depth, motion_x, motion_y, metadata, width, height

# Load metadata
metadata = loader.load_frame_metadata('ActorName', 'CameraName', 0)

# Get frame count
count = loader.get_frame_count('ActorName', 'CameraName')

# Convert for display
rgb_bgr = loader.convert_rgb_for_display(frame_data['rgb'])
depth_color = loader.convert_depth_for_display(frame_data['depth'])
motion_color = loader.convert_motion_for_display(frame_data['motion_x'], frame_data['motion_y'])
```

## Visualization Details

**RGB**: Linear float (0-1) → sRGB uint8 (0-255) → BGR for OpenCV

**Depth**: 
- Auto-range based on mean ± 2σ
- Invalid depths (0 or too large) shown as black
- Statistics printed to console

**Motion Vectors**:
- HSV encoding: Hue=direction, Saturation=magnitude
- Zero motion shown as black

## Troubleshooting

### OpenEXR Not Found
```sh
# Windows
pip install OpenEXR

# Linux
sudo apt-get install libopenexr-dev
pip install OpenEXR

# macOS  
brew install openexr
pip install OpenEXR
```

### No Data Found
- Verify path points to `Saved/CameraCaptures`
- Check actor/camera subdirectories exist
- Ensure `frame_*.exr` files are present

## Legacy Usage

```sh
# Legacy format (pre-centralized system)
python display_raw.py ../unreal-camera-capture-example/camera_data front
```

