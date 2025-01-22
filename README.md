# Python Tools for Unreal Engine 5

Some python libraries and scripts for interfacing with Unreal Engine 5, the
plugins I've made, as well as the data they produce.

Currently useful for interfacing with:
- [Camera Capture Component](https://github.com/finger563/unreal-camera-capture)
- [RTSP Display](https://github.com/finger563/unreal-rtsp-display)

## Setup

``` sh
# set up the python environment
python3 -m venv env
# source the new environment
source env/bin/activate
# install the requirements
```

## Usage

``` sh
# source the new environment
source env/bin/activate
# run the scripts you want, e.g.:
python ./display_raw.py ../unreal-camera-capture-example/camera_data front
```

