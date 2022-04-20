#!/bin/sh -x

d=.

export LD_LIBRARY_PATH=${d}

export PLAY_PTX_FILE=${d}/2080.ptx

export PLAY_LIVEVIDEOPORT=4071 # http://127.0.0.1:4071/mjpeg
export PLAY_DISPLAY_FRAMERATE=25
export PLAY_FRAMERATE=25

export PLAY_RENDER_IMAGE_WIDTH=1920
export PLAY_RENDER_IMAGE_HEIGHT=1080

export PLAY_WEBDISPLAY_PORT=8080

# on of:
#1)
#export PLAY_CONTROL_PORT=6080
#export PLAY_RECORDS_DIR=$d
#2)
export PLAY_ALL_FILE=${d}/44.glb


exec ${d}/gplay

