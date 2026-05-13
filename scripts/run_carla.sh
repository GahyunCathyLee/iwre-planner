#!/bin/bash

cd $HOME/iwre-planner/carla-simulator

./CarlaUE4.sh \
  -RenderOffScreen \
  -quality-level=Low \
  -carla-rpc-port=2000 \
  -nosound