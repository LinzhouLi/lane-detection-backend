#!/bin/bash
# Trained weights: resnet18_bezierlanenet_tusimple-aug1b_20211109.pt
# Training
python3 main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_tusimple-aug1b.py
# Predicting lane points for testing
python3 main_landet.py --test --config=configs/lane_detection/bezierlanenet/resnet18_tusimple-aug1b.py
# Testing with official scripts
./autotest_tusimple.sh resnet18_bezierlanenet_tusimple-aug2 test checkpoints
