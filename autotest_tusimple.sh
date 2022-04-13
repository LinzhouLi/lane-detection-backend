#!/bin/bash
data_dir=../../docs/datasets/tusimple/
echo experiment name: $1
echo status: $2
echo save dir: $3

# ./autotest_<my_dataset>.sh <exp_name> <mode> <save_dir>

# Perform test/validation with official scripts
cd tools/tusimple_evaluation
if [ "$2" = "test" ]; then
    python3 lane.py ../../output/${1}.json ${data_dir}test_label.json $1 $3
else
    python3 lane.py ../../output/${1}.json ${data_dir}label_data_0531.json $1 $3
fi
cd ../../
