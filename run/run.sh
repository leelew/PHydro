#!/bin/bash

work_path="/data/lilu/PHydro"
inputs_path=$work_path/input/
outputs_path=$work_path/output/
model_name="multi_tasks"
epochs=1
num_repeat=1

nohup python3 $work_path/src/main.py --num_repeat $num_repeat --epochs $epochs --work_path $work_path --model_name $model_name --inputs_path $inputs_path --outputs_path $outputs_path >> $work_path/logs/$model_name.log 2>&1 &

