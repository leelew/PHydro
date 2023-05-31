#!/bin/bash

work_path="/data/lilu/PHydro_era"
inputs_path=$work_path/input/
outputs_path=$work_path/output/
model_name="hard_multi_tasks_v3"
epochs=400
num_repeat=5
scaling_factor=5
main_idx=2
alpha=0.1



nohup python3 $work_path/src/main.py --alpha $alpha --main_idx $main_idx  --scaling_factor $scaling_factor --num_repeat $num_repeat --epochs $epochs --work_path $work_path --model_name $model_name --inputs_path $inputs_path --outputs_path $outputs_path >> $work_path/logs/$model_name.log 2>&1 &

