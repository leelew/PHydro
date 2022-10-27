#!/bin/bash

work_path="/data/lilu/PHydro"
inputs_path=$work_path/input/
outputs_path=$work_path/output/
model_name="soft_multi_tasks"

python3 $work_path/src/postprocess.py --model_name $model_name --inputs_path $inputs_path --outputs_path $outputs_path

