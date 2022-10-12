#!/bin/bash
#cd /tera05/lilu/PHydro/input/
#python3 test_CoLM_wberror.py
#cd /tera05/lilu/PHydro/preprocess/
#python3 make_input_data.py
#cd /tera05/lilu/PHydro/

work_path="/work/PHydro"
inputs_path=$work_path/input/
outputs_path=$work_path/output/
model_name="single_task"
epochs=1000

nohup python3 $work_path/src/main.py --work_path $work_path --model_name $model_name --inputs_path $inputs_path --outputs_path $outputs_path --epochs $epochs >> $work_path/logs/$model_name.log 2>&1 &

