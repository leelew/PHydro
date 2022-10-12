#!/bin/bash
#cd /tera05/lilu/PHydro/input/
#python3 test_CoLM_wberror.py
#cd /tera05/lilu/PHydro/preprocess/
#python3 make_input_data.py
#cd /tera05/lilu/PHydro/
python3 src/cal_perf.py --model_name "single_task" \
	            --inputs_path "/work/PHydro/input/" \
		    --outputs_path "/work/PHydro/output/"

