#!/usr/bin/bash
accelerate launch --main_process_port "${KA_MASTER_PORT}" --num_processes "${TOTAL_PROCESSES}"  train_kontext.py --work_dir output