#!/usr/bin/bash
accelerate launch --main_process_port 12345 train_kontext.py --work_dir output