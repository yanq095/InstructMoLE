export config_path="./train/config/XVerse_config_INF.yaml"
export model_checkpoint="./checkpoints/XVerse"
export target_size=768
export condition_size=256
export test_list_name="XVerseBench_single"
export save_name="./eval/XVerseBench_singleidip"

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[-1]}

accelerate launch \
    --main_process_port $port \
    -m eval.tools.idip_gen_split_idip \
    --config_name "$config_path" \
    --model_path "$model_checkpoint" \
    --target_size "$target_size" \
    --condition_size "$condition_size" \
    --save_name "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    --main_process_port $port \
    -m eval.tools.idip_dpg_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    --main_process_port $port \
    -m eval.tools.idip_aes_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    --main_process_port $port \
    -m eval.tools.idip_face_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

accelerate launch \
    --main_process_port $port \
    -m eval.tools.idip_sam-dino_score \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"

python \
    -m eval.tools.log_scores \
    --input_dir "$save_name" \
    --test_list_name "$test_list_name"
