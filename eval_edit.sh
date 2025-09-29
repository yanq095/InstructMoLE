
# out="output/moe/complex_edit/$2"
# # if [ ! -d "$out" ] || [ -z "$(ls -A "$out" 2>/dev/null)" ]; then
# if [ ! -d "$out" ]; then
#     mkdir -p $out
#     accelerate launch --main_process_port 21568  eval_kontext.py --work_dir $2 --ckpt $1 --task complex_edit
#     python eval/image_editing_eval/complex-edit/eval.py --path output/moe/complex_edit/$2   -c 8   --image-type real   -n 2   -m 5   --num-processes 16 
# fi
# out="output/moe/omnicontext/$2"
# # if [ ! -d "$out" ] || [ -z "$(ls -A "$out" 2>/dev/null)" ]; then
# if [ ! -d "$out" ]; then
#     mkdir -p $out
#     accelerate launch --main_process_port 21568  eval_kontext.py --work_dir $2 --ckpt $1 --task omnicontext
#     python eval/image_editing_eval/omnicontext/test_omnicontext_score.py \
#       --result_dir output/moe/omnicontext \
#       --model_name $2
#     python eval/image_editing_eval/omnicontext/calculate_statistics.py \
#       --save_path output/moe/omnicontext \
#       --model_name $2
# fi
out="output/moe/gedit/$2"
# if [ ! -d "$out" ] || [ -z "$(ls -A "$out" 2>/dev/null)" ]; then
if [ ! -d "$out" ]; then
    mkdir -p $out
    if [ ! -d output/$1 ]; then
      mkdir -p output/$2
      cp $1/*pt output/$2
      cp $1/*safetensors output/$2
      cp $1/train_config.json output/$2
    fi
    accelerate launch --main_process_port 21568  eval_kontext.py --work_dir $2 --ckpt output/$2 --task gedit
fi
out="output/moe/imagedit/$2"
# if [ ! -d "$out" ] || [ -z "$(ls -A "$out" 2>/dev/null)" ]; then
if [ ! -d "$out" ]; then
    mkdir -p $out
    if [ ! -d output/$1 ]; then
      mkdir -p output/$2
      cp $1/*pt output/$2
      cp $1/*safetensors output/$2
      cp $1/train_config.json output/$2
    fi
    accelerate launch --main_process_port 21568  eval_kontext.py --work_dir $2 --ckpt output/$2 --task imagedit
fi