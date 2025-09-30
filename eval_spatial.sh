ckpt=$1
work_dir=$2
if [ ! -d "output/moe/eval_result/$work_dir" ]; then
    accelerate launch --main_process_port 21568 eval_spatial_align.py --task pose --ckpt $ckpt --work_dir $work_dir --gen_gt_img
    accelerate launch --main_process_port 21568 eval_spatial_align.py --task canny --ckpt $ckpt --work_dir $work_dir --gen_gt_img
    accelerate launch --main_process_port 21568 eval_spatial_align.py --task depth --ckpt $ckpt --work_dir $work_dir --gen_gt_img
    python eval/generate_conditions.py --task all --input_dir output/moe/eval_result/$work_dir
    python eval/eval_conditions.py --anno_dir /opt/tiger/efficient_ai/UniCombine/output/moe/eval_result/typeD_tokenS_new_49k --pred_dir output/moe/eval_result/$work_dir
fi