ckpt=$1
work_dir=$2
if [ ! -d "output/moe/eval_result/$work_dir" ]; then
    accelerate launch --main_process_port 21568 eval_spatial_align.py --task pose --ckpt $ckpt --work_dir $work_dir --gen_gt_img
    accelerate launch --main_process_port 21568 eval_spatial_align.py --task canny --ckpt $ckpt --work_dir $work_dir --gen_gt_img
    accelerate launch --main_process_port 21568 eval_spatial_align.py --task depth --ckpt $ckpt --work_dir $work_dir --gen_gt_img
    python eval/generate_conditions.py --task all --input_dir output/moe/eval_result/$work_dir
    python eval/eval_conditions.py --anno_dir /opt/tiger/efficient_ai/UniCombine/output/moe/eval_result/typeD_tokenS_new_49k --pred_dir output/moe/eval_result/$work_dir
fi
# fidelity --gpu 0,1,2,3,4,5,6,7 --fid --input1 output/eval_result/$work_dir/eval_depth --input2 output/eval_result/moe_typeD_tokenS/gt_depth  --json > output/eval_result/$work_dir/depth_fid.json
# fidelity --gpu 0,1,2,3,4,5,6,7 --fid --input1 output/eval_result/$work_dir/eval_canny --input2  output/eval_result/moe_typeD_tokenS/gt_canny  --json > output/eval_result/$work_dir/canny_fid.json
# python eval/preprocess_for_fid.py --input_dir  output/eval_result/$work_dir/eval_pose  --output_dir  output/eval_result/$work_dir/eval_pose_resized
# python eval/preprocess_for_fid.py --input_dir  output/eval_result/moe_typeD_tokenS/gt_pose  --output_dir output/eval_result/moe_typeD_tokenS/gt_pose_resized
# fidelity --gpu 0,1,2,3,4,5,6,7 --fid --input1 output/eval_result/$work_dir/eval_pose_resized --input2 output/eval_result/moe_typeD_tokenS/gt_pose_resized  --json > output/eval_result/$work_dir/pose_fid.json