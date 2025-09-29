# export CKPT="/opt/tiger/efficient_ai/UniCombine/output/train_result/moe_V2.5_typeS_tokenD/checkpoint-20000/"
# export save_name="./output/bench/moe_V2.5_typeS_tokenD"
# bash eval/eval_scripts/run_eval_multi.sh 

# export CKPT="/opt/tiger/efficient_ai/UniCombine/output/ablation/moe_V2.5_fixed/checkpoint-20000/"
# export save_name="./output/bench/moe_V2.5_fixed"
# bash eval/eval_scripts/run_eval_multi.sh 

export CKPT=$1
export save_name="./output/moe/xverse_multi_bench/$2"
bash eval/eval_scripts/run_eval_multi.sh 

# cd ..
# python gpu_occupy.py
