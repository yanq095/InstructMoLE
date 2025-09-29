for dir in output/moe/*/checkpoint-20000; do
    if [ -d "$dir" ]; then
        echo "$dir"
        temp_path="${dir#output/moe/}"
        moe_dir_name="${temp_path%/checkpoint-20000}"
        echo "$moe_dir_name"
        # bash eval_edit.sh $dir "$moe_dir_name"_false
        bash eval_edit.sh $dir "$moe_dir_name"
        echo "------------------------------------"
    fi
done

bash eval_edit.sh output/moe/typeD_tokenS_new_49k typeD_tokenS_new_49k
bash eval_edit.sh output/moe/typeD_tokenS_old_69k typeD_tokenS_old_69k
cd ..
python gpu_occupy.py 