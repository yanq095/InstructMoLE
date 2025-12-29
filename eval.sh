for dir in output/moe/*/checkpoint; do
    if [ -d "$dir" ]; then
        echo "$dir"
        temp_path="${dir#output/moe/}"
        moe_dir_name="${temp_path%/checkpoint}"
        echo "$moe_dir_name"
        bash eval_edit.sh $dir "$moe_dir_name"
        echo "------------------------------------"
    fi
done

bash eval_edit.sh your_model weights
