# generate_conditions.py (multi-gpu version with 'all' support)

import os
import argparse
from PIL import Image
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import pipeline

# 动态导入检查
try:
    from controlnet_aux import OpenposeDetector, CannyDetector
except ImportError:
    print("必需的库未安装。")
    print(
        "请运行: pip install torch torchvision transformers opencv-python-headless Pillow numpy tqdm einops torchmetrics accelerate controlnet_aux==0.0.7"
    )
    exit(1)

def get_image_files(directory):
    """获取目录中所有支持的图像文件"""
    return sorted(
        [
            f
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]
    )


# --- Worker Functions (保持不变) ---


def canny_worker(input_dir, output_dir, file_list, rank):
    """Canny边缘图生成的worker"""
    os.makedirs(output_dir, exist_ok=True)
    canny_detector = CannyDetector()

    # 只让 rank 0 的进程显示总进度条
    pbar = tqdm(file_list, desc=f"GPU {rank} [Canny]", disable=(rank != 0))
    for filename in pbar:
        image_path = os.path.join(input_dir, filename)
        try:
            input_image = Image.open(image_path)
            detected_map = canny_detector(
                input_image, low_threshold=100, high_threshold=200
            )
            save_path = os.path.join(output_dir, filename)
            detected_map.save(save_path)
        except Exception as e:
            print(f"GPU {rank} 处理文件 {filename} 时出错: {e}")


def depth_worker(input_dir, output_dir, file_list, rank):
    """深度图生成的worker"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{rank}")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="models/depth-anything-small-hf",
        device=device,
    )
    pbar = tqdm(file_list, desc=f"GPU {rank} [Depth]", disable=(rank != 0))
    for filename in pbar:
        image_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            result = depth_pipe(image)
            depth_map = result["depth"]
            save_path = os.path.join(output_dir, filename)
            depth_map.save(save_path)
        except Exception as e:
            print(f"GPU {rank} 处理文件 {filename} 时出错: {e}")


def openpose_worker(input_dir, output_dir, file_list, rank):
    """OpenPose姿态图生成的worker"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{rank}")
    # 推荐使用 Hub ID 自动下载和缓存。如果您已手动修改了 controlnet_aux 源码以修复 torch.load 问题，
    # 并且模型文件在本地，请使用您的本地路径 "models/annotators"
    openpose_detector = OpenposeDetector.from_pretrained(
        "models/lllyasviel/ControlNet/annotator/ckpts"
    )

    pbar = tqdm(file_list, desc=f"GPU {rank} [Pose]", disable=(rank != 0))
    for filename in pbar:
        image_path = os.path.join(input_dir, filename)
        try:
            input_image = Image.open(image_path)
            detected_map = openpose_detector(input_image, hand_and_face=False)
            save_path = os.path.join(output_dir, filename)
            detected_map.save(save_path)
        except Exception as e:
            print(f"GPU {rank} 处理文件 {filename} 时出错: {e}")


# --- 主启动函数 ---


def main_worker(rank, world_size, all_files, task_name, input_dir, output_dir):
    """每个GPU进程的入口函数。现在接收 task_name。"""
    files_for_this_rank = all_files[rank::world_size]
    print(
        f"GPU {rank}/{world_size} [任务: {task_name.upper()}]: 已分配 {len(files_for_this_rank)} 个文件。"
    )

    output_path = output_dir

    WORKER_FUNCTIONS = {
        "canny": canny_worker,
        "depth": depth_worker,
        "pose": openpose_worker,
    }

    worker_func = WORKER_FUNCTIONS[task_name]
    worker_func(input_dir, output_path, files_for_this_rank, rank)

    print(f"GPU {rank} 已完成任务: {task_name.upper()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从源图像并行生成 Canny, Depth, 或 OpenPose 条件图。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="包含源图像的目录。"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="保存生成条件图的根目录。"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["canny", "depth", "pose", "all"],
    )
    parser.add_argument(
        "--gen_gt_cond",
        action="store_true",
    )
    args = parser.parse_args()
    tasks = []
    if args.task == "all":
        tasks = ["canny", "depth"]
    else:
        tasks = [args.task]
    for task in tasks:
        if args.gen_gt_cond:
            input_dir = os.path.join(args.input_dir, "gt_" + task)
        else:
            input_dir = os.path.join(args.input_dir, "eval_" + task)
        # if args.output_dir is None:
        output_path = input_dir + "_cond"

        # --- 确定要执行的任务列表 ---
        tasks_to_run = [task]

        # --- 获取所有文件 (只执行一次) ---
        all_files = get_image_files(input_dir)
        if not all_files:
            print(f"输入目录 '{input_dir}' 中没有找到图片文件。")
            exit(0)

        # --- 依次执行任务列表中的每个任务 ---
        for current_task in tasks_to_run:
            print(f"\n{'='*20} 开始执行任务: {current_task.upper()} {'='*20}")

            # --- 并行处理逻辑 ---
            if not torch.cuda.is_available():
                print("CUDA不可用。将在单个CPU进程上运行。")
                WORKER_FUNCTIONS_CPU = {
                    "canny": canny_worker,
                    "depth": depth_worker,
                    "pose": openpose_worker,
                }
                worker_func = WORKER_FUNCTIONS_CPU[current_task]
                worker_func(input_dir, output_path, all_files, 0)
            else:
                world_size = torch.cuda.device_count()
                print(
                    f"检测到 {world_size} 个GPU。为任务 '{current_task}' 开始并行处理..."
                )

                # 准备传递给每个子进程的参数，现在包括 current_task
                spawn_args = (
                    world_size,
                    all_files,
                    current_task,
                    input_dir,
                    output_path,
                )

                # 为当前任务启动并行进程
                mp.spawn(main_worker, args=spawn_args, nprocs=world_size, join=True)

            print(f"{'='*20} 任务完成: {current_task.upper()} {'='*20}\n")

    print("\n所有指定任务已完成！")
