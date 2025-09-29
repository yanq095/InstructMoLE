# eval.py (multi-gpu pose evaluation with unified JSON output)

import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import torch.multiprocessing as mp

try:
    from torchmetrics.classification import BinaryF1Score
    from torchmetrics.image import (
        MultiScaleStructuralSimilarityIndexMeasure,
        PeakSignalNoiseRatio,
    )
except ImportError:
    print("错误: 库 'torchmetrics' 未安装。")
    print("请运行: pip install torchmetrics")
    exit(1)
try:
    from controlnet_aux.open_pose import PoseResult
    from controlnet_aux import OpenposeDetector
    from pycocotools.mask import iou
except ImportError:
    print("错误: 库 'controlnet_aux' 未安装。")
    print("请运行: pip install controlnet_aux==0.0.7")
    exit(1)


def get_main_person_keypoints(poses):
    if not isinstance(poses, PoseResult) or not poses.bodies:
        print("It's not a valid PoseResult object.", poses)
        return None
    best_body = max(poses.bodies, key=lambda body: body.total_score * body.total_person)
    return best_body.keypoints


def extract_keypoints_from_image(openpose_detector, image_path):
    try:
        input_image = Image.open(image_path)
        poses = openpose_detector(
            input_image, include_body=True, include_hand=False, include_face=False
        )
        return poses
    except Exception as e:
        print(f"处理文件 {image_path} 时出错: {e}")
        return None


def calculate_pck(gt_kpts, pred_kpts, threshold=0.5):
    L_SHOULDER_IDX, R_HIP_IDX = 5, 11
    gt_visible = [kpt for kpt in gt_kpts if kpt is not None and kpt.x > 0 and kpt.y > 0]
    pred_visible_dict = {
        kpt.idx: kpt for kpt in pred_kpts if kpt is not None and kpt.x > 0 and kpt.y > 0
    }
    if not gt_visible:
        return None
    gt_l_shoulder = next((kpt for kpt in gt_visible if kpt.idx == L_SHOULDER_IDX), None)
    gt_r_hip = next((kpt for kpt in gt_visible if kpt.idx == R_HIP_IDX), None)
    if not gt_l_shoulder or not gt_r_hip:
        return None
    torso_diameter = np.linalg.norm(
        [gt_l_shoulder.x - gt_r_hip.x, gt_l_shoulder.y - gt_r_hip.y]
    )
    if torso_diameter < 1e-4:
        return None
    correct_kpts_count, total_visible_kpts = 0, 0
    for gt_kpt in gt_visible:
        pred_kpt = pred_visible_dict.get(gt_kpt.idx)
        if pred_kpt:
            total_visible_kpts += 1
            distance = np.linalg.norm([gt_kpt.x - pred_kpt.x, gt_kpt.y - pred_kpt.y])
            if distance <= threshold * torso_diameter:
                correct_kpts_count += 1
    return correct_kpts_count / total_visible_kpts if total_visible_kpts > 0 else None


# =================================================================================
# 1. DEPTH EVALUATION LOGIC (增加返回值)
# =================================================================================
def evaluate_depth(anno_dir, pred_dir):
    """计算深度图的RMSE，并返回结果字典。"""
    print("\n--- 评估任务: Depth (RMSE) ---")
    # ... (函数主体代码保持不变) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")
    rmses = []
    anno_files = sorted(os.listdir(anno_dir))
    for filename in tqdm(anno_files, desc="评估 Depth"):
        anno_path = os.path.join(anno_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            continue
        try:
            anno_img = (
                torch.from_numpy(np.array(Image.open(anno_path).convert("L")))
                .to(device)
                .float()
            )
            pred_img = (
                torch.from_numpy(np.array(Image.open(pred_path).convert("L")))
                .to(device)
                .float()
            )
            rmse = torch.sqrt(F.mse_loss(pred_img, anno_img))
            rmses.append(rmse.item())
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    if not rmses:
        print("评估失败：没有找到可比较的文件。")
        return None

    metrics = {"Number of Samples": len(rmses), "Mean Per-Pixel RMSE": np.mean(rmses)}

    print("\n--- 评估结果 (Depth) ---")
    print(json.dumps(metrics, indent=4))
    print("-" * 28)
    return metrics


# =================================================================================
# 2. CANNY EVALUATION LOGIC
# =================================================================================
def evaluate_canny(anno_dir, pred_dir):
    """计算Canny边缘图的指标，并返回结果字典。"""
    print(f"\n--- 评估任务: Canny (F1, PSNR, SSIM) ---")
    # ... (函数主体代码保持不变) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")
    f1_metric = BinaryF1Score().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    results = defaultdict(list)
    anno_files = sorted(os.listdir(anno_dir))
    for filename in tqdm(anno_files, desc="评估 Canny"):
        anno_path = os.path.join(anno_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            continue
        anno_img = Image.open(anno_path).convert("L")
        pred_img = Image.open(pred_path).convert("L")
        if anno_img.size != pred_img.size:
            pred_img = pred_img.resize(anno_img.size)
        anno_tensor = (
            torch.from_numpy(np.array(anno_img)).unsqueeze(0).unsqueeze(0).to(device)
        )
        pred_tensor = (
            torch.from_numpy(np.array(pred_img)).unsqueeze(0).unsqueeze(0).to(device)
        )
        anno_norm, pred_norm = anno_tensor / 255.0, pred_tensor / 255.0
        psnr_score = psnr_metric(pred_norm, anno_norm)
        if torch.isinf(psnr_score):
            psnr_score = torch.tensor(100.0)
        ssim_score = ssim_metric(pred_norm, anno_norm)
        results["psnr"].append(psnr_score.item())
        results["ssim"].append(ssim_score.item())
        anno_binary, pred_binary = (anno_tensor > 100).int(), (pred_tensor > 100).int()
        f1_score = f1_metric(pred_binary.flatten(), anno_binary.flatten())
        results["f1"].append(f1_score.item())

    if not results["f1"]:
        print("评估失败：没有找到可比较的文件。")
        return None

    metrics = {
        "Number of Samples": len(results["f1"]),
        "Mean F1-Score": np.mean(results["f1"]),
        "Mean PSNR": np.mean(results["psnr"]),
        "Mean MS-SSIM": np.mean(results["ssim"]),
    }

    print(f"\n--- 评估结果 (Canny) ---")
    print(json.dumps(metrics, indent=4))
    print("-" * 28)
    return metrics


# =================================================================================
# 3. POSE EVALUATION LOGIC (Multi-Person OKS / AP)
# =================================================================================

# --- 新的、更强大的辅助函数 ---


def extract_all_persons(openpose_detector, image_path):
    """
    从单个图像文件中提取所有检测到的人物。
    返回一个包含每个人物信息的字典列表。
    """
    persons = []
    try:
        input_image = Image.open(image_path).convert("RGB")
        # detect_poses 返回一个 PoseResult 对象的列表
        pose_results_list = openpose_detector.detect_poses(
            np.array(input_image), include_hand=False, include_face=False
        )

        if not isinstance(pose_results_list, list):
            return []

        for pose_result in pose_results_list:
            if not isinstance(pose_result, PoseResult) or not pose_result.body:
                continue

            body = pose_result.body
            keypoints_coco_format = np.zeros(18 * 3)
            valid_kpts_x, valid_kpts_y = [], []

            for kpt in body.keypoints:
                if kpt is not None and kpt.score > 0.1:  # 增加一个置信度阈值
                    idx = body.keypoints.index(kpt)
                    if idx < 18:
                        keypoints_coco_format[idx * 3] = kpt.x
                        keypoints_coco_format[idx * 3 + 1] = kpt.y
                        keypoints_coco_format[idx * 3 + 2] = 2  # 标记为可见且已标注
                        valid_kpts_x.append(kpt.x)
                        valid_kpts_y.append(kpt.y)

            if not valid_kpts_x:
                continue

            x_min, x_max = min(valid_kpts_x), max(valid_kpts_x)
            y_min, y_max = min(valid_kpts_y), max(valid_kpts_y)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            persons.append(
                {
                    "keypoints": keypoints_coco_format.tolist(),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                }
            )

    except Exception as e:
        print(f"处理文件 {image_path} 时出错: {e}")

    return persons


def compute_oks(gt_person, pred_person):
    """计算单个真实人物和单个预测人物之间的 OKS 分数。"""
    sigmas = (
        np.array(
            [
                0.26,
                0.25,
                0.25,
                0.35,
                0.35,
                0.79,
                0.79,
                0.72,
                0.72,
                0.62,
                0.62,
                1.07,
                1.07,
                0.87,
                0.87,
                0.89,
                0.89,
                0.79,
            ]
        )
        / 10.0
    )
    gt_kpts = np.array(gt_person["keypoints"]).reshape(-1, 3)
    pred_kpts = np.array(pred_person["keypoints"]).reshape(-1, 3)

    # 使用 GT 的面积进行归一化
    if gt_person["area"] == 0:
        return 0.0

    d_sq = (gt_kpts[:, 0] - pred_kpts[:, 0]) ** 2 + (
        gt_kpts[:, 1] - pred_kpts[:, 1]
    ) ** 2
    visible = gt_kpts[:, 2] > 0

    e = d_sq / (gt_person["area"] * (sigmas**2) * 2 + np.spacing(1))
    oks = np.sum(np.exp(-e)[visible]) / np.sum(visible) if np.sum(visible) > 0 else 0.0
    return oks


# --- Worker 和主评估函数 (重构后支持多人) ---


def pose_worker(
    rank, world_size, all_files, anno_dir, pred_dir, oks_thresholds, result_queue
):
    """
    每个GPU进程上运行的Pose评估worker。
    对每张图片只检测一次，然后用所有阈值进行评估。
    """
    openpose_detector = OpenposeDetector.from_pretrained(
        "models/lllyasviel/ControlNet/annotator/ckpts"
    )

    files_for_this_rank = all_files[rank::world_size]

    # 为每个阈值初始化一套独立的统计数据
    local_stats = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "oks_scores": []}
        for threshold in oks_thresholds
    }

    pbar = tqdm(
        files_for_this_rank, desc=f"GPU {rank} [Multi-Pose OKS]", disable=(rank != 0)
    )
    for filename in pbar:
        anno_path = os.path.join(anno_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            continue

        # --- 昂贵的检测操作，只执行一次 ---
        gt_persons = extract_all_persons(openpose_detector, anno_path)
        dt_persons = extract_all_persons(openpose_detector, pred_path)
        # ------------------------------------

        if not gt_persons and not dt_persons:
            continue  # 如果两边都为空，对任何阈值都没有贡献，直接跳过

        if not gt_persons:
            for threshold in oks_thresholds:
                local_stats[threshold]["fp"] += len(dt_persons)
            continue

        if not dt_persons:
            for threshold in oks_thresholds:
                local_stats[threshold]["fn"] += len(gt_persons)
            continue

        # --- OKS 矩阵计算，也只执行一次 ---
        oks_matrix = np.zeros((len(gt_persons), len(dt_persons)))
        for i, gt in enumerate(gt_persons):
            for j, dt in enumerate(dt_persons):
                oks_matrix[i, j] = compute_oks(gt, dt)

        # --- 快速的内部循环，为每个阈值进行匹配 ---
        for threshold in oks_thresholds:
            # 使用 OKS 矩阵的副本来进行匹配，以防修改原矩阵
            temp_oks_matrix = oks_matrix.copy()

            gt_matched = [False] * len(gt_persons)
            dt_matched = [False] * len(dt_persons)

            # 贪心匹配
            for _ in range(min(len(gt_persons), len(dt_persons))):
                if np.max(temp_oks_matrix) < threshold:
                    break

                gt_idx, dt_idx = np.unravel_index(
                    np.argmax(temp_oks_matrix), temp_oks_matrix.shape
                )

                # 更新这个阈值对应的 TP 和 OKS 分数
                local_stats[threshold]["tp"] += 1
                local_stats[threshold]["oks_scores"].append(
                    temp_oks_matrix[gt_idx, dt_idx]
                )

                gt_matched[gt_idx] = True
                dt_matched[dt_idx] = True
                temp_oks_matrix[gt_idx, :] = 0
                temp_oks_matrix[:, dt_idx] = 0

            # 更新这个阈值对应的 FP 和 FN
            local_stats[threshold]["fn"] += len(gt_persons) - sum(gt_matched)
            local_stats[threshold]["fp"] += len(dt_persons) - sum(dt_matched)

    result_queue.put(local_stats)


def evaluate_pose(anno_dir, pred_dir, oks_thresholds):
    """主函数，负责启动并行评估并汇总多人评估结果。"""
    print(f"\n--- 评估任务: Multi-Person Pose (OKS, AP-like @{oks_thresholds}) ---")

    if not torch.cuda.is_available() or torch.cuda.device_count() == 1:
        # 单GPU/CPU模式
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        all_files = sorted(
            [
                f
                for f in os.listdir(anno_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        pose_worker(0, 1, all_files, anno_dir, pred_dir, oks_thresholds, q)
        world_size = 1
    else:
        # 多GPU模式
        world_size = torch.cuda.device_count()
        print(f"检测到 {world_size} 个GPU。开始并行评估 Pose...")
        all_files = sorted(
            [
                f
                for f in os.listdir(anno_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        # 传递 oks_thresholds 列表给 worker
        spawn_args = (world_size, all_files, anno_dir, pred_dir, oks_thresholds, q)
        mp.spawn(pose_worker, args=spawn_args, nprocs=world_size, join=True)
    # --- 汇总所有进程在所有阈值上的结果 ---
    final_stats = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "oks_scores": []}
        for threshold in oks_thresholds
    }

    for _ in range(world_size):
        proc_results = q.get()
        for threshold, stats in proc_results.items():
            final_stats[threshold]["tp"] += stats["tp"]
            final_stats[threshold]["fp"] += stats["fp"]
            final_stats[threshold]["fn"] += stats["fn"]
            final_stats[threshold]["oks_scores"].extend(stats["oks_scores"])

    # --- 为每个阈值计算并打印最终指标 ---
    all_metrics = {}
    for threshold, stats in final_stats.items():
        total_tp = stats["tp"]
        total_fp = stats["fp"]
        total_fn = stats["fn"]

        total_gt = total_tp + total_fn
        total_dt = total_tp + total_fp

        if total_gt == 0 or total_dt == 0:
            print(f"在 OKS@{threshold} 阈值下没有找到有效的姿态进行评估。")
            continue

        precision = total_tp / total_dt if total_dt > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0

        f1_score = 0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        # -------------------------

        metrics = {
            f"F1-Score@{threshold}": f1_score,
            f"Precision@{threshold}": precision,
            f"Recall@{threshold}": recall,
            "Total Ground Truth Poses": total_gt,
            "Total Detected Poses": total_dt,
            "True Positives": total_tp,
            "False Positives": total_fp,
            "False Negatives": total_fn,
        }

        if stats["oks_scores"]:
            metrics[f"Mean OKS (on {len(stats['oks_scores'])} TP matches)"] = np.mean(
                stats["oks_scores"]
            )
        else:
            metrics["Mean OKS"] = "N/A (No matches found)"

        print(f"\n--- 评估结果 (Multi-Person Pose @ OKS={threshold}) ---")
        print(json.dumps(metrics, indent=4))
        print("-" * 35)
        all_metrics[f"oks_at_{threshold}"] = metrics

    return all_metrics if all_metrics else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统一评估脚本，用于 Depth, Canny, 和 Pose，并将结果保存到JSON文件。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--anno_dir",
        type=str,
        required=True,
        help="包含 gt_*_cond 子目录的真实标注根目录。",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="包含 eval_*_cond 子目录的预测结果根目录。",
    )
    parser.add_argument(
        "--oks_thresholds",
        type=float,
        nargs="+",  # 允许接收一个或多个浮点数
        default=[0.5, 0.25, 0.1],  # 默认测试三个标准：严格，宽松，非常宽松
        help="用于判断匹配是否成功的 OKS 阈值列表。",
    )
    args = parser.parse_args()

    # 创建一个主字典来存储所有结果
    all_results = {}

    # --- 任务 1: Depth ---
    pred_dir_depth = os.path.join(args.pred_dir, "eval_depth_cond")
    anno_dir_depth = os.path.join(args.anno_dir, "gt_depth_cond")
    if os.path.isdir(pred_dir_depth) and os.path.isdir(anno_dir_depth):
        depth_results = evaluate_depth(anno_dir_depth, pred_dir_depth)
        if depth_results:
            all_results["depth_metrics"] = depth_results
    else:
        print(
            "\n跳过 Depth 评估：找不到对应的 'gt_depth_cond' 或 'eval_depth_cond' 目录。"
        )

    # --- 任务 2: Canny ---
    pred_dir_canny = os.path.join(args.pred_dir, "eval_canny_cond")
    anno_dir_canny = os.path.join(args.anno_dir, "gt_canny_cond")
    if os.path.isdir(pred_dir_canny) and os.path.isdir(anno_dir_canny):
        canny_results = evaluate_canny(anno_dir_canny, pred_dir_canny)
        if canny_results:
            all_results["canny_metrics"] = canny_results
    else:
        print(
            "\n跳过 Canny 评估：找不到对应的 'gt_canny_cond' 或 'eval_canny_cond' 目录。"
        )

    # --- 任务 3: Pose ---
    pred_dir_pose = os.path.join(args.pred_dir, "eval_pose")
    anno_dir_pose = os.path.join(args.anno_dir, "gt_pose")
    if os.path.isdir(pred_dir_pose) and os.path.isdir(anno_dir_pose):
        # 传递阈值列表给评估函数
        pose_results = evaluate_pose(anno_dir_pose, pred_dir_pose, args.oks_thresholds)
        if pose_results:
            # pose_results 现在是一个字典，其键是 'oks_at_0.5', 'oks_at_0.25' 等
            all_results["pose_metrics"] = pose_results
    else:
        print("\n跳过 Pose 评估...")
    # --- 将所有结果保存到文件 ---
    if all_results:
        save_path = os.path.join(args.pred_dir, "res.json")
        print(f"\n{'='*20}\n所有评估完成，结果将保存至: {save_path}\n{'='*20}")

        # 确保输出目录存在
        os.makedirs(args.pred_dir, exist_ok=True)

        # 写入JSON文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        print("结果文件保存成功。")
    else:
        print("\n没有生成任何有效的评估结果，不创建 res.json 文件。")
