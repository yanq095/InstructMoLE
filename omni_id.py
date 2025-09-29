# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from datasets import load_from_disk
from collections import defaultdict
import csv
from glob import glob
# Assume the FaceID class and related utility functions are in the following path
# Please adjust according to your project structure
from eval.tools.face_id import FaceID
from eval.data_utils import pil2tensor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate face similarity for generated images.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/tiger/efficient_ai/UniCombine/output/moe/omnicontext",
        help="The root directory containing model outputs (e.g., dreamo, kontext)."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/OmniContext",
        help="Path to the cached OmniContext dataset."
    )
    return parser.parse_args()


def export():
    """Main function to find, parse, and pivot results into a CSV."""
    root_dir = "/mnt/hdfs/harunasg/ablation_study_moe/omnicontext/"
    print(f"Searching for result files in: {root_dir}")

    # Define the search pattern to find all result files
    search_pattern = os.path.join(root_dir, '*', 'face_similarity_results.json')
    json_files = glob(search_pattern)

    if not json_files:
        print(f"Error: No 'face_similarity_results.json' files found in the subdirectories of '{root_dir}'.")
        return

    print(f"Found {len(json_files)} result files to process.")

    # This will store all data in a nested dictionary format:
    # { 'model_name': { 'task_name': score, ... }, ... }
    all_results = defaultdict(dict)
    all_task_names = set()

    # --- Step 1: Read all JSON files and populate the data structures ---
    for json_path in tqdm(json_files, desc="Parsing JSON files"):
        try:
            model_name = os.path.basename(os.path.dirname(json_path))

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key, value in data.items():
                if key == 'overall_model_average':
                    all_results[model_name]['overall_average'] = value
                elif isinstance(value, dict) and 'task_average' in value:
                    task_name = key
                    all_task_names.add(task_name)
                    all_results[model_name][task_name] = value.get('task_average')

        except Exception as e:
            print(f"Warning: An error occurred while processing {json_path}: {e}. Skipping.")

    # --- Step 2: Prepare the header and write to CSV ---
    # Sort task names for consistent column order
    sorted_tasks = sorted(list(all_task_names))
    header = ['model_name'] + sorted_tasks + ['overall_average']
    output_csv = 'pivoted_results.csv'
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Sort model names for consistent row order
            sorted_models = sorted(all_results.keys())

            for model_name in tqdm(sorted_models, desc="Writing CSV rows"):
                model_data = all_results[model_name]
                
                # Build the row in the correct order based on the header
                row = [model_name]
                for task in sorted_tasks:
                    # Use .get() to handle cases where a model might be missing a task
                    row.append(model_data.get(task, 'N/A'))
                
                # Append the overall average at the end
                row.append(model_data.get('overall_average', 'N/A'))
                
                writer.writerow(row)

        print(f"\nSuccessfully created pivoted summary file at: {output_csv}")

    except IOError as e:
        print(f"\nError: Could not write to CSV file at {output_csv}. Reason: {e}")


    


def main():
    args = parse_args()
    print(f"Starting evaluation with arguments: {args}")

    # --- 1. Initialize models and load data ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_score_model = FaceID(device)
    print(f"FaceID model loaded on device: {device}")

    try:
        dataset = load_from_disk(args.dataset_path)
        print(f"Successfully loaded dataset from {args.dataset_path}")
    except Exception as e:
        print(f"Error loading dataset from {args.dataset_path}: {e}")
        return

    # To speed up lookup, convert the dataset to a dictionary with 'key' as the key
    dataset_dict = {item['key']: item for item in dataset}
    print(f"Converted dataset to a dictionary with {len(dataset_dict)} items for fast lookup.")

    # --- 2. Iterate through all models and task types ---
    # Find all model directories under output_dir
    model_dirs = [d for d in glob(os.path.join(args.output_dir, '*')) if os.path.isdir(d)]
    model_dirs =['/opt/tiger/efficient_ai/UniCombine/output/moe/omnicontext/ExpertRace_E8T4']
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        model_results = {}
        all_task_scores = []

        fullset_path = os.path.join(model_dir, 'fullset')

        if not os.path.exists(fullset_path):
            print(f"Skipping model '{model_name}': 'fullset' directory not found.")
            continue

        print(f"\nProcessing model: {model_name}")

        # Find all directories containing "character"
        task_type_dirs = [d for d in glob(os.path.join(fullset_path, '*character*')) if os.path.isdir(d)]

        for task_dir in task_type_dirs:
            task_type = os.path.basename(task_dir)
            print(f"  Processing task_type: {task_type}")

            generated_images = glob(os.path.join(task_dir, '*.png'))
            if not generated_images:
                print(f"    No .png images found in {task_dir}. Skipping.")
                continue

            task_scores = {}
            for gen_img_path in tqdm(generated_images, desc=f"    {task_type}"):
                # Extract the key from the filename (assuming the filename is key.png or similar)
                img_key = os.path.splitext(os.path.basename(gen_img_path))[0]

                if img_key not in dataset_dict:
                    # print(f"Warning: Key '{img_key}' from image file not found in dataset. Skipping.")
                    continue

                # --- 3. Get Ground Truth and generated images ---
                ground_truth_sample = dataset_dict[img_key]
                # ground_truth_images is a list of Image objects
                real_faces = ground_truth_sample.get('input_images', [])

                if not real_faces or not isinstance(real_faces, list) or len(real_faces) == 0:
                    continue

                # The first image in the 'input_images' column is usually the main reference face
                # If there are multiple, all are used as references
                real_faces = [img.convert("RGB") for img in real_faces]

                try:
                    # Read the generated image and handle possible image grids
                    gen_img = Image.open(gen_img_path).convert("RGB")
                    # gen_images = split_grid(gen_img_grid) # This line is commented out as split_grid is not defined
                except Exception as e:
                    print(f"Failed to open or process generated image {gen_img_path}: {e}")
                    continue

                # --- 4. Calculate similarity ---
                # Here we only process the single generated image. If your task requires evaluating multiple images from a grid, add a loop.
                
                with torch.no_grad():
                    # Detect faces in the generated image
                    gen_bboxes = face_score_model.detect(
                        (pil2tensor(gen_img).unsqueeze(0) * 255).to(torch.uint8)
                    )
                    gen_faces = [gen_img.crop(bbox) for bbox in gen_bboxes]

                    scores_for_sample = []
                    # Iterate through each Ground Truth face
                    for real_face in real_faces:
                        if len(gen_faces) > 0:
                            # Calculate the similarity between the current GT face and all detected faces, and take the maximum value
                            score = max([face_score_model(real_face, gen_face) for gen_face in gen_faces])
                        else:
                            # If no face is detected, the similarity is 0
                            score = 0.0
                        scores_for_sample.append(score)

                    # If there are multiple GT faces, take the average score as the final score for this sample
                    final_score = np.mean(scores_for_sample) if scores_for_sample else 0.0
                    task_scores[img_key] = final_score

            if task_scores:
                # Calculate the average for the current task
                task_average = np.mean(list(task_scores.values()))
                model_results[task_type] = {
                    "image_scores": task_scores,
                    "task_average": task_average
                }
                # Collect all scores to calculate the overall average later
                all_task_scores.extend(list(task_scores.values()))

        # Calculate the overall average for the model if any scores were recorded
        if all_task_scores:
            overall_average = np.mean(all_task_scores)
            model_results['overall_model_average'] = overall_average

        # --- 5. Save results to a JSON file in the model's directory ---
        if model_results:
            json_output_path = os.path.join(model_dir, "face_similarity_results.json")
            print(f"\nEvaluation for model '{model_name}' finished. Saving results to {json_output_path}")
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, indent=4, ensure_ascii=False)
                print("Results saved successfully.")
            except Exception as e:
                print(f"Error saving JSON file: {e}")
        else:
            print(f"\nNo results generated for model '{model_name}'. Nothing to save.")

if __name__ == "__main__":
    main()
    export()
