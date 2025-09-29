import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
import gc
import sys

from eval.grounded_sam.florence2.modeling_florence2 import Florence2ForConditionalGeneration
from eval.grounded_sam.florence2.processing_florence2 import Florence2Processor
from eval.grounded_sam.sam2.build_sam import build_sam2
from eval.grounded_sam.sam2.sam2_image_predictor import SAM2ImagePredictor


class FlorenceSAM:

    # official usage: https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
    TASK_PROMPT = {
        "original": "<GIVEN>",
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "object_detection": "<OD>",
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
        "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
        "region_to_category": "<REGION_TO_CATEGORY>",
        "region_to_description": "<REGION_TO_DESCRIPTION>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>",
    }


    def __init__(self, device):
        """
        Init Florence-2 and SAM 2 Model
        """
        print(f"[{self}] init on device {device}")
        self.device = torch.device(device)

        # with torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
        # self.torch_dtype = torch.float32
        # self.torch_dtype = torch.float16
        self.torch_dtype = torch.bfloat16

        try:
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # self.torch_dtype = torch.bfloat16
            # else:
                # self.torch_dtype = torch.float16
        except:
            self.torch_dtype = torch.bfloat16
            
        FLORENCE2_MODEL_ID = os.getenv('FLORENCE2_MODEL_PATH', "models/microsoft/Florence-2-large")
        SAM2_CHECKPOINT = os.getenv('SAM2_MODEL_PATH', "models/facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt")
        SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.florence2_model = Florence2ForConditionalGeneration.from_pretrained(
            FLORENCE2_MODEL_ID, 
            torch_dtype=self.torch_dtype,
        ).eval().to(self.device)
        self.florence2_processor = Florence2Processor.from_pretrained(
            FLORENCE2_MODEL_ID, 
        )
        self.sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def __str__(self):
        return "FlorenceSAM"


    @torch.no_grad()
    def run_florence2(self, task_prompt, text_input, image):
        model = self.florence2_model
        processor = self.florence2_processor
        device = self.device
        assert model is not None, "You should pass the init florence-2 model here"
        assert processor is not None, "You should set florence-2 processor here"

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            if text_input is None:
                prompt = task_prompt
            else:
                prompt = task_prompt + text_input
            
            inputs = processor(
                text=prompt, images=image, 
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            ).to(device, self.torch_dtype)
            # inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, self.torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                # max_new_tokens=1024,
                max_new_tokens=768,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            return parsed_answer



    def caption(self, image, caption_task_prompt='<CAPTION>'):
        assert caption_task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
        caption_results = self.run_florence2(caption_task_prompt, None, image)
        text_input = caption_results[caption_task_prompt]
        caption = text_input
        return caption


    def segmentation(self, image, input_boxes, seg_model="sam"):
        if seg_model == "sam":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                sam2_predictor = self.sam2_predictor
                sam2_predictor.set_image(np.array(image))
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                if scores.ndim == 2:
                    scores = scores.squeeze(1)
        else:
            raise NotImplementedError()

        return masks, scores

    def post_process_results(self, image, caption, labels, detections, output_dir=None):
        result_dict = {
            "caption": caption,
            "instance_images": [],
            "instance_labels": [],
            "instance_bboxes": [],
            "instance_mask_scores": [],
        }
        
        if detections is None:
            return detections, result_dict

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=cv_image.copy(), detections=detections)
        
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        if output_dir is not None: 
            cv2.imwrite(os.path.join(output_dir, "detections.jpg"), annotated_frame)
        
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        if output_dir is not None: 
            cv2.imwrite(os.path.join(output_dir, "masks.jpg"), annotated_frame)

        for detection in detections:
            xyxy, mask, confidence, class_id, tracker_id, data = detection

            label = labels[class_id]
            cropped_img = sv.crop_image(image=cv_image, xyxy=xyxy)
            if output_dir is not None: 
                cv2.imwrite(os.path.join(output_dir, f"cropped_image_{label}.jpg"), cropped_img)

            if mask is None:
                result_dict["instance_mask_scores"].append(0)
                result_dict["instance_images"].append(cropped_img)
            else:
                mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)
                masked_img = np.where(mask, cv_image, 255)
                cropped_masked_img = sv.crop_image(image=masked_img, xyxy=xyxy)
                result_dict["instance_mask_scores"].append(confidence.item())
                result_dict["instance_images"].append(cropped_masked_img)
                
            result_dict["instance_labels"].append(label)
            result_dict["instance_bboxes"].append(xyxy)
            if output_dir is not None: 
                cv2.imwrite(os.path.join(output_dir, f"masked_image_{label}.jpg"), cropped_masked_img)

        torch.cuda.empty_cache()
        gc.collect()  
        return detections, result_dict

    def caption_phrase_grounding_and_segmentation(
        self,
        image,
        seg_model="sam",
        caption_task_prompt='<CAPTION>',
        original_caption=None,
        output_dir=None
    ):
        
        assert caption_task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<GIVEN>", "<OPEN_VOCABULARY_DETECTION>"]
        assert seg_model in ["sam", "florence2"]
        
        # image caption
        if caption_task_prompt in ["<GIVEN>", "<OPEN_VOCABULARY_DETECTION>"]:
            assert original_caption is not None
            caption = original_caption
        else:
            caption_results = self.run_florence2(caption_task_prompt, None, image)
            text_input = caption_results[caption_task_prompt]
            caption = text_input
        
        # phrase grounding
        grounding_results = self.run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', caption, image)['<CAPTION_TO_PHRASE_GROUNDING>']
        input_boxes = np.array(grounding_results["bboxes"])
        class_names = grounding_results["labels"]
        class_ids = np.array(list(range(len(class_names))))
        
        # segmentation
        masks, scores = self.segmentation(image, input_boxes, seg_model)
        
        labels = [f"{class_name}" for class_name in class_names]
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=scores,
        )

        return self.post_process_results(image, caption, labels, detections, output_dir)

    def od_grounding_and_segmentation(
        self,
        image,
        text_input,
        seg_model="sam",
        output_dir=None
    ):
        assert seg_model in ["sam", "florence2"]
        
        # od grounding
        grounding_results = self.run_florence2('<OPEN_VOCABULARY_DETECTION>', text_input, image)['<OPEN_VOCABULARY_DETECTION>']
        if len(grounding_results["bboxes"]) == 0:
            detections = None
            labels = []
        else:
            input_boxes = np.array(grounding_results["bboxes"])
            class_names = grounding_results["bboxes_labels"]
            class_ids = np.array(list(range(len(class_names))))

            # segmentation
            masks, scores = self.segmentation(image, input_boxes, seg_model)
            
            labels = [f"{class_name}" for class_name in class_names]
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=class_ids,
                confidence=scores,
            )

        return self.post_process_results(image, text_input, labels, detections, output_dir)
    
    def od_grounding(
        self,
        image,
        text_input,
        output_dir=None
    ):
        
        # od grounding
        grounding_results = self.run_florence2('<OPEN_VOCABULARY_DETECTION>', text_input, image)['<OPEN_VOCABULARY_DETECTION>']
        if len(grounding_results["bboxes"]) == 0:
            detections = None
            labels = []
        else:
            input_boxes = np.array(grounding_results["bboxes"])
            class_names = grounding_results["bboxes_labels"]
            class_ids = np.array(list(range(len(class_names))))

            labels = [f"{class_name}" for class_name in class_names]
            detections = sv.Detections(
                xyxy=input_boxes,
                class_id=class_ids,
            )

        return self.post_process_results(image, text_input, labels, detections, output_dir)

    def phrase_grounding_and_segmentation(
        self,
        image,
        text_input,
        seg_model="sam",
        output_dir=None
    ):
        assert seg_model in ["sam", "florence2"]

        # phrase grounding
        grounding_results = self.run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', text_input, image)['<CAPTION_TO_PHRASE_GROUNDING>']
        input_boxes = np.array(grounding_results["bboxes"])
        class_names = grounding_results["labels"]
        # print(f"[phrase_grounding_and_segmentation] input_label={text_input}, output_label={class_names}")
        class_ids = np.array(list(range(len(class_names))))

        # segmentation
        masks, scores = self.segmentation(image, input_boxes, seg_model)
        
        labels = [f"{class_name}" for class_name in class_names]
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=scores,
        )

        return self.post_process_results(image, text_input, labels, detections, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded SAM 2 Florence-2 Demos", add_help=True)
    parser.add_argument("--image_path", type=str, default="./notebooks/images/cars.jpg", required=True, help="path to image file")
    parser.add_argument("--caption_type", type=str, default="caption", required=False, help="granularity of caption")
    args = parser.parse_args()



    # IMAGE_PATH = args.image_path
    PIPELINE = "caption_to_phrase_grounding"
    CAPTION_TYPE = args.caption_type
    assert CAPTION_TYPE in ["caption", "detailed_caption", "more_detailed_caption", "original"]
    
    print(f"Running pipeline: {PIPELINE} now.")

    pipeline = FlorenceSAM("cuda:0")

    from glob import glob
    from tqdm import tqdm
    for image_path in tqdm(glob("/mnt/bn/lq-prompt-alignment/personal/chenbowen/code/IPVerse/prompt_alignment/Grounded-SAM-2/notebooks/images/*") * 3):
    # for image_path in tqdm(glob("/mnt/bn/lq-prompt-alignment/personal/chenbowen/code/IPVerse/prompt_alignment/Grounded-SAM-2/outputs/gcg_pipeline/00001.tar_debug/*.png")):
        print(pipeline.TASK_PROMPT, CAPTION_TYPE)
        image = Image.open(image_path).convert("RGB")
        pipeline.caption_phrase_grounding_and_segmentation(
            image=image,
            seg_model="sam",
            caption_task_prompt=pipeline.TASK_PROMPT[CAPTION_TYPE],
            output_dir=f"./outputs/{os.path.basename(image_path)}"
        )