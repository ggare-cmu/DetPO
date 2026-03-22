'''
Run cmd: CUDA_VISIBLE_DEVICES=0,1 python ipt/run_detpo_optimization.py --model_name Qwen2.5-VL-7B-Instruct --ipt_mode --vqa_rescore --apply_nms --nms_threshold 0.5 --num_ipt_iterations 10 --output_dir results/rf100vl_IPT_tmp/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1 --dataset_path wb-prova
'''

import os

import json
import torch
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import time

import numpy as np
import argparse
import re
import random

import detpo.run_evaluation as evaluator

import detpo.utils as utils
# Token stats singleton — shared with ipt_utils via module import
from detpo.utils import TOKEN_STATS


# =============================================================================
# Seed state helpers
# =============================================================================

def get_seed_state():
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def set_seed_from_state(seed_state):
    random.setstate(seed_state['python_random_state'])
    np.random.set_state(seed_state['numpy_random_state'])
    torch.set_rng_state(seed_state['torch_random_state'])
    if torch.cuda.is_available() and seed_state['torch_cuda_random_state'] is not None:
        torch.cuda.set_rng_state_all(seed_state['torch_cuda_random_state'])


# =============================================================================
# Dataset evaluation (train split)
# =============================================================================

def evaluate_dataset(args, model, processor, dataset_path, run_name="", output_dir="results",
                     eval_class_name=None, eval_cat_id=None,
                     max_samples=None,
                     dataset_instructions_json=None, coco_override=None, siglip_pipe=None, dataset_type="train"):
    train_dir = os.path.join(dataset_path, dataset_type)
    ann_path = os.path.join(train_dir, "_annotations.coco.json")

    if coco_override is not None:
        coco_gt = coco_override
        print(f"Using coco_override for {dataset_path}.")
    else:
        coco_gt = COCO(ann_path)

    dataset_name = os.path.basename(dataset_path)
    predictions_dir = os.path.join(output_dir, "predictions", run_name)
    viz_dir = os.path.join(output_dir, "visuals", run_name)
    eval_dir = os.path.join(output_dir, "evaluations", run_name)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    eval_types = ["model", "ranking"]
    prediction_cache_paths = {
        eval_type: os.path.join(predictions_dir, f"predictions_{dataset_name}_{eval_type}.json")
        for eval_type in eval_types
    }

    all_predictions_exist = all(os.path.isfile(p) for p in prediction_cache_paths.values())

    if all_predictions_exist and not args.ipt_mode:
        print(f"Using cached predictions for {dataset_path}")
        detections_all_by_type = {}
        for eval_type, path in prediction_cache_paths.items():
            with open(path, "r", encoding="utf-8") as f:
                detections_all_by_type[eval_type] = json.load(f)
    else:
        detections_all_by_type = {eval_type: [] for eval_type in eval_types}
        total_count = 0

        images_to_process = coco_gt.dataset["images"]
        if max_samples is not None:
            num_to_process = min(max_samples, len(images_to_process))
            images_to_process = images_to_process[:num_to_process]

        ds_cat_ids = coco_gt.getCatIds()
        ds_cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids]
        cat_name2id_dict = {coco_gt.cats[cat_id]["name"]: cat_id for cat_id in ds_cat_ids}
        cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}

        print(f"Categories in {dataset_name}: {cat_dict}")

        for img_info in tqdm(images_to_process, desc=f"Processing images in {os.path.basename(dataset_path)}"):
            img_id = img_info["id"]
            img_filename = img_info["file_name"]
            image_path = os.path.join(train_dir, img_filename)

            if not os.path.isfile(image_path):
                print(f"Image file not found: {image_path}. Skipping.")
                continue

            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids)

            cat_ids_for_image = set(ann["category_id"] for ann in anns)
            print(f"Image {img_filename} has categories: {[coco_gt.cats[cat_id]['name'] for cat_id in cat_ids_for_image]}")

            raw_output, all_detections = utils.run_inference_on_single_image(
                args,
                model, processor,
                image_path=image_path,
                dataset_instructions_json=dataset_instructions_json,
                class_name_list=[eval_class_name],
                output_dir=output_dir,
                siglip_pipe=siglip_pipe,
            )

            for eval_type, detections in all_detections.items():
                for det in detections:
                    detections_all_by_type[eval_type].append({
                        "image_id": img_id,
                        "category_id": cat_name2id_dict.get(det["category_name"], -1),
                        "bbox": det["bbox"],
                        "score": det["score"]
                    })

            total_count += 1

            yield {
                "img_id": img_id,
                "image_path": image_path,
                "gt_bboxes": [ann["bbox"] for ann in anns],
                "pred_bboxes": [det["bbox"] for det in all_detections["ranking"]],
                "raw_output": raw_output,
                "parsed_detections_model": all_detections["model"],
                "parsed_detections_ranking": all_detections["ranking"],
                "all_detections": all_detections,
                "gt_anns": anns,
                "cat_dict": cat_dict,
            }

        for eval_type, detections in detections_all_by_type.items():
            with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
                json.dump(detections, f)

    # --- Evaluate ---
    all_stats = {}
    for eval_type, detections in detections_all_by_type.items():
        print(f"\n--- Evaluating: {eval_type} ---")
        if not detections:
            print(f"No detections for {eval_type} in {dataset_path}.")
            all_stats[eval_type] = [0.0] * 12
            continue

        if max_samples is not None:
            processed_img_ids = {img['id'] for img in images_to_process}
            coco_gt_subset = COCO()
            coco_gt_subset.dataset['info'] = coco_gt.dataset.get('info', {})
            coco_gt_subset.dataset['licenses'] = coco_gt.dataset.get('licenses', [])
            coco_gt_subset.dataset['images'] = [img for img in coco_gt.dataset['images'] if img['id'] in processed_img_ids]
            coco_gt_subset.dataset['annotations'] = [ann for ann in coco_gt.dataset['annotations'] if ann['image_id'] in processed_img_ids]
            coco_gt_subset.dataset['categories'] = coco_gt.dataset['categories']
            coco_gt_subset.createIndex()
            coco_dt = coco_gt_subset.loadRes(detections)
            coco_eval = COCOeval(coco_gt_subset, coco_dt, "bbox")
        else:
            coco_dt = coco_gt.loadRes(detections)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        if eval_cat_id is not None:
            print(f"Evaluating on single category ID: {eval_cat_id}")
            coco_eval.params.catIds = [eval_cat_id]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        all_stats[eval_type] = coco_eval.stats

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, f"evaluation_{dataset_name}.json")
    serializable_stats = {
        eval_type: stats.tolist() if hasattr(stats, "tolist") else stats
        for eval_type, stats in all_stats.items()
    }
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)
    print(f"Saved evaluation results to {eval_results_path}")

    return all_stats


# =============================================================================
# Class definition generation helpers
# =============================================================================

def generate_initial_class_definition(args, model, processor, class_name, initial_instructions, few_shot_examples):
    utils.set_seed(args.seed)

    if not few_shot_examples:
        return ""

    content = [
        {"type": "text", "text":
         f"""
            Analyze the following images and describe the subjects or objects highlighted in green bounding boxes.
            Identify and summarize the key visual characteristics that are consistently observed across these objects.
            Emphasize the distinctive features that clearly differentiate this object class from other elements in the scene.

            Your goal is to produce a concise, clear, detailed, and generalizable definition that enables accurate recognition of this object class in future images and makes it easily distinguishable from other objects.
            Do not mention bounding boxes, colors, or any annotation details in your response.
        """},
    ]
    for example in few_shot_examples:
        content.append({"type": "image", "image": example})

    messages = [{"role": "user", "content": content}]
    definition, _ = utils.model_generate(messages, model, processor, token_stage="class_def_initial")

    definition = definition.replace(f"The visual characteristics of the '{class_name}' class are:", "").strip()
    definition = definition.replace(f"Definition of '{class_name}':", "").strip()
    print(f"Generated initial definition for '{class_name}': {definition}")
    return definition


def generate_class_definition_withFP(args, model, processor, class_name, current_instructions, correct_image, FP_error_image):
    content = [
        {"type": "text", "text":
         f"""
            Analyze the image carefully and identify the key visual differences between the object shown in the green bounding box and the one shown in the red bounding box.

            Follow the following steps:
            Step-1. Describe the distinguishing visual characteristics that set apart the object in the green bounding box from the object in the red bounding box.
            Step-2. Based on these distinguishing traits, formulate a clear and descriptive class definition for the object in the green bounding box. This definition should focus on its unique visual and contextual features that help differentiate it from the object in the red bounding box.
            Step-3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:

            Current class definition of the '{class_name}' class:
            \n{current_instructions}\n

            Step-4. Synthesize both definitions to produce an improved, more precise descriptive class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify true instances of the '{class_name}' class while reducing false positives similar to the one seen in the red bounding box.

            Note: Do not mention bounding boxes, colors, or image annotations in your response. The updated class definition should be a textual description of the '{class_name}' class objects.

            Return the final updated class definition as descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.
        """},
        {"type": "image", "image": correct_image},
        {"type": "image", "image": FP_error_image}
    ]

    messages = [{"role": "user", "content": content}]
    definition, _ = utils.model_generate(messages, model, processor, token_stage="class_def_fp")
    print(f"Generated definition for '{class_name}': {definition}")
    return definition


def generate_class_definition_withFN(args, model, processor, class_name, current_instructions, correct_image, FN_error_image):
    content = [
        {"type": "text", "text":
         f"""
            Analyze the image carefully and identify the key visual similarities between the object shown in the green bounding box and the one shown in the blue bounding box.

            Follow the following steps:
            Step-1. Describe the similar visual characteristics that set apart the object in the green bounding box from the object in the blue bounding box.
            Step-2. Based on these similarity traits, formulate a clear and descriptive class definition for the object in the blue bounding box as well as the object in the green bounding box. This definition should focus on its unique visual and contextual features that help identify both instances of the object in the green and blue bounding boxes.
            Step-3. Compare your new class definition with the existing definition of the '{class_name}' class provided below:

            Current class definition of the '{class_name}' class:
            \n{current_instructions}\n

            Step-4. Synthesize both definitions to produce an improved, more precise descriptive class definition for the '{class_name}' class. The updated definition should make it easier to accurately identify all true instances of the '{class_name}' class similar to the one seen in the blue and green bounding boxes.

            Note: Do not mention bounding boxes, colors, or image annotations in your response. The updated class definition should be a textual description of the '{class_name}' class objects.

            Return the final updated class definition as descriptive text in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.
        """},
        {"type": "image", "image": correct_image},
        {"type": "image", "image": FN_error_image}
    ]

    messages = [{"role": "user", "content": content}]
    definition, _ = utils.model_generate(messages, model, processor, token_stage="class_def_fn")
    print(f"Generated definition for '{class_name}': {definition}")
    return definition


def generate_refined_class_definition(args, model, processor, class_name, best_instructions):
    utils.set_seed(args.seed)

    content = [
        {"type": "text", "text":
         f"""
            Refine the class definition for the '{class_name}' category.

            Objective:
            Produce a concise, precise, and generalizable definition that enables reliable recognition of '{class_name}' instances across diverse images. The definition should clearly distinguish this class from visually or functionally similar object categories.

            Current definition:
            {best_instructions}

            Guidelines:
            - Focus on intrinsic, stable characteristics such as structure, shape, components, function, and typical physical configuration.
            - Ensure the description is detailed enough for accurate visual identification, yet broadly applicable across variations.
            - Do NOT mention bounding boxes, colors, image annotations, or dataset-specific context.
            - Avoid referencing specific images or examples.
            - Output only a textual class definition.

            Return the result strictly in the following format: ```python\n{{'{class_name}': <updated definition>}}\n```.
        """},
    ]

    messages = [{"role": "user", "content": content}]
    definition, _ = utils.model_generate(messages, model, processor, token_stage="class_def_refine")
    print(f"Generated refined definition for '{class_name}': {definition}")
    return extract_class_definition(definition, class_name)


import ast as _ast

def extract_class_definition(response, class_name):
    """Extract JSON/Python dict blocks from a text response."""
    import ast as _ast

    results = []
    code_blocks = re.findall(r"```(?:python|json)?\s*(.*?)\s*```", response, re.DOTALL)

    for block in code_blocks:
        block = block.strip()
        parsed = None
        try:
            parsed = _ast.literal_eval(block)
        except Exception:
            try:
                parsed = json.loads(block)
            except Exception:
                parsed = block
        results.append(parsed)

    return str(results)


# =============================================================================
# IPT methods
# =============================================================================

def method_generate_initial_class_definition(args, model, processor, cat_id, class_name,
                                             dataset_result_dir, coco_gt, ds_cat_ids, train_dir,
                                             class_instructions_json):
    img_ids = coco_gt.getImgIds(catIds=[cat_id])
    if not img_ids or len(img_ids) == 0:
        raise ValueError(f"No images found for category '{class_name}' in the dataset.")

    selected_img_ids = sorted(img_ids)
    gt_examples_for_class = []

    for chosen_img_id in selected_img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[chosen_img_id], catIds=[cat_id])
        anns = coco_gt.loadAnns(ann_ids)
        img_info_list = coco_gt.loadImgs(chosen_img_id)
        if not img_info_list:
            continue
        img_info = img_info_list[0]
        image_path = os.path.join(train_dir, img_info["file_name"])
        if not os.path.isfile(image_path):
            continue
        gt_bboxes = [ann['bbox'] for ann in anns if ann['category_id'] == cat_id]
        if len(gt_bboxes) == 0:
            continue

        img = Image.open(image_path).convert("RGB")
        img_with_boxes = utils.draw_colored_bboxes_on_image(img, "green", gt_bboxes)
        img_viz_path = os.path.join(dataset_result_dir,
                                    f"few_shot_example_cls_{class_name}_initial_imId_{chosen_img_id}_file_{os.path.basename(img_info['file_name'])}.png")
        max_dimension = (1920, 1080)
        img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)
        img_with_boxes.save(img_viz_path)
        gt_examples_for_class.append(img_with_boxes)

    if len(gt_examples_for_class) == 0:
        raise ValueError(f"No GT examples found for category '{class_name}' in the dataset.")

    def getInstructionsForClass(class_name, dataset_instructions_json):
        if class_name in dataset_instructions_json:
            return dataset_instructions_json[class_name]
        matched_key = next(
            (key for key in dataset_instructions_json.keys()
             if key.lower() == class_name.lower() or key.replace(" ", "_").lower() == class_name.lower()),
            None
        )
        if matched_key:
            return dataset_instructions_json[matched_key]
        raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")

    initial_instructions = getInstructionsForClass(class_name, class_instructions_json)

    org_instructions_path = os.path.join(dataset_result_dir, f"{class_name}_original_definition.txt")
    with open(org_instructions_path, "w", encoding="utf-8") as f:
        f.write(initial_instructions)

    examples_to_use = gt_examples_for_class

    init_def_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition.txt")
    if os.path.exists(init_def_path):
        print(f"Found existing initial definition for '{class_name}'. Loading from: {init_def_path}")
        with open(init_def_path, "r", encoding="utf-8") as f:
            initial_instructions = f.read()
    else:
        print(f"Generating initial definition for '{class_name}' using only GT examples.")
        initial_instructions = generate_initial_class_definition(
            args, model, processor, class_name, initial_instructions, examples_to_use
        )

        init_def_path_gt_only = os.path.join(dataset_result_dir, f"{class_name}_initial_with_only_gt_definition.txt")
        with open(init_def_path_gt_only, "w", encoding="utf-8") as f:
            f.write(initial_instructions)

        for idx, other_cat_id in enumerate(ds_cat_ids):
            if other_cat_id == cat_id:
                continue
            other_class_name = coco_gt.cats[other_cat_id]["name"]
            other_img_ids = coco_gt.getImgIds(catIds=[other_cat_id])
            if not other_img_ids or len(other_img_ids) == 0:
                continue

            chosen_other_img_id = random.choice(other_img_ids)
            other_ann_ids = coco_gt.getAnnIds(imgIds=[chosen_other_img_id], catIds=[other_cat_id])
            other_anns = coco_gt.loadAnns(other_ann_ids)
            other_img_info_list = coco_gt.loadImgs(chosen_other_img_id)
            if not other_img_info_list:
                continue
            other_img_info = other_img_info_list[0]
            other_image_path = os.path.join(train_dir, other_img_info["file_name"])
            if not os.path.isfile(other_image_path):
                continue
            other_gt_bboxes = [ann['bbox'] for ann in other_anns if ann['category_id'] == other_cat_id]
            if len(other_gt_bboxes) == 0:
                continue

            other_img = Image.open(other_image_path).convert("RGB")
            other_img_with_boxes = utils.draw_colored_bboxes_on_image(other_img, "red", other_gt_bboxes)
            max_dimension = (1920, 1080)
            other_img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)

            fp_examples_for_class = other_img_with_boxes
            positive_examples_for_class = examples_to_use[idx] if len(examples_to_use) > idx else random.choice(examples_to_use)

            fp_generated_definition_analysis = generate_class_definition_withFP(
                args, model, processor, class_name, initial_instructions,
                positive_examples_for_class, fp_examples_for_class
            )
            fp_generated_definition = extract_class_definition(fp_generated_definition_analysis, class_name)
            if fp_generated_definition:
                initial_instructions = fp_generated_definition
                init_def_fp_path = os.path.join(dataset_result_dir,
                                                 f"{class_name}_initial_definition_with_FP_{other_class_name}.txt")
                with open(init_def_fp_path, "w", encoding="utf-8") as f:
                    f.write(initial_instructions)

        with open(init_def_path, "w", encoding="utf-8") as f:
            f.write(initial_instructions)

        return initial_instructions


def method_evaluate_current_instructions(args, model, iter, processor, class_name, cat_id, current_instructions,
                                         dataset_name, dataset_path, dataset_result_dir, coco_gt, siglip_pipe,
                                         i, instruction_refinements,
                                         best_instructions, best_mAP,
                                         prev_instructions, prev_mAP,
                                         num_samples=None,
                                         stats_type="ranking"):
    run_name = f"class_{class_name}_ipt_iter_{i}"

    current_instructions_json = {class_name: current_instructions}
    seed_state = get_seed_state()

    eval_generator = evaluate_dataset(
        args, model, processor, dataset_path,
        run_name=run_name,
        output_dir=dataset_result_dir,
        dataset_instructions_json=current_instructions_json,
        eval_class_name=class_name,
        eval_cat_id=cat_id,
        coco_override=coco_gt if num_samples is not None else None,
        siglip_pipe=siglip_pipe
    )

    set_seed_from_state(seed_state)

    all_results_for_iter = []
    for j, result in enumerate(eval_generator):
        all_results_for_iter.append(result)

    eval_dir = os.path.join(dataset_result_dir, "evaluations", run_name)
    eval_results_path = os.path.join(eval_dir, f"evaluation_{dataset_name}.json")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            all_stats_dict = json.load(f)

        ap50_95 = all_stats_dict.get(stats_type, [0.0] * 12)[0]
        ar1 = all_stats_dict.get(stats_type, [0.0] * 12)[6]

        print(f"Iteration {iter} - Class '{class_name}': mAP@.50-.95 = {ap50_95:.4f}, AR@1 = {ar1:.4f}")

        instruction_refinements[f"class_{class_name}_iter_{iter}"] = {
            "mAP_50_95": ap50_95,
            "AR_1": ar1,
            "instructions": current_instructions
        }

        if ap50_95 >= best_mAP:
            best_mAP = ap50_95
            best_instructions = current_instructions

        if ap50_95 < prev_mAP:
            print(f"Iteration {iter} - Class '{class_name}': mAP decreased ({prev_mAP:.4f} -> {ap50_95:.4f}). Reverting.")
            current_instructions = prev_instructions
        else:
            prev_instructions = current_instructions
            prev_mAP = ap50_95

        current_mAP = ap50_95

    # ── Per-iteration token snapshot ─────────────────────────────────────────
    TOKEN_STATS.snapshot(label=f"iter {iter} class {class_name}")
    # ─────────────────────────────────────────────────────────────────────────

    return (all_results_for_iter, best_instructions, best_mAP, current_mAP,
            current_instructions, prev_instructions, prev_mAP, instruction_refinements)


def method_identify_worst_performing_examples(all_results_for_iter, iter, cat_id, class_name, dataset_result_dir,
                                              prev_worst_examples_map, stats_type="ranking"):
    def get_other_cls_ious(other_cls_gt_bboxes, pred_box):
        if len(other_cls_gt_bboxes) == 0:
            return 0.0
        other_cls_iou = [utils.calculate_iou(gt_box, pred_box) for gt_box in other_cls_gt_bboxes]
        return max(other_cls_iou)

    image_pred_performance = []
    for result in all_results_for_iter:
        gt_bboxes = [ann['bbox'] for ann in result['gt_anns'] if ann['category_id'] == cat_id]
        other_cls_gt_bboxes = [ann['bbox'] for ann in result['gt_anns'] if ann['category_id'] != cat_id]
        pred_detections = result['all_detections'][stats_type]

        if len(pred_detections) == 0:
            if len(gt_bboxes) > 0:
                for gt_bbox in gt_bboxes:
                    image_pred_performance.append({
                        "img_id": result['img_id'],
                        "gt_iou": 0.0, "best_score": 0.0,
                        "other_cls_iou": 0.0, "fp_error": 0.0,
                        "image_path": result['image_path'],
                        "gt_bbox": gt_bbox, "pred_bbox": [], "det_score": 0.0, "det": None,
                    })
            continue

        for det in pred_detections:
            det_score = det['score']
            pred_box = det['bbox']

            if len(gt_bboxes) == 0:
                other_cls_iou = get_other_cls_ious(other_cls_gt_bboxes, pred_box)
                fp_error = det_score * max(0.2, other_cls_iou)
                image_pred_performance.append({
                    "img_id": result['img_id'],
                    "gt_iou": 0.0, "best_score": -1.0,
                    "other_cls_iou": other_cls_iou, "fp_error": fp_error,
                    "image_path": result['image_path'],
                    "gt_bbox": None, "pred_bbox": pred_box, "det_score": det_score, "det": det,
                })
                continue

            gt_iou_list = [utils.calculate_iou(gt_box, pred_box) for gt_box in gt_bboxes]
            gt_bbox = gt_bboxes[np.argmax(gt_iou_list)]
            gt_iou = max(gt_iou_list)
            best_score = det_score * gt_iou
            other_cls_iou = get_other_cls_ious(other_cls_gt_bboxes, pred_box)
            fp_error = 0.0 if gt_iou > 0.0 else det_score * max(0.2, other_cls_iou)

            image_pred_performance.append({
                "img_id": result['img_id'],
                "gt_iou": gt_iou, "best_score": best_score,
                "other_cls_iou": other_cls_iou, "fp_error": fp_error,
                "image_path": result['image_path'],
                "gt_bbox": gt_bbox, "pred_bbox": pred_box, "det_score": det_score, "det": det,
            })

    top_n = 5
    best_match_candidates = sorted(image_pred_performance, key=lambda x: x['best_score'], reverse=True)[:top_n]
    best_match_candidates = [c for c in best_match_candidates if c['best_score'] > 0.0]
    if len(best_match_candidates) > 1:
        filtered = [c for c in best_match_candidates
                    if c['img_id'] != prev_worst_examples_map.get('best_match', {}).get('img_id')]
        if filtered:
            best_match_candidates = filtered

    worst_fp_candidates = sorted(image_pred_performance, key=lambda x: x['fp_error'], reverse=True)[:top_n]
    worst_fp_candidates = [c for c in worst_fp_candidates if c['fp_error'] > 0.0]
    if len(worst_fp_candidates) > 1:
        filtered = [c for c in worst_fp_candidates
                    if c['img_id'] != prev_worst_examples_map.get('worst_fp', {}).get('img_id')]
        if filtered:
            worst_fp_candidates = filtered

    worst_fn_candidates_dict = {}
    for c in image_pred_performance:
        if c['best_score'] == -1.0:
            continue
        fn_error = 1.0 - c['best_score']
        c['fn_error'] = fn_error
        key = f"{c['img_id']}_{c['gt_bbox']}"
        if key not in worst_fn_candidates_dict or c['best_score'] > worst_fn_candidates_dict[key]['best_score']:
            worst_fn_candidates_dict[key] = c

    worst_fn_candidates = sorted(worst_fn_candidates_dict.values(), key=lambda x: x['fn_error'], reverse=True)[:top_n]
    worst_fn_candidates = [c for c in worst_fn_candidates if c['fn_error'] > 0.0]
    if len(worst_fn_candidates) > 1:
        filtered = [c for c in worst_fn_candidates
                    if c['img_id'] != prev_worst_examples_map.get('worst_fn', {}).get('img_id')]
        if filtered:
            worst_fn_candidates = filtered

    best_match_example = random.choice(best_match_candidates) if best_match_candidates else None
    worst_fp_example = random.choice(worst_fp_candidates) if worst_fp_candidates else None
    worst_fn_example = random.choice(worst_fn_candidates) if worst_fn_candidates else None

    worst_examples_map = {
        'best_match': best_match_example,
        'worst_fp': worst_fp_example,
        'worst_fn': worst_fn_example,
    }
    prev_worst_examples_map = worst_examples_map
    print(f"Worst examples selected for iteration {iter+1}, class '{class_name}': \n{worst_examples_map}")

    few_shot_examples = {}
    for idx, (ex_type, ex) in enumerate(worst_examples_map.items()):
        if ex is None:
            continue

        img = Image.open(ex['image_path']).convert("RGB")
        if ex_type == 'best_match':
            img_with_boxes = utils.draw_colored_bboxes_on_image(img, "green", [ex['gt_bbox']])
            caption = f"Best Match (score: {ex['best_score']:.2f})"
        elif ex_type == 'worst_fp':
            img_with_boxes = utils.draw_colored_bboxes_on_image(img, "red", [ex['pred_bbox']])
            caption = f"Worst FP (Error: {ex['fp_error']:.2f})"
        elif ex_type == 'worst_fn':
            img_with_boxes = utils.draw_colored_bboxes_on_image(img, "blue", [ex['gt_bbox']])
            caption = f"Worst FN (Error: {ex['fn_error']:.2f})"
        else:
            continue

        max_dimension = (1920, 1080)
        img_with_boxes.thumbnail(max_dimension, Image.LANCZOS)
        img_viz_path = os.path.join(dataset_result_dir,
                                    f"few_shot_example_cls_{class_name}_iter{iter}_{ex_type}_imId_{ex['img_id']}_caption_{caption}.png")
        img_with_boxes.save(img_viz_path)
        few_shot_examples[ex_type] = img_with_boxes

    return few_shot_examples, prev_worst_examples_map


def method_refine_prompt(args, model, processor, class_name, current_instructions, few_shot_examples,
                         dataset_result_dir, iter):
    analysis_path = os.path.join(dataset_result_dir, f"{class_name}_analysis_iter_{iter}.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis for class '{class_name}' at iteration {iter+1}:\n\n")
        f.write(f"Current Instructions:\n{current_instructions}\n\n")

    fn_generated_definition, fp_generated_definition = None, None

    if 'best_match' in few_shot_examples and 'worst_fn' in few_shot_examples:
        fn_generated_definition_analysis = generate_class_definition_withFN(
            args, model, processor, class_name, current_instructions,
            few_shot_examples['best_match'], few_shot_examples['worst_fn']
        )
        fn_generated_definition = extract_class_definition(fn_generated_definition_analysis, class_name)
        if fn_generated_definition:
            current_instructions = fn_generated_definition
            with open(analysis_path, "a", encoding="utf-8") as f:
                f.write(f"Generated Analysis for False-Negative based Class Definition:\n{fn_generated_definition_analysis}\n\n")

    if 'best_match' in few_shot_examples and 'worst_fp' in few_shot_examples:
        fp_generated_definition_analysis = generate_class_definition_withFP(
            args, model, processor, class_name, current_instructions,
            few_shot_examples['best_match'], few_shot_examples['worst_fp']
        )
        fp_generated_definition = extract_class_definition(fp_generated_definition_analysis, class_name)
        if fp_generated_definition:
            current_instructions = fp_generated_definition
            with open(analysis_path, "a", encoding="utf-8") as f:
                f.write(f"Generated Analysis for False-Positive based Class Definition:\n{fp_generated_definition_analysis}\n\n")
                f.write(f"Refined Instructions:\n{current_instructions}\n")

    return current_instructions


def method_eval_on_val(args, model, processor, class_name, cat_id, dataset_name, dataset_path,
                       dataset_result_dir, siglip_pipe, stats_type, val_coco_gt=None):
    org_instructions_path = os.path.join(dataset_result_dir, f"{class_name}_original_definition.txt")
    with open(org_instructions_path, "r", encoding="utf-8") as f:
        original_instructions = f.read()

    init_instructions_path = os.path.join(dataset_result_dir, f"{class_name}_initial_definition.txt")
    with open(init_instructions_path, "r", encoding="utf-8") as f:
        initial_instructions = f.read()

    final_refined_instructions_path = os.path.join(dataset_result_dir,
                                                    f"refined_instructions_{dataset_name}_cls_{class_name}.txt")
    with open(final_refined_instructions_path, "r", encoding="utf-8") as f:
        final_refined_instructions = f.read()

    best_instructions_path = os.path.join(dataset_result_dir, f"best_instructions_{dataset_name}_cls_{class_name}.txt")
    with open(best_instructions_path, "r", encoding="utf-8") as f:
        valSet_best_instructions = f.read()

    altered_best_instruction = generate_refined_class_definition(args, model, processor, class_name, valSet_best_instructions)

    valSet_best_mAP = -1.0
    valSet_instruction_eval_result = {}

    for instr, instr_name in zip(
        [original_instructions, initial_instructions, valSet_best_instructions, final_refined_instructions, altered_best_instruction],
        ["original_instructions", "initial_instructions", "best_instructions", "final_refined_instructions", "altered_best_instruction"]
    ):
        run_name = f"class_{class_name}_instr_name_{instr_name}"
        results_dir = f"{dataset_result_dir}/class_{class_name}_{instr_name}_valSet_eval"

        current_instructions_json = {class_name: instr}
        seed_state = get_seed_state()

        eval_generator = evaluate_dataset(
            args, model, processor, dataset_path,
            run_name=run_name,
            output_dir=results_dir,
            dataset_instructions_json=current_instructions_json,
            eval_class_name=class_name,
            eval_cat_id=cat_id,
            siglip_pipe=siglip_pipe,
            dataset_type="valid",
            coco_override=val_coco_gt if val_coco_gt is not None else None,
        )
        set_seed_from_state(seed_state)

        all_results_for_iter = []
        for j, result in enumerate(eval_generator):
            all_results_for_iter.append(result)

        eval_dir = os.path.join(results_dir, "evaluations", run_name)
        eval_results_path = os.path.join(eval_dir, f"evaluation_{dataset_name}.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                all_stats_dict = json.load(f)

            ap50_95 = all_stats_dict.get(stats_type, [0.0] * 12)[0]
            ar1 = all_stats_dict.get(stats_type, [0.0] * 12)[6]

            print(f"Instruction {instr_name} - Class '{class_name}': mAP@.50-.95 = {ap50_95:.4f}, AR@1 = {ar1:.4f}")
            valSet_instruction_eval_result[f"class_{class_name}_{instr_name}"] = {
                "mAP_50_95": ap50_95, "AR_1": ar1, "instructions": instr
            }
            if ap50_95 >= valSet_best_mAP:
                valSet_best_mAP = ap50_95
                valSet_best_instructions = instr

    # ── Token snapshot after val evaluation ──────────────────────────────────
    TOKEN_STATS.snapshot(label=f"val_eval class {class_name}")
    # ─────────────────────────────────────────────────────────────────────────

    return valSet_instruction_eval_result, valSet_best_instructions, valSet_best_mAP


# =============================================================================
# Dataset subsampling
# =============================================================================

import logging

def subsample_dataset(coco_gt, num_samples, ds_cat_ids, log_file="subsample_dataset.log"):
    logger = logging.getLogger("subsample_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    def log(msg=""):
        logger.info(msg)

    dataset_stats = {}
    for cat_id in ds_cat_ids:
        img_ids = coco_gt.getImgIds(catIds=[cat_id])
        for img_id in img_ids:
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
            count = len(ann_ids)
            if count == 0:
                continue
            if img_id not in dataset_stats:
                dataset_stats[img_id] = {}
            dataset_stats[img_id][cat_id] = count

    log(f"Dataset stats built: {len(dataset_stats)} candidate images across {len(ds_cat_ids)} classes.")

    def recompute_achieved(selected):
        counts = {cat_id: 0 for cat_id in ds_cat_ids}
        for img_id in selected:
            for cat_id, count in dataset_stats[img_id].items():
                if cat_id in counts:
                    counts[cat_id] += count
        return counts

    def is_admissible_given(img_id, current_achieved):
        for cat_id, count in dataset_stats[img_id].items():
            if cat_id in current_achieved and current_achieved[cat_id] + count > num_samples:
                return False
        return True

    def still_needed_given(img_id, current_achieved):
        return any(current_achieved.get(cat_id, 0) < num_samples for cat_id in dataset_stats[img_id])

    def total_bbox_count(img_id):
        return sum(dataset_stats[img_id].values())

    def total_deficit(achieved):
        return sum(max(0, num_samples - achieved[cat_id]) for cat_id in ds_cat_ids)

    achieved = {cat_id: 0 for cat_id in ds_cat_ids}
    selected_img_ids = set()
    all_img_ids = list(dataset_stats.keys())
    random.shuffle(all_img_ids)

    while not all(achieved[cat_id] >= num_samples for cat_id in ds_cat_ids):
        candidates = [
            img_id for img_id in all_img_ids
            if img_id not in selected_img_ids
            and is_admissible_given(img_id, achieved)
            and still_needed_given(img_id, achieved)
        ]
        if not candidates:
            log("Greedy phase exhausted — some classes may be under quota. Attempting swap phase.")
            break
        best_img = min(candidates, key=total_bbox_count)
        selected_img_ids.add(best_img)
        for cat_id, count in dataset_stats[best_img].items():
            if cat_id in achieved:
                achieved[cat_id] += count

    log(f"After greedy phase: {len(selected_img_ids)} images selected. Deficit: {total_deficit(achieved)}")

    under_quota_cats = [cat_id for cat_id in ds_cat_ids if achieved[cat_id] < num_samples]
    if under_quota_cats:
        improved = True
        swap_count = 0
        while improved:
            improved = False
            for cat_id in under_quota_cats:
                if achieved[cat_id] >= num_samples:
                    continue
                deficit_before = total_deficit(achieved)
                swap_in_candidates = [
                    img_id for img_id in all_img_ids
                    if img_id not in selected_img_ids and cat_id in dataset_stats[img_id]
                ]
                best_swap = None
                best_deficit_after = deficit_before
                for swap_in in swap_in_candidates:
                    for swap_out in list(selected_img_ids):
                        trial_selected = (selected_img_ids - {swap_out}) | {swap_in}
                        trial_achieved = recompute_achieved(trial_selected)
                        if any(trial_achieved[c] > num_samples for c in ds_cat_ids):
                            continue
                        deficit_after = total_deficit(trial_achieved)
                        if deficit_after < best_deficit_after:
                            best_deficit_after = deficit_after
                            best_swap = (swap_out, swap_in, trial_selected, trial_achieved)
                if best_swap is not None:
                    swap_out, swap_in, trial_selected, trial_achieved = best_swap
                    log(f"  Swap: removed IMG {swap_out}, added IMG {swap_in} | deficit {deficit_before} -> {best_deficit_after}")
                    selected_img_ids = trial_selected
                    achieved = trial_achieved
                    swap_count += 1
                    improved = True

    coco_gt_subset = COCO()
    coco_gt_subset.dataset['info'] = coco_gt.dataset.get('info', {})
    coco_gt_subset.dataset['licenses'] = coco_gt.dataset.get('licenses', [])
    coco_gt_subset.dataset['categories'] = coco_gt.dataset['categories']
    coco_gt_subset.dataset['images'] = [img for img in coco_gt.dataset['images'] if img['id'] in selected_img_ids]
    coco_gt_subset.dataset['annotations'] = [ann for ann in coco_gt.dataset['annotations'] if ann['image_id'] in selected_img_ids]
    coco_gt_subset.createIndex()
    return coco_gt_subset


# =============================================================================
# Main IPT loop
# =============================================================================

def iterative_prompt_refinement(args, model, processor, dataset_path, num_iterations=3,
                                num_samples=None, siglip_pipe=None):
    utils.set_seed(args.seed)

    readme_json_path = os.path.join("./data_instr/default", f"README.dataset_{os.path.basename(dataset_path)}.json")
    class_instructions_json = {}
    if os.path.isfile(readme_json_path):
        with open(readme_json_path, "r", encoding="utf-8") as f:
            class_instructions_json = json.load(f)

    train_dir = os.path.join(dataset_path, "train")
    ann_path = os.path.join(train_dir, "_annotations.coco.json")
    dataset_name = os.path.basename(dataset_path)
    coco_gt = COCO(ann_path)

    ds_cat_ids = coco_gt.getCatIds()
    cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}
    print(f"Categories in {dataset_name}: {cat_dict}")

    result_dir = os.path.join(args.output_dir, "iterative_prompt_refinement")
    os.makedirs(result_dir, exist_ok=True)

    dataset_result_dir = os.path.join(result_dir, dataset_name)
    os.makedirs(dataset_result_dir, exist_ok=True)

    val_coco_gt = None
    if num_samples is not None:
        utils.set_seed(args.seed)
        print(f"[Warning!] Limiting to {num_samples} samples per class for IPT.")
        coco_gt = subsample_dataset(coco_gt, num_samples, ds_cat_ids,
                                    log_file=os.path.join(dataset_result_dir, "subsample_train_dataset.log"))

        val_dir = os.path.join(dataset_path, "valid")
        val_ann_path = os.path.join(val_dir, "_annotations.coco.json")
        val_coco_gt_full = COCO(val_ann_path)
        val_ds_cat_ids = val_coco_gt_full.getCatIds()
        val_coco_gt = subsample_dataset(val_coco_gt_full, num_samples, val_ds_cat_ids,
                                        log_file=os.path.join(dataset_result_dir, "subsample_val_dataset.log"))

    utils.set_seed(args.seed)

    all_iterm_refined_instructions_path = os.path.join(result_dir, f"all_iterm_refined_class_instructions_{dataset_name}.json")
    refined_class_instructions_json = {}
    if os.path.exists(all_iterm_refined_instructions_path):
        print(f"Found existing refined instructions file. Loading to resume: {all_iterm_refined_instructions_path}")
        try:
            with open(all_iterm_refined_instructions_path, "r", encoding="utf-8") as f:
                refined_class_instructions_json = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing instructions file. Starting from scratch. Error: {e}")
            refined_class_instructions_json = {}

    for cat_id in ds_cat_ids:
        class_name = coco_gt.cats[cat_id]["name"]

        if class_name in refined_class_instructions_json:
            print(f"Skipping class '{class_name}' as its definition already exists.")
            continue

        print(f"\n\n\n=== Generating initial class definition for '{class_name}' [{ds_cat_ids.index(cat_id)+1}/{len(ds_cat_ids)}] ===\n\n\n")
        initial_instructions = method_generate_initial_class_definition(
            args, model, processor, cat_id, class_name,
            dataset_result_dir, coco_gt, ds_cat_ids, train_dir, class_instructions_json
        )

        current_instructions = initial_instructions
        prev_instructions = initial_instructions
        best_mAP = -1.0
        best_instructions = initial_instructions
        prev_mAP = -1.0
        prev_worst_examples_map = {}

        iteration_state_path = os.path.join(dataset_result_dir, f"ipt_state_{dataset_name}_{class_name}.json")
        start_iteration = 0
        instruction_refinements = {}

        if os.path.exists(iteration_state_path):
            try:
                with open(iteration_state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                start_iteration = state.get("last_completed_iteration", -1) + 1
                current_instructions = state.get("current_instructions", initial_instructions)
                best_instructions = state.get("best_instructions", initial_instructions)
                best_mAP = state.get("best_mAP", -1.0)
                prev_mAP = state.get("prev_mAP", -1.0)
                prev_instructions = state.get("prev_instructions", initial_instructions)
                instruction_refinements = state.get("instruction_refinements", {})
                print(f"Resuming from iteration {start_iteration} for class '{class_name}'.")
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Could not read iteration state file. Starting from 0. Error: {e}")
                start_iteration = 0

        if start_iteration >= num_iterations:
            valSet_best_instructions_path = os.path.join(dataset_result_dir,
                                                          f"valSet_best_instructions_{dataset_name}_cls_{class_name}.txt")
            if os.path.exists(valSet_best_instructions_path):
                continue
        else:
            iter_instructions = ""

            for iter in range(start_iteration, num_iterations + 1):
                print(f"\n\n\n--- Iteration {iter} for class '{class_name}' [{ds_cat_ids.index(cat_id)+1}/{len(ds_cat_ids)}] ---\n\n\n")
                stats_type = "ranking"

                (all_results_for_iter, best_instructions, best_mAP, current_mAP,
                 current_instructions, prev_instructions, prev_mAP, instruction_refinements) = \
                    method_evaluate_current_instructions(
                        args, model, iter, processor, class_name, cat_id, current_instructions,
                        dataset_name, dataset_path, f"{dataset_result_dir}/{class_name}_iter{iter}",
                        coco_gt, siglip_pipe,
                        iter, instruction_refinements,
                        best_instructions, best_mAP,
                        prev_instructions, prev_mAP,
                        num_samples=num_samples, stats_type=stats_type
                    )

                iteration_state = {
                    "last_completed_iteration": iter,
                    "current_instructions": current_instructions,
                    "best_instructions": best_instructions,
                    "best_mAP": best_mAP,
                    "prev_mAP": prev_mAP,
                    "current_mAP": current_mAP,
                    "prev_instructions": prev_instructions,
                    "iter_instructions": iter_instructions,
                    "instruction_refinements": instruction_refinements,
                }
                with open(iteration_state_path, "w", encoding="utf-8") as f:
                    json.dump(iteration_state, f, indent=2)

                if iter == num_iterations:
                    break

                few_shot_examples, prev_worst_examples_map = method_identify_worst_performing_examples(
                    all_results_for_iter, iter, cat_id, class_name, dataset_result_dir,
                    prev_worst_examples_map, stats_type=stats_type
                )
                current_instructions = method_refine_prompt(
                    args, model, processor, class_name, current_instructions,
                    few_shot_examples, dataset_result_dir, iter=iter
                )
                iter_instructions = current_instructions

        refined_instructions_path = os.path.join(dataset_result_dir,
                                                  f"refined_instructions_{dataset_name}_cls_{class_name}.txt")
        with open(refined_instructions_path, "w", encoding="utf-8") as f:
            f.write(current_instructions)

        best_instructions_path = os.path.join(dataset_result_dir,
                                               f"best_instructions_{dataset_name}_cls_{class_name}.txt")
        with open(best_instructions_path, "w", encoding="utf-8") as f:
            f.write(best_instructions)

        instruction_refinements["final_refined_instructions"] = current_instructions
        instruction_refinements["best_instructions"] = best_instructions

        instruction_refinements_log_path = os.path.join(dataset_result_dir,
                                                         f"instruction_refinements_log_{dataset_name}_cls_{class_name}.json")
        with open(instruction_refinements_log_path, "w", encoding="utf-8") as f:
            json.dump(instruction_refinements, f, indent=2)

        refined_class_instructions_json[class_name] = best_instructions

        with open(all_iterm_refined_instructions_path, "w", encoding="utf-8") as f:
            json.dump(refined_class_instructions_json, f, indent=2)

        valSet_instruction_eval_result, valSet_best_instructions, valSet_best_mAP = method_eval_on_val(
            args, model, processor, class_name, cat_id, dataset_name, dataset_path,
            dataset_result_dir, siglip_pipe, stats_type="ranking",
            val_coco_gt=val_coco_gt
        )

        valSet_eval_results_path = os.path.join(dataset_result_dir,
                                                 f"valSet_instruction_eval_results_{dataset_name}_cls_{class_name}.json")
        with open(valSet_eval_results_path, "w", encoding="utf-8") as f:
            json.dump(valSet_instruction_eval_result, f, indent=2)

        valSet_best_instructions_path = os.path.join(dataset_result_dir,
                                                      f"valSet_best_instructions_{dataset_name}_cls_{class_name}.txt")
        with open(valSet_best_instructions_path, "w", encoding="utf-8") as f:
            f.write(valSet_best_instructions)

        refined_class_instructions_json[class_name] = valSet_best_instructions

        with open(all_iterm_refined_instructions_path, "w", encoding="utf-8") as f:
            json.dump(refined_class_instructions_json, f, indent=2)

    all_refined_instructions_path = os.path.join(result_dir, f"all_refined_class_instructions_{dataset_name}.json")
    with open(all_refined_instructions_path, "w", encoding="utf-8") as f:
        json.dump(refined_class_instructions_json, f, indent=2)
    print(f"Saved all refined class instructions to {all_refined_instructions_path}")

    return refined_class_instructions_json


# =============================================================================
# Entry point
# =============================================================================

def run_single_dataset_evaluation(args):
    root_dir = args.root_path
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}")
        return

    dataset_path = os.path.join(root_dir, args.dataset_path)
    if not dataset_path or not os.path.isdir(dataset_path):
        print(f"Error: Invalid or missing --dataset_path: {dataset_path}")
        return

    utils.set_seed(args.seed)
    print(f"Using model: {args.model_name}")
    model, processor = utils.load_qwen_model(args.model_name, server_url=args.server_url)

    siglip_pipe = None
    if args.siglip_rescore:
        siglip_pipe = utils.load_siglip_pipeline()
        print("Loaded SigLip pipeline for confidence scoring.")

    print("=" * 60)
    print(f"Evaluating dataset: {dataset_path}")

    if os.path.exists(os.path.join(
        args.output_dir, "iterative_prompt_refinement",
        f"all_refined_class_instructions_{os.path.basename(dataset_path)}.json"
    )):
        print("Refined class instructions already exist. Skipping IPT.")
    else:
        iterative_prompt_refinement(
            args,
            model=model,
            processor=processor,
            dataset_path=dataset_path,
            num_iterations=args.num_ipt_iterations,
            siglip_pipe=siglip_pipe,
            num_samples=args.num_samples,
        )

    print("\n\n" + "*" * 60 + "\n")
    print("Starting the final evaluation with the new refined class definitions...")
    print("\n" + "*" * 60 + "\n\n")

    args.data_instr_path = os.path.join(
        args.output_dir, "iterative_prompt_refinement", "all_refined_class_instructions"
    )
    args.output_dir = os.path.join(args.output_dir, "final_instruction_eval")
    evaluator.run_single_dataset_evaluation(args, model=model, processor=processor)

    # ── Final token report for the entire IPT run ────────────────────────────
    dataset_name = os.path.basename(dataset_path)
    print(TOKEN_STATS.report())
    token_stats_path = os.path.join(args.output_dir, f"{dataset_name}_ipt_token_stats.json")
    TOKEN_STATS.save(token_stats_path)
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--root_path", type=str, default="./datasets/rf100-vl-fsod/")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/rf100vl_IPT/rf20_IPT_singleclass_vqaScore_withNMS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--server_url', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--vqa_batch_size', type=int, default=8)
    parser.add_argument("--rank_rescore", action="store_true", default=True)
    parser.add_argument("--siglip_rescore", action="store_true")
    parser.add_argument("--ipt_mode", action="store_true")
    parser.add_argument("--num_ipt_iterations", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()
    run_single_dataset_evaluation(args)