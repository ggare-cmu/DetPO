'''
Run cmd: python detpo/run_vqa_rescore.py --model_name Qwen3-VL-30B-A3B-Instruct --vqa_rescore --data_instr_type ipt --output_dir results/rf100vl_IPT/Qwen3-VL-30B-A3B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS --vqa_batch_size 1 --dataset_path wb-prova
'''

import os

import json

from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import time

import numpy as np
import argparse

import detpo.utils as utils
from detpo.utils import TOKEN_STATS


def rescore_dataset(args, model, processor, dataset_path, data_instr_path,
                    run_name="", output_dir="results", siglip_pipe=None):

    test_dir = os.path.join(dataset_path, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.isfile(ann_path):
        print(f"No test annotations found in {test_dir}, skipping.")
        return None

    dataset_instructions_json = {}
    if os.path.isfile(data_instr_path):
        with open(data_instr_path, "r", encoding="utf-8") as f:
            dataset_instructions_json = json.load(f)
    print(f"\n\n\nLoaded dataset instructions for {dataset_path} from {data_instr_path}: \n{dataset_instructions_json}\n\n\n\n")

    coco_gt = COCO(ann_path)

    dataset_name = os.path.basename(dataset_path)
    predictions_dir = os.path.join(output_dir, "predictions", run_name)
    viz_dir = os.path.join(output_dir, "visuals", run_name)
    eval_dir = os.path.join(output_dir, "evaluations", run_name)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    ds_cat_ids = coco_gt.getCatIds()
    ds_cat_names = [coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids]
    cat_name2id_dict = {coco_gt.cats[cat_id]["name"]: cat_id for cat_id in ds_cat_ids}
    cat_dict = {cat_id: coco_gt.cats[cat_id]["name"] for cat_id in ds_cat_ids}

    print(f"Categories in {dataset_name}: {cat_dict}")

    if args.vqa_rescore:
        eval_types = ["vqa"]
    elif args.siglip_rescore:
        eval_types = ["siglip"]
    else:
        raise ValueError("No rescoring method specified. Use --vqa_rescore or --siglip_rescore.")

    prediction_cache_paths = {
        eval_type: os.path.join(predictions_dir, f"predictions_{dataset_name}_{eval_type}.json")
        for eval_type in eval_types
    }

    # *** Load raw predictions from previous inference run ***
    raw_predictions_path = os.path.join(output_dir, "live_results", "rank", f"{dataset_name}_live_results.json")
    if not os.path.isfile(raw_predictions_path):
        raise FileNotFoundError(
            f"Raw predictions file not found: {raw_predictions_path}. "
            "Please ensure that the inference step has been completed."
        )
    with open(raw_predictions_path, "r", encoding="utf-8") as f:
        raw_predictions = json.load(f)

    # --- Resume Logic ---
    detections_all_by_type = {eval_type: [] for eval_type in eval_types}
    processed_image_ids = set()

    if any(os.path.isfile(p) for p in prediction_cache_paths.values()):
        print(f"Attempting to resume from cached predictions for {dataset_path}")
        for eval_type, path in prediction_cache_paths.items():
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded_detections = json.load(f)
                        detections_all_by_type[eval_type] = loaded_detections
                        for det in loaded_detections:
                            processed_image_ids.add(det['image_id'])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load or parse {path}. Starting from scratch. Error: {e}")
                    detections_all_by_type[eval_type] = []
                    processed_image_ids = set()
                    break
        if processed_image_ids:
            print(f"Resuming. Found {len(processed_image_ids)} already processed images.")

    if all(os.path.isfile(p) for p in prediction_cache_paths.values()) and \
            len(processed_image_ids) == len(raw_predictions):
        print(f"All predictions already exist for {dataset_path}, skipping rescoring.")
    else:
        total_count = 0

        for prediction in tqdm(raw_predictions, desc=f"Rescoring {os.path.basename(dataset_path)}"):
            if prediction["img_id"] in processed_image_ids:
                print(f"Skipping already processed image_id: {prediction['img_id']}")
                total_count += 1
                continue

            parsed_detections_ranking = prediction["parsed_detections_ranking"]
            if not parsed_detections_ranking:
                print(f"Warning: No parsed detections for image_id {prediction['img_id']}. Skipping.")
                continue

            image_path = prediction["image_path"]

            all_detections = utils.run_rescorer(
                args, model, processor, image_path,
                dataset_instructions_json, parsed_detections_ranking,
                siglip_pipe=siglip_pipe,
            )

            for eval_type, detections in all_detections.items():
                for det in detections:
                    detections_all_by_type[eval_type].append({
                        "image_id": prediction["img_id"],
                        "category_id": cat_name2id_dict.get(det["category_name"], -1),
                        "bbox": det["bbox"],
                        "score": det["score"],
                        "image_path": image_path,
                        "category_name": det["category_name"],
                        "model_score": det.get("model_score"),
                        "vqa_score": det.get("vqa_score"),
                        "ranking_score": det.get("ranking_score"),
                        "siglip_score": det.get("siglip_score"),
                        "bbox_model_xyxy": det.get("bbox_model_xyxy"),
                    })

            total_count += 1

            if total_count % 5 == 0:
                print(f"\nSaving intermediate results at image {total_count}/{len(raw_predictions)}...")
                for eval_type, detections in detections_all_by_type.items():
                    with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
                        json.dump(detections, f)

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

        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        all_stats[eval_type] = coco_eval.stats

    eval_results_path = os.path.join(eval_dir, f"evaluation_{dataset_name}.json")
    serializable_stats = {
        eval_type: stats.tolist() if hasattr(stats, "tolist") else stats
        for eval_type, stats in all_stats.items()
    }
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)
    print(f"Saved evaluation results to {eval_results_path}")

    return all_stats


def run_single_dataset_evaluation(args, model=None, processor=None):
    """
    Runs VQA/SigLip rescoring for a single dataset using pre-computed
    rank-scored predictions from a prior inference run.
    """
    root_dir = args.root_path
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}")
        return

    dataset_path = os.path.join(root_dir, args.dataset_path)
    if not dataset_path or not os.path.isdir(dataset_path):
        print(f"Error: Invalid or missing --dataset_path: {dataset_path}")
        return

    output_dir = os.path.join(args.output_dir, "final_instruction_eval")

    if args.vqa_rescore:
        run_name = "vqa"
        if args.vqa_nocontext:
            run_name += "_nocontext"
    elif args.siglip_rescore:
        run_name = "siglip"
    else:
        raise ValueError("Specify --vqa_rescore or --siglip_rescore.")

    if args.data_instr_type == "ipt":
        data_instr_path = os.path.join(
            args.output_dir, "iterative_prompt_refinement",
            f"all_refined_class_instructions_{os.path.basename(dataset_path)}.json"
        )
    elif args.data_instr_type == "default":
        data_instr_path = os.path.join(
            "./data_instr/default",
            f"README.dataset_{os.path.basename(dataset_path)}.json"
        )
        run_name += "_defaultInstr"
    else:
        raise ValueError(f"Invalid --data_instr_type: {args.data_instr_type!r}. Must be 'ipt' or 'default'.")

    utils.set_seed(args.seed)

    siglip_pipe = None
    print(f"Using model: {args.model_name}")
    if args.vqa_rescore:
        if model is None or processor is None:
            model, processor = utils.load_qwen_model(args.model_name, server_url=args.server_url)
        else:
            print("Using provided model and processor.")
    if args.siglip_rescore:
        siglip_pipe = utils.load_siglip_pipeline()
        print("Loaded SigLip pipeline for confidence scoring.")

    print("=" * 60)
    print(f"Rescoring dataset: {dataset_path}")

    ds_stats = rescore_dataset(
        args, model, processor, dataset_path, data_instr_path,
        run_name=run_name, output_dir=output_dir,
        siglip_pipe=siglip_pipe,
    )

    if ds_stats is not None:
        print("\n--- Summary of Results ---")
        if args.vqa_rescore:
            print(f"[vqa] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['vqa'][0]:.4f}")
            print(f"[vqa] AR@1          for {os.path.basename(dataset_path)}: {ds_stats['vqa'][6]:.4f}")
        if args.siglip_rescore:
            print(f"[siglip] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['siglip'][0]:.4f}")
            print(f"[siglip] AR@1          for {os.path.basename(dataset_path)}: {ds_stats['siglip'][6]:.4f}")
    else:
        print(f"Rescoring failed for {dataset_path}")

    # ── Final token report ───────────────────────────────────────────────────
    print(TOKEN_STATS.report())
    dataset_name = os.path.basename(dataset_path)
    token_stats_path = os.path.join(args.output_dir, f"{dataset_name}_token_stats.json")
    TOKEN_STATS.save(token_stats_path)
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument('--server_url', type=str, default="http://localhost:8000/v1")
    parser.add_argument("--root_path", type=str, default="./datasets/rf100-vl-fsod/")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/rf100vl-zeroshot/rf20_IPT_singleclass_vqaScore_withNMS")
    parser.add_argument("--data_instr_type", type=str, default="ipt",
                        help="'ipt' for iterative-prompt-refinement instructions, 'default' for README.dataset defaults")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--vqa_batch_size', type=int, default=8)
    parser.add_argument("--vqa_rescore", action="store_true")
    parser.add_argument("--vqa_nocontext", action="store_true")
    parser.add_argument("--siglip_rescore", action="store_true")

    args = parser.parse_args()
    run_single_dataset_evaluation(args)
