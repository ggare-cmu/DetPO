'''
Run cmd: CUDA_VISIBLE_DEVICES=0,1 python ipt/run_evaluation.py --model_name Qwen2.5-VL-7B-Instruct --vqa_rescore --apply_nms --nms_threshold 0.5 --data_instr_path results/rf100vl_IPT/Qwen2.5-VL-7B-Instruct/rf20_IPT_singleclass_vqaScore_withNMS/iterative_prompt_refinement/all_refined_class_instructions --output_dir results/rf100vl_IPT_eval_tmp/rf20_IPT_singleclass_vqaScore_withNMS_tmp --vqa_batch_size 1 --dataset_path wb-prova
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
# Token stats singleton — shared with ipt_utils via module import
from detpo.utils import TOKEN_STATS


def evaluate_dataset(args, model, processor, dataset_path,
                     run_name="", output_dir="results", max_samples=None, siglip_pipe=None):
    test_dir = os.path.join(dataset_path, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    readme_json_path = os.path.join(f"{args.data_instr_path}_{os.path.basename(dataset_path)}.json")
    if not os.path.isfile(ann_path):
        print(f"No test annotations found in {test_dir}, skipping.")
        return None

    dataset_instructions_json = {}
    if os.path.isfile(readme_json_path):
        with open(readme_json_path, "r", encoding="utf-8") as f:
            dataset_instructions_json = json.load(f)
    print(f"\n\n\nLoaded dataset instructions for {dataset_path} from {readme_json_path}: \n{dataset_instructions_json}\n\n\n\n")

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
                    print(f"Warning: Could not load or parse {path}. Starting this eval type from scratch. Error: {e}")
                    detections_all_by_type[eval_type] = []
                    processed_image_ids = set()
                    break

        if processed_image_ids:
            print(f"Resuming. Found {len(processed_image_ids)} already processed images.")

    if all(os.path.isfile(p) for p in prediction_cache_paths.values()) and \
            len(processed_image_ids) == len(coco_gt.dataset["images"]):
        print(f"All predictions already exist for {dataset_path}, skipping inference and using cached predictions.")
    else:
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
            if img_info['id'] in processed_image_ids:
                print(f"Skipping already processed image_id: {img_info['id']}")
                total_count += 1
                continue

            img_id = img_info["id"]
            img_filename = img_info["file_name"]
            image_path = os.path.join(test_dir, img_filename)

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
                class_name_list=ds_cat_names,
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

            # Incremental save
            if total_count % 5 == 0:
                print(f"\nSaving intermediate results at image {total_count + 1}/{len(images_to_process)}...")
                for eval_type, detections in detections_all_by_type.items():
                    with open(prediction_cache_paths[eval_type], "w", encoding="utf-8") as f:
                        json.dump(detections, f)

            yield {
                "img_id": img_id,
                "image_path": image_path,
                "gt_bboxes": [ann["bbox"] for ann in anns],
                "pred_bboxes": [det["bbox"] for det in all_detections["ranking"]],
                "raw_output": raw_output,
                "parsed_detections_model": all_detections["model"],
                "parsed_detections_ranking": all_detections["ranking"],
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
    Runs evaluation for a single dataset.
    Prints and saves token usage statistics at the end.
    """
    root_dir = args.root_path
    print(f"Root directory: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"Root directory not found: {root_dir}")
        return

    dataset_path = os.path.join(root_dir, args.dataset_path)
    print(f"Dataset path: {dataset_path}")

    if not dataset_path or not os.path.isdir(dataset_path):
        print(f"Error: Invalid or missing --dataset_path: {dataset_path}")
        return

    run_name = "default"

    utils.set_seed(args.seed)

    print(f"Using model: {args.model_name}")
    if model is None or processor is None:
        model, processor = utils.load_qwen_model(args.model_name, server_url=args.server_url)
    else:
        print("Using provided model and processor.")

    siglip_pipe = None
    if args.siglip_rescore:
        siglip_pipe = utils.load_siglip_pipeline()
        print("Loaded Siglip pipeline for confidence scoring.")

    print("=" * 60)
    print(f"Evaluating dataset: {dataset_path}")

    eval_generator = evaluate_dataset(
        args, model, processor, dataset_path,
        run_name=run_name, output_dir=args.output_dir,
        siglip_pipe=siglip_pipe,
    )

    live_results = []
    ds_stats = None

    def _save_live_results_snapshot(live_results, suffix=""):
        try:
            dataset_basename = os.path.basename(dataset_path.rstrip('/'))
            save_dir = os.path.join(args.output_dir, "live_results", run_name)
            os.makedirs(save_dir, exist_ok=True)

            master_jsonl = os.path.join(save_dir, f"{dataset_basename}_live_results.jsonl")
            part_jsonl = os.path.join(save_dir, f"{dataset_basename}_live_results{suffix}.jsonl") if suffix else None
            pretty_json = (
                os.path.join(save_dir, f"{dataset_basename}_live_results{suffix}.json") if suffix
                else os.path.join(save_dir, f"{dataset_basename}_live_results.json")
            )

            with open(master_jsonl, "a", encoding="utf-8") as fjsonl:
                for rec in live_results:
                    json.dump(rec, fjsonl)
                    fjsonl.write("\n")

            if part_jsonl is not None:
                with open(part_jsonl, "w", encoding="utf-8") as fpart:
                    for rec in live_results:
                        json.dump(rec, fpart)
                        fpart.write("\n")

            if not suffix:
                all_recs = []
                try:
                    with open(master_jsonl, "r", encoding="utf-8") as fmaster:
                        for line in fmaster:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                all_recs.append(json.loads(line))
                            except Exception:
                                continue
                except FileNotFoundError:
                    all_recs = []

                with open(pretty_json, "w", encoding="utf-8") as fjson:
                    json.dump(all_recs, fjson, indent=2)

                print(f"Saved final {len(all_recs)} live results to {save_dir}")
            else:
                print(f"Appended {len(live_results)} live results to {master_jsonl} and saved part {part_jsonl}")
        except Exception as e:
            print(f"Failed to save live results: {e}")

    try:
        counter = 0
        while True:
            try:
                item = next(eval_generator)
                live_results.append(item)
                counter += 1
                if counter % 20 == 0:
                    _save_live_results_snapshot(live_results, suffix=f"_part{counter}")
                    live_results.clear()
                    # ── Mid-run token snapshot ───────────────────────────
                    TOKEN_STATS.snapshot(label=f"after {counter} images")
                    # ─────────────────────────────────────────────────────
            except StopIteration as e:
                ds_stats = e.value
                break
    except Exception as e:
        print(f"Error while consuming eval_generator: {e}")

    if live_results:
        _save_live_results_snapshot(live_results, suffix="")
    else:
        _save_live_results_snapshot([], suffix="")

    if ds_stats is not None:
        print("\n--- Summary of Results ---")
        print(f"[model] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['model'][0]:.4f}")
        print(f"[ranking] mAP (AP50-95) for {os.path.basename(dataset_path)}: {ds_stats['ranking'][0]:.4f}")
        print(f"[model] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['model'][6]:.4f}")
        print(f"[ranking] AR@1 for {os.path.basename(dataset_path)}: {ds_stats['ranking'][6]:.4f}")
    else:
        print(f"Evaluation failed for {dataset_path}")

    # ── Final token report ───────────────────────────────────────────────────
    print(TOKEN_STATS.report())
    dataset_name = os.path.basename(dataset_path)
    token_stats_path = os.path.join(args.output_dir, f"{dataset_name}_token_stats.json")
    TOKEN_STATS.save(token_stats_path)
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--root_path", type=str, default="./datasets/rf100-vl-fsod/")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/rf100vl-zeroshot/rf20_IPT_singleclass_vqaScore_withNMS")
    parser.add_argument("--data_instr_path", type=str, default="./data_instr/default/README.dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--server_url', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--vqa_batch_size', type=int, default=8)
    parser.add_argument("--rank_rescore", action="store_true")
    parser.add_argument("--siglip_rescore", action="store_true")

    args = parser.parse_args()
    run_single_dataset_evaluation(args)