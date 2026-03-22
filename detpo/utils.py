import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams

from tqdm import tqdm
import gc
import argparse

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PIL import Image, ImageDraw

import json
import time
import re

import numpy as np
import random

from transformers import pipeline

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Optional
import threading


# =============================================================================
# Token Usage Tracking
# =============================================================================

@dataclass
class TokenStats:
    """
    Thread-safe accumulator for vLLM token usage.

    vLLM's RequestOutput carries per-request token counts in:
        output.metrics.num_prompt_tokens   (int)
        output.metrics.num_generation_tokens (int)

    We read these directly from the outputs returned by model.generate().
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    stage_breakdown: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
    )
    _lock: object = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def record_outputs(self, outputs, stage: str = "unknown"):
        """
        Extract token counts from a list of vLLM RequestOutput objects and
        add them to the running totals.  Call this immediately after every
        model.generate() call.

            outputs = model.generate(inputs, sampling_params=sampling_params)
            TOKEN_STATS.record_outputs(outputs, stage="detection")
        """
        if outputs is None:
            return

        prompt_toks = 0
        completion_toks = 0
        n_calls = 0

        for req_output in outputs:
            # vLLM >= 0.4: token counts live in req_output.metrics
            metrics = getattr(req_output, "metrics", None)
            if metrics is not None:
                p = getattr(metrics, "num_prompt_tokens", None)
                c = getattr(metrics, "num_generation_tokens", None)
                if p is not None and c is not None:
                    prompt_toks += p
                    completion_toks += c
                    n_calls += 1
                    continue

            # Fallback: count tokens from the output sequences themselves.
            # prompt_token_ids is always available; sum output token_ids for completion.
            p = len(getattr(req_output, "prompt_token_ids", []) or [])
            c = sum(
                len(getattr(seq_out, "token_ids", []) or [])
                for seq_out in (req_output.outputs or [])
            )
            prompt_toks += p
            completion_toks += c
            n_calls += 1

        with self._lock:
            self.prompt_tokens += prompt_toks
            self.completion_tokens += completion_toks
            self.total_tokens += prompt_toks + completion_toks
            self.call_count += n_calls
            self.stage_breakdown[stage]["prompt"] += prompt_toks
            self.stage_breakdown[stage]["completion"] += completion_toks
            self.stage_breakdown[stage]["calls"] += n_calls

    def snapshot(self, label: str = ""):
        """Print a one-line summary of current totals."""
        tag = f" [{label}]" if label else ""
        print(
            f"[TokenStats{tag}]"
            f"  calls={self.call_count:,}"
            f"  prompt={self.prompt_tokens:,}"
            f"  completion={self.completion_tokens:,}"
            f"  total={self.total_tokens:,}"
        )

    def report(self) -> str:
        lines = [
            "\n" + "=" * 60,
            "              TOKEN USAGE SUMMARY",
            "=" * 60,
            f"  {'Calls':<25} {self.call_count:>12,}",
            f"  {'Prompt Tokens':<25} {self.prompt_tokens:>12,}",
            f"  {'Completion Tokens':<25} {self.completion_tokens:>12,}",
            f"  {'Total Tokens':<25} {self.total_tokens:>12,}",
        ]
        if self.stage_breakdown:
            lines += ["-" * 60, "  BY STAGE:"]
            for stage, counts in sorted(self.stage_breakdown.items()):
                lines.append(
                    f"  {stage:<25}"
                    f"  calls={counts['calls']:>6,}"
                    f"  in={counts['prompt']:>10,}"
                    f"  out={counts['completion']:>10,}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path):
        data = {
            "total": {
                "call_count": self.call_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "by_stage": {k: dict(v) for k, v in self.stage_breakdown.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Token stats saved to {path}")


# Module-level singleton — import and use anywhere in the project.
TOKEN_STATS = TokenStats()


# =============================================================================
# Seed utilities
# =============================================================================

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model loading
# =============================================================================

def load_siglip_pipeline():
    ckpt = "google/siglip2-base-patch16-naflex"
    pipe = pipeline(model=ckpt, task="zero-shot-image-classification")
    return pipe


def load_qwen_model(model_name):
    dtype = "auto" if "-FP8" in model_name else torch.bfloat16
    print(f"Loading using LLM class from vLLM with dtype: {dtype}")

    enable_expert_parallel = True if (
        model_name.startswith("Qwen3-VL-235B-A22B-Instruct-FP8") or
        model_name.startswith("Qwen3-VL-30B-A3B-Instruct")
    ) else False
    print(f"enable_expert_parallel: {enable_expert_parallel}")

    tensor_parallel_size = 4 if model_name.startswith("Qwen2.5-VL-7B") else torch.cuda.device_count()
    print(f"tensor_parallel_size: {tensor_parallel_size}")

    model = LLM(
        model="Qwen/" + model_name,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_expert_parallel=enable_expert_parallel,
        tensor_parallel_size=tensor_parallel_size,
        seed=0
    )

    processor = AutoProcessor.from_pretrained("Qwen/" + model_name)

    print("processor.tokenizer.padding_side:", processor.tokenizer.padding_side)
    print("processor.tokenizer.pad_token:", processor.tokenizer.pad_token)
    print("processor.tokenizer.eos_token:", processor.tokenizer.eos_token)

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    print("processor.tokenizer.padding_side:", processor.tokenizer.padding_side)
    print("processor.tokenizer.pad_token:", processor.tokenizer.pad_token)
    print("processor.tokenizer.eos_token:", processor.tokenizer.eos_token)

    return model, processor


# =============================================================================
# Model inference utils
# =============================================================================

def model_generate(messages, model, processor, token_stage: str = "generate"):
    """
    Run a standard (non-scored) generation pass.
    Token usage is recorded into TOKEN_STATS under ``token_stage``.
    """
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs_org = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")

    with torch.no_grad():
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs

        inputs = {
            'prompt': text_input,
            'multi_modal_data': mm_data,
        }
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=2048,
            top_k=-1,
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params=sampling_params)

        # ── Token tracking ──────────────────────────────────────────────
        TOKEN_STATS.record_outputs(outputs, stage=token_stage)
        # ────────────────────────────────────────────────────────────────

        output_text = None
        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text

    return output_text, inputs_org


def model_generate_with_scores(conversations, model, processor, max_new_tokens=1,
                               token_stage: str = "generate_with_scores"):
    """
    Run a generation pass that also returns per-token logprobs.
    Token usage is recorded into TOKEN_STATS under ``token_stage``.
    """
    text_input = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversations)
    inputs_org = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")

    with torch.no_grad():
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs

        inputs = {
            'prompt': text_input,
            'multi_modal_data': mm_data,
        }
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            top_k=-1,
            logprobs=10,
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params=sampling_params)

        # ── Token tracking ──────────────────────────────────────────────
        TOKEN_STATS.record_outputs(outputs, stage=token_stage)
        # ────────────────────────────────────────────────────────────────

        output_text = None
        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text

    return output_text, inputs_org, outputs


# =============================================================================
# SigLip utils
# =============================================================================

def rescore_with_siglip(siglip_pipe, pil_image, candidate_label):
    output = siglip_pipe(pil_image, candidate_labels=[candidate_label])
    assert len(output) == 1, "Error: SigLip output length is not 1."
    return output[0]['score']


# =============================================================================
# VQA utils
# =============================================================================

def get_masked_image_vqa_scores(qwen_model, qwen_processor, prompt_list, pil_images: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    Token usage is recorded under stage "vqa_score".
    """
    def getPrompt(prompt):
        question = (
            f"Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box "
            f"in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, "
            f"with no part outside, and it must be the only object present inside - no other objects should appear "
            f"within the box."
        )
        return question

    all_final_scores = []
    for i in range(0, len(pil_images)):
        img = pil_images[i]
        prompt = prompt_list[i]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": getPrompt(prompt)}
        ]}]

        output_text, inputs_org, outputs = model_generate_with_scores(
            messages, qwen_model, qwen_processor,
            token_stage="vqa_score"
        )

        assert len(outputs) == 1, "Error: Expected single output for single input."

        token_logprobs = outputs[0].outputs[0].logprobs[0]

        yes_logprob = None
        no_logprob = None

        for token_id, token_info in token_logprobs.items():
            logprob = token_info.logprob
            decoded_token = token_info.decoded_token

            if "Yes" == decoded_token:
                yes_logprob = logprob
            if yes_logprob is None and "yes" == decoded_token:
                yes_logprob = logprob
            if "No" == decoded_token:
                no_logprob = logprob
            if no_logprob is None and "no" == decoded_token:
                no_logprob = logprob

        if yes_logprob is None and no_logprob is None:
            all_final_scores.append(-1.0)
            continue
        if yes_logprob is None:
            no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
            yes_prob = 1 - no_prob
            all_final_scores.append(yes_prob.item())
            continue

        yes_prob = torch.exp(torch.tensor(yes_logprob)) if yes_logprob is not None else torch.tensor(0.0)
        no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
        score = yes_prob / (yes_prob + no_prob + 1e-18)
        all_final_scores.append(score.item())

    return np.array(all_final_scores)


def get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json,
                                                   prompt_list, pil_images: list, batch_size: int = 8):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt + dataset instructions.
    Token usage is recorded under stage "vqa_score_with_instructions".
    """
    def getDatasetInstructions(dataset_instructions_json, class_name):
        if class_name in dataset_instructions_json:
            return dataset_instructions_json[class_name]
        matched_key = next(
            (key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None
        )
        if matched_key:
            return dataset_instructions_json[matched_key]
        raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")

    def getPrompt(prompt, dataset_instructions_json):
        question = f"""
            Given the '{prompt}' class defined as follows: {getDatasetInstructions(dataset_instructions_json, prompt)}

            Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box.
        """
        return question

    all_final_scores = []
    for i in range(0, len(pil_images)):
        img = pil_images[i]
        prompt = prompt_list[i]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": getPrompt(prompt, dataset_instructions_json)}
        ]}]

        output_text, inputs_org, outputs = model_generate_with_scores(
            messages, qwen_model, qwen_processor,
            token_stage="vqa_score_with_instructions"
        )

        assert len(outputs) == 1, "Error: Expected single output for single input."

        token_logprobs = outputs[0].outputs[0].logprobs[0]

        yes_logprob = None
        no_logprob = None

        for token_id, token_info in token_logprobs.items():
            logprob = token_info.logprob
            decoded_token = token_info.decoded_token

            if "Yes" == decoded_token:
                yes_logprob = logprob
            if yes_logprob is None and "yes" == decoded_token:
                yes_logprob = logprob
            if "No" == decoded_token:
                no_logprob = logprob
            if no_logprob is None and "no" == decoded_token:
                no_logprob = logprob

        if yes_logprob is None and no_logprob is None:
            all_final_scores.append(-1.0)
            continue
        if yes_logprob is None:
            no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
            yes_prob = 1 - no_prob
            all_final_scores.append(yes_prob.item())
            continue

        yes_prob = torch.exp(torch.tensor(yes_logprob)) if yes_logprob is not None else torch.tensor(0.0)
        no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
        score = yes_prob / (yes_prob + no_prob + 1e-18)
        all_final_scores.append(score.item())

    return np.array(all_final_scores)


# =============================================================================
# Geometry / NMS utils
# =============================================================================

def calculate_iou(boxA_xywh, boxB_xywh):
    """
    Calculates IoU.  Boxes are in [x, y, w, h] format.
    """
    boxA = [boxA_xywh[0], boxA_xywh[1], boxA_xywh[0] + boxA_xywh[2], boxA_xywh[1] + boxA_xywh[3]]
    boxB = [boxB_xywh[0], boxB_xywh[1], boxB_xywh[0] + boxB_xywh[2], boxB_xywh[1] + boxB_xywh[3]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def apply_nms(detections, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression.  Each detection is a dict with 'bbox' ([x,y,w,h]) and 'score'.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    kept_detections = []
    while detections:
        best_det = detections.pop(0)
        kept_detections.append(best_det)
        remaining_detections = []
        for det in detections:
            if calculate_iou(best_det['bbox'], det['bbox']) < iou_threshold:
                remaining_detections.append(det)
        detections = remaining_detections

    return kept_detections


def assign_score_based_on_ranking(parsed_bboxes, max_score=1.0, min_score=0.1):
    """
    Assigns a confidence score to each detection based on its rank.
    """
    ranked_detections = [det.copy() for det in parsed_bboxes]
    num_detections = len(ranked_detections)

    for i, det in enumerate(ranked_detections):
        det['model_score'] = det['score']
        det['rank_score'] = (
            max_score - (max_score - min_score) * (i / (num_detections - 1))
            if num_detections > 1 else max_score
        )
        det['score'] = det['rank_score']

    return ranked_detections


# =============================================================================
# Qwen inference
# =============================================================================

def run_qwen_inference(args, model, processor, image, dataset_instructions, class_name):
    """
    Given a model, processor, image, instructions, and class_name, run Qwen VL
    and return the raw text output.  Token usage is recorded under "detection".
    """
    set_seed(args.seed)

    prompt_text = (
        f"""
            Identify and localize all instances of "{class_name}" in the image.

            Output Requirements:
            - Return valid JSON only. Do not include explanations or extra text.
            - Output a ranked list of detections sorted by confidence (highest first).
            - Include at most 20 detections.
            - If no objects are detected, return an empty list [].

            For each detection, provide:
            - "bbox_2d": [x1, y1, x2, y2]
                * Pixel coordinates.
                * (x1, y1) = top-left corner.
                * (x2, y2) = bottom-right corner.
            - "label": "{class_name}"
            - "score": float confidence score from 0.0 (lowest) to 1.0 (highest) indicating the likelihood that the bounding box contains the specified object.

            Additional Constraints:
            - Only include detections that clearly correspond to "{class_name}".
            - Avoid duplicate or highly overlapping boxes for the same object.
            - Follow these annotator instructions to improve detection accuracy:

            {dataset_instructions}

            Return a JSON list in the following format:
            [
            {{
                "bbox_2d": [x1, y1, x2, y2],
                "label": "{class_name}",
                "score": 0.95
            }}
            ]
            """
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
            ],
        }
    ]

    # model_generate already records token usage under stage "generate".
    # We pass a more descriptive stage name here.
    output_text, inputs = model_generate(messages, model, processor, token_stage="detection")

    input_height = inputs['image_grid_thw'][0][1] * 14
    input_width = inputs['image_grid_thw'][0][2] * 14

    return output_text, input_width, input_height, None


# =============================================================================
# Output parsing
# =============================================================================

def parse_qwen_output_to_detections(output_text, class_name_list, output_dir="."):
    """
    Robust parser for Qwen detection output.
    Returns a list of detections: {"bbox": [x, y, w, h], "score": float, "category_name": str}
    """
    skipped_log_path = os.path.join(output_dir, "skipped_detections.log")

    Flag_log_output_text = True

    def log_skipped(reason, item, output_text, Flag_log_output_text):
        with open(skipped_log_path, "a") as f:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "item": item,
            }
            if Flag_log_output_text:
                log_entry["original_output"] = output_text
            f.write(json.dumps(log_entry) + "\n")
        return False

    text_clean = re.sub(r'```(?:json)?\s*', '', output_text)
    text_clean = text_clean.replace('```', '')
    text_clean = re.sub(r'"bbox_2d\s*\[(.*?)\]', r'"bbox_2d":[\1]', text_clean)

    detections = []
    data = []
    try:
        data = json.loads(text_clean)
        if not isinstance(data, list):
            reason = "Top-level JSON is not a list"
            print(f"{reason}; skipping.")
            Flag_log_output_text = log_skipped(reason, text_clean, output_text, Flag_log_output_text)
            return detections
    except json.JSONDecodeError:
        reason = "Malformed JSON"
        print(f"{reason}, attempting to salvage valid detections.")
        print(text_clean)
        object_strings = re.findall(r'\{[^}]+\}', text_clean)
        for obj_str in object_strings:
            try:
                obj = json.loads(obj_str)
                data.append(obj)
            except json.JSONDecodeError:
                try:
                    obj = json.loads(obj_str.replace('="', '":').replace(']"', ']').replace('"[', '['))
                    data.append(obj)
                except json.JSONDecodeError:
                    reason = "Skipping malformed object during salvage"
                    print(f"{reason}: {obj_str}")
                    Flag_log_output_text = log_skipped(reason, obj_str, output_text, Flag_log_output_text)

    for item in data:
        try:
            if not isinstance(item, dict):
                reason = "Skipping item (not a dict)"
                print(f"{reason}: {item}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            bbox_2d = item.get("bbox_2d", [])
            if len(bbox_2d) != 4:
                reason = "Skipping invalid bbox_2d length"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            try:
                x1, y1, x2, y2 = map(float, bbox_2d)
            except Exception as e:
                reason = f"Skipping item due to conversion error: {e}"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            if x2 < x1 or y2 < y1:
                reason = "Skipping reversed coords in bbox_2d"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue
            w = x2 - x1
            h = y2 - y1
            if w == 0 or h == 0:
                reason = "Skipping zero dimension"
                print(f"{reason}: {bbox_2d}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            label = item.get("label", "unknown")
            if not isinstance(label, str):
                label = str(label)

            if label == "unknown":
                reason = "Skipping item (label is unknown)"
                print(f"{reason}: {item}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)
                continue

            if label not in class_name_list:
                lb_check = [c for c in class_name_list if c.lower() == label.lower()]
                partial_lb_check = [c for c in class_name_list if c.lower() in label.lower()]
                if len(lb_check) > 0:
                    label = lb_check[0]
                elif len(partial_lb_check) > 0:
                    label = partial_lb_check[0]
                else:
                    label = class_name_list[0]
                    reason = f"Label not in {class_name_list}, assigning default {class_name_list[0]}"
                    print(f"{reason}: {item}")
                    Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)

            score = float(item.get("score", -1.0))
            if score == -1.0:
                score = 0.5
                reason = f"Score is -1.0, assigning default 0.5"
                print(f"{reason}: {item}")
                Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)

            detections.append({
                "bbox": [x1, y1, w, h],
                "score": score,
                "category_name": label
            })

        except Exception as e:
            reason = f"Skipping item due to unexpected error: {e}"
            print(f"{reason}: {item}")
            Flag_log_output_text = log_skipped(reason, item, output_text, Flag_log_output_text)

    return detections


# =============================================================================
# Image resizing utils
# =============================================================================

def getMaxInputSizeForQwen(width, height, max_dimension=(2880, 1620)):
    aspect_ratio = width / height
    if aspect_ratio >= 1.0:
        if width > max_dimension[0]:
            new_width = max_dimension[0]
            new_height = int(new_width / aspect_ratio)
            max_dimension = (new_width, new_height)
        elif height > max_dimension[1]:
            new_height = max_dimension[1]
            new_width = int(new_height * aspect_ratio)
            max_dimension = (new_width, new_height)
    else:
        max_dimension = (max_dimension[1], max_dimension[0])
        if height > max_dimension[1]:
            new_height = max_dimension[1]
            new_width = int(new_height * aspect_ratio)
            max_dimension = (new_width, new_height)
        elif width > max_dimension[0]:
            new_width = max_dimension[0]
            new_height = int(new_width / aspect_ratio)
            max_dimension = (new_width, new_height)

    return max_dimension


def run_model_with_retries(args, model, processor, original_image, dataset_instructions, class_name):
    try:
        raw_output_i, input_width, input_height, outputs_probs = run_qwen_inference(
            args, model, processor,
            image=original_image,
            dataset_instructions=dataset_instructions,
            class_name=class_name,
        )
    except Exception as e:
        print(f"❌ Unexpected error during inference: {e}")
        print("Retrying with downsized image...")
        torch.cuda.empty_cache()

        resized_image = original_image.resize((1280, 720), Image.Resampling.LANCZOS)
        try:
            raw_output_i, input_width, input_height, outputs_probs = run_qwen_inference(
                args, model, processor,
                image=resized_image,
                dataset_instructions=dataset_instructions,
                class_name=class_name,
            )
            print("✅ Retry succeeded with downsized image.")
        except Exception as e:
            print(f"❌ Unexpected error during inference: {e}")
            torch.cuda.empty_cache()
            raw_output_i, input_width, input_height, outputs_probs = '', None, None, None

    return raw_output_i, input_width, input_height, outputs_probs


def run_inference_on_single_image(args, model, processor, image_path, dataset_instructions_json,
                                  class_name_list, output_dir=".", siglip_pipe=None):
    """
    Runs Qwen inference on a single image and parses the output.
    Token usage is accumulated automatically in TOKEN_STATS.
    """
    set_seed(args.seed)

    original_image = Image.open(image_path).convert("RGB")
    or_width, or_height = original_image.size
    width, height = original_image.size
    print(f"Original image size: {width}x{height}")

    max_dimension = (2880, 1620)
    if width > max_dimension[0] or height > max_dimension[1]:
        max_dimension = getMaxInputSizeForQwen(width, height, max_dimension)
        if width > max_dimension[0] or height > max_dimension[1]:
            print(f"Resizing image from {width}x{height} to fit within {max_dimension[0]}x{max_dimension[1]}")
            original_image = original_image.resize(max_dimension, Image.Resampling.LANCZOS)
            width, height = original_image.size
            print(f"Resized image size: {width}x{height}")

    raw_output = ""
    parsed_bboxes = []

    for class_name in class_name_list:
        if class_name in dataset_instructions_json:
            dataset_instructions = dataset_instructions_json[class_name]
        else:
            matched_key = next(
                (key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None
            )
            if matched_key:
                dataset_instructions = dataset_instructions_json[matched_key]
            else:
                raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")

        set_seed(args.seed)

        raw_output_i, input_width, input_height, outputs_probs = run_model_with_retries(
            args, model, processor, original_image, dataset_instructions, class_name
        )

        parsed_bboxes_i = parse_qwen_output_to_detections(raw_output_i, [class_name], output_dir=output_dir)

        if not args.model_name.startswith("Qwen2.5-VL"):
            input_height = 1000
            input_width = 1000
            assert args.model_name.startswith("Qwen3-VL")

        for det in parsed_bboxes_i:
            bbox = det["bbox"]
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            abs_y1 = int(y1 / input_height * or_height)
            abs_x1 = int(x1 / input_width * or_width)
            abs_y2 = int(y2 / input_height * or_height)
            abs_x2 = int(x2 / input_width * or_width)

            abs_w = abs_x2 - abs_x1
            abs_h = abs_y2 - abs_y1

            det["bbox"] = [abs_x1, abs_y1, abs_w, abs_h]
            det["bbox_model_xyxy"] = [x1, y1, x2, y2]

        parsed_bboxes.extend(parsed_bboxes_i)
        raw_output += f"\n\n--- For class '{class_name}' ---\n{raw_output_i}"

    detections_model = [det.copy() for det in parsed_bboxes]
    detections_ranking = []

    if args.rank_rescore and parsed_bboxes:
        detections_ranking = assign_score_based_on_ranking(parsed_bboxes, max_score=1.0, min_score=0.1)

    return raw_output, {
        "model": detections_model,
        "ranking": detections_ranking if args.rank_rescore else None,
    }


def run_rescorer(args, model, processor, image_path, dataset_instructions_json, parsed_bboxes, siglip_pipe=None):
    """
    Runs VQA or SigLip rescoring on pre-parsed bboxes.
    Token usage is accumulated automatically in TOKEN_STATS.
    """
    set_seed(args.seed)

    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    print(f"Original image size: {width}x{height}")

    max_dimension = (2880, 1620)
    if width > max_dimension[0] or height > max_dimension[1]:
        max_dimension = getMaxInputSizeForQwen(width, height, max_dimension)
        if width > max_dimension[0] or height > max_dimension[1]:
            print(f"Resizing image from {width}x{height} to fit within {max_dimension[0]}x{max_dimension[1]}")
            original_image = original_image.resize(max_dimension, Image.Resampling.LANCZOS)
            width, height = original_image.size
            print(f"Resized image size: {width}x{height}")

    if args.vqa_rescore and parsed_bboxes:
        vqa_images = [create_img_with_bbox(original_image, det["bbox"]) for det in parsed_bboxes]
        vqa_prompts = [det["category_name"] for det in parsed_bboxes]

        try:
            if args.vqa_nocontext:
                vqa_scores = get_masked_image_vqa_scores(model, processor, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size)
            else:
                vqa_scores = get_masked_image_vqa_scores_with_instructions(model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size)
        except Exception as e:
            print(f"❌ Unexpected error during inference: {e}")
            print("Retrying with downsized image...")
            torch.cuda.empty_cache()
            vqa_images_small = []
            for img in vqa_images:
                img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                vqa_images_small.append(img)
            vqa_images = vqa_images_small
            try:
                if args.vqa_nocontext:
                    vqa_scores = get_masked_image_vqa_scores(model, processor, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size)
                else:
                    vqa_scores = get_masked_image_vqa_scores_with_instructions(model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=args.vqa_batch_size)
                print("✅ Retry succeeded with downsized image.")
            except Exception as e:
                print(f"❌ Unexpected error during inference: {e}")
                torch.cuda.empty_cache()
                vqa_scores = [-1] * len(parsed_bboxes)

        detections_vqa = [det.copy() for det in parsed_bboxes]
        for i, det in enumerate(detections_vqa):
            det["vqa_score"] = vqa_scores[i]
            det["score"] = vqa_scores[i] if vqa_scores[i] != -1 else det["score"]

    elif args.siglip_rescore and parsed_bboxes:
        detections_siglip = [det.copy() for det in parsed_bboxes]
        for i, det in enumerate(detections_siglip):
            det["model_score"] = det["score"]
            x, y, w, h = map(int, det["bbox"])
            if w == 0 or h == 0:
                continue
            cropped_img = original_image.crop((x, y, x + w, y + h))
            siglip_score = rescore_with_siglip(siglip_pipe, cropped_img, det["category_name"])
            det["siglip_score"] = siglip_score
            det["score"] = siglip_score
    else:
        raise ValueError(
            "No rescoring method specified or parsed_bboxes is empty. "
            "Please provide valid parsed_bboxes and specify either vqa_rescore or siglip_rescore in args."
        )

    return {"vqa": detections_vqa} if args.vqa_rescore else {"siglip": detections_siglip}


# =============================================================================
# Drawing utils
# =============================================================================

def draw_bboxes_on_image(image, pred_bboxes, gt_bboxes):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for (x, y, w, h) in gt_bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=8)
    for (x, y, w, h) in pred_bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=8)
    return img_copy


def visualize_bboxes(image_path, pred_bboxes, gt_bboxes, save_path):
    image = Image.open(image_path).convert("RGB")
    img_with_boxes = draw_bboxes_on_image(image, pred_bboxes, gt_bboxes)
    img_with_boxes.save(save_path)
    print(f"Saved visualization to {save_path}")


def draw_colored_bboxes_on_image(image, color, bboxes):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for (x, y, w, h) in bboxes:
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=8)
    return img_copy


def create_img_with_bbox(original_image, bbox_xywh):
    """Draws a single red bounding box on an image."""
    img_with_bbox = original_image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]
    draw.rectangle(bbox_xyxy, outline='red', width=3)
    return img_with_bbox