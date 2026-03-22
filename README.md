# DetPO: In-Context Learning with Multi-Modal LLMs for Few-Shot Object Detection:

A framework for zero-shot object detection using Vision-Language Models (VLMs), featuring iterative prompt refinement, confidence rescoring, and COCO-format evaluation. Inference is served through a **vLLM OpenAI-compatible HTTP server**, decoupling model hosting from the evaluation pipeline.

This repository contains scripts to set up the environment and run DetPO-based prompt optimization for object detection.

## Setup

Follow the steps below **in order** to prepare the environment.

### 1. Install the Conda Environment

Create and set up the Conda environment using the provided setup script. The setup script needs ROBOFLOW_API_KEY to download RF20-datasets:

```bash
bash setup.sh ROBOFLOW_API_KEY
```

After the script completes, activate the environment:

```bash
conda activate detpo-env
```

## Usage

All scripts require a running vLLM server. Start it before running any evaluation or optimization script.

### 0. Launch the vLLM Server

```bash
bash detpo/launch_vllm_server.sh
# override model or port:
bash detpo/launch_vllm_server.sh --model Qwen3-VL-8B-Instruct --port 8001
```

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1` by default. All scripts accept `--server_url` to point at a different host/port.

Once the server prints `Application startup complete`, proceed with the steps below.

### Detection Prompt Optimization (DetPO)

Automatically refine class descriptions over N iterations, then run final evaluation:

```bash
python -m detpo.run_detpo_optimization \
    --model_name Qwen3-VL-30B-A3B-Instruct \
    --root_path ./datasets/rf100-vl-fsod/ \
    --dataset_path my-dataset \
    --output_dir results/ipt_output \
    --ipt_mode \
    --num_ipt_iterations 5 \
    --num_samples 10 \
    --vqa_rescore
```

### Zero-Shot Evaluation

Evaluate a dataset on the test split using existing class instructions:

```bash
python -m detpo.run_evaluation \
    --model_name Qwen3-VL-8B-Instruct \
    --root_path ./datasets/rf100-vl-fsod/ \
    --dataset_path my-dataset \
    --data_instr_path ./data_instr/default/README.dataset \
    --output_dir results/eval_output \
    --vqa_rescore
```

### VQA Rescoring (standalone)

Re-score detections from a prior inference run without re-running detection. Reads pre-computed rank-scored predictions from `<output_dir>/live_results/rank/`.

```bash
python -m detpo.run_vqa_rescore \
    --model_name Qwen3-VL-30B-A3B-Instruct \
    --root_path ./datasets/rf100-vl-fsod/ \
    --dataset_path my-dataset \
    --output_dir results/eval_output \
    --data_instr_type ipt \
    --vqa_rescore
```


### Optional Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_name` | Qwen model variant served by the vLLM server | `Qwen3-VL-235B-A22B-Instruct` |
| `--server_url` | Base URL of the vLLM OpenAI-compatible server | `http://localhost:8000/v1` |
| `--root_path` | Root directory containing dataset subdirectories | `./datasets/rf100-vl-fsod/` |
| `--dataset_path` | Name of the specific dataset subdirectory | required |
| `--output_dir` | Directory for saving results, predictions, and visuals | required |
| `--data_instr_path` | Path prefix for class instruction JSON files | `./data_instr/default/README.dataset` |
| `--data_instr_type` | `ipt` for refined instructions, `default` for README defaults | — |
| `--seed` | Random seed for reproducibility | `42` |
| `--vqa_rescore` | Re-score detections using a VQA yes/no prompt | off |
| `--siglip_rescore` | Re-score detections using SigLIP zero-shot classification | off |
| `--ipt_mode` | Enable iterative prompt refinement | off |
| `--num_ipt_iterations` | Number of IPT refinement iterations per class | `3` |
| `--num_samples` | Max annotations per class for train/val subsampling during IPT | `None` (use all) |
| `--vqa_batch_size` | Batch size for VQA rescoring calls | `8` |

## Dataset Requirements

The script expects a dataset structure similar to **Roboflow COCO exports**. The directory at `--dataset` must contain:

1. **`README.dataset.txt`**: Used to parse class metadata and instructions.
2. **`train/` directory**: Contains images and `_annotations.coco.json`.
3. **`valid/` directory**: Contains images and `_annotations.coco.json`.
3. **`test/`  directory**: Contains images and `_annotations.coco.json`.

**Example Structure:**

```text
/my-dataset
  ├── README.dataset.txt
  ├── train/
  │   ├── _annotations.coco.json
  │   └── image1.jpg ...
  └── valid/
      ├── _annotations.coco.json
      └── image2.jpg ...
  └── test/
      ├── _annotations.coco.json
      └── image3.jpg ...

```


## How IPT Works

For each class in the dataset, IPT runs the following loop:

```
1. Generate initial class definition
     └── Uses all GT training examples (green boxes) + negative examples
         from other classes (red boxes) to produce a rich textual definition

2. For N iterations:
     a. Run inference on the training split using the current definition
     b. Evaluate with COCO metrics (mAP, AR)
     c. Identify the worst false positive (highest-confidence wrong detection)
        and worst false negative (most-missed GT object)
     d. Show the VLM a correct example alongside the error case and ask it
        to refine the class definition to fix that error type
     e. Only accept the new definition if mAP does not decrease

3. Evaluate all candidate definitions on the validation split
     └── Selects the best-performing definition as the final output

4. Save all refined definitions to:
     <output_dir>/iterative_prompt_refinement/all_refined_class_instructions_<dataset>.json
```

After IPT, the evaluator automatically runs a final test-split evaluation using the refined definitions.

---

## Output Structure

```
<output_dir>/
├── predictions/                          # Cached COCO-format prediction JSON files
├── evaluations/                          # COCO eval stats per iteration and eval type
├── visuals/                              # Visualizations of GT vs predicted boxes
├── live_results/                         # Per-image inference results (JSONL + JSON)
├── iterative_prompt_refinement/
│   ├── <dataset>/
│   │   ├── <class>_original_definition.txt
│   │   ├── <class>_initial_definition.txt
│   │   ├── <class>_best_instructions_<dataset>_cls_<class>.txt
│   │   ├── <class>_refined_instructions_<dataset>_cls_<class>.txt
│   │   ├── instruction_refinements_log_<dataset>_cls_<class>.json
│   │   └── ipt_state_<dataset>_<class>.json   # Resume checkpoint
│   └── all_refined_class_instructions_<dataset>.json
├── <dataset>_token_stats.json            # Token usage summary
└── final_instruction_eval/               # Final test-split evaluation results
```

---

## Confidence Rescoring Modes

Three rescoring strategies are available and can be selected via flags:

| Mode | Flag | Description |
|---|---|---|
| **Model score** | (default) | Uses the raw confidence score from the model's JSON output |
| **VQA rescore** | `--vqa_rescore` | For each detected box, draws it on the image and asks the model "Is `<class>` inside the red box? Yes/No" — uses the Yes/No log-probability ratio as the new score |
| **SigLip rescore** | `--siglip_rescore` | Crops each detected box and scores it with a SigLIP zero-shot classifier |

---

## Token Usage Tracking

All scripts share a module-level `TOKEN_STATS` singleton (defined in `utils.py`). Token counts are read from the `usage` field of each OpenAI-compatible API response returned by the vLLM server.

Counts are broken down by stage:

| Stage | What it covers |
|---|---|
| `detection` | Main object detection inference calls |
| `vqa_score` | VQA yes/no rescoring calls |
| `vqa_score_with_instructions` | VQA rescoring with dataset instructions |
| `class_def_initial` | Initial class definition generation |
| `class_def_fp` | False-positive-focused definition refinement |
| `class_def_fn` | False-negative-focused definition refinement |
| `class_def_refine` | Final definition polishing step |

A summary is printed to stdout and saved to `<output_dir>/<dataset>_token_stats.json` at the end of every run. Mid-run snapshots are printed every 20 images.

---

## Resume Support

Long runs can be interrupted and resumed without restarting from scratch:

- **Prediction caching** — completed per-image predictions are saved incrementally to JSON every 5 images. On restart, already-processed image IDs are skipped.
- **IPT iteration state** — after each iteration, a `ipt_state_<dataset>_<class>.json` checkpoint is written containing the current instructions, best instructions, mAP scores, and iteration number. The loop resumes from the last completed iteration.
- **Class-level resume** — refined instructions for completed classes are saved to `all_iterm_refined_class_instructions_<dataset>.json` after each class finishes. Classes already present in this file are skipped on restart.

---

## Notes

- **Coordinate system** — Qwen3-VL outputs relative coordinates in the range 0–1000. Qwen2.5-VL outputs absolute pixel coordinates. The code handles both automatically based on `--model_name`.
- **vLLM server** — GPU count, tensor parallelism, expert parallelism, and memory utilization are all configured in `detpo/launch_vllm_server.sh`. The client scripts are GPU-agnostic and communicate with the server over HTTP.
- **Qwen2.5-VL processor** — when using a Qwen2.5-VL model, the `AutoProcessor` is still loaded on CPU by the client to compute image patch dimensions for bounding-box coordinate rescaling. No GPU is required on the client side.
- **Image resizing** — images larger than 2880×1620 are automatically downsampled before being sent to the server. If a request still fails, a 1280×720 fallback is attempted.
