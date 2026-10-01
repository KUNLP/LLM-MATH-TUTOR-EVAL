# LLM-MATH-TUTOR-EVAL
Official code for "Interactive and Scalable Evaluation of LLM Math Tutoring via Simulated Teacher-Student Dialogues" (Findings of AACL).
This framework evaluates LLM math tutors through simulated teacher–student dialogues with automatic answer verification. It measures error correction (ECR), feedback efficiency (FEI), answer reachability gain (ARG), and teacher-judgment F1, with a hint-only test for overinformative feedback.

## Installation

Use Linux, Python 3.12, and CUDA-capable GPUs with enough memory for the selected models.

```bash
python -m pip install vllm==0.25.1 transformers==5.5.3 \
  hydra-core datasets math-verify==0.9.0 openai tqdm openpyxl
```

Set `HF_TOKEN` when using gated Hugging Face models and `OPENAI_API_KEY` when using API models. Full evaluation, including OI and ARG, requires a local student model.

## Usage

Run commands from the repository root. Configure your experiment in:

- `configs/model_list.yaml`: model names and generation settings.
- `configs/config.yaml`: teacher/student models, dataset, feedback budget, and GPUs.
- `configs/analyze.yaml`: teacher/student model lists to evaluate, checker model, and evaluation GPUs.

Use separate GPUs for the local teacher and student. Set the evaluation model lists to match your completed simulation runs, and adjust GPU IDs and file paths for your environment.

### 1. Generate dialogues

```bash
python tutoring_simulator.py paths.cache_dir=cache run.id=demo \
  test_args.max_feedback_count=4
```

GSM8K is downloaded automatically with the default dataset setting. Add `test_args.limit=10` for a small test. The feedback budget counts student retries after hints: `4` allows up to five student attempts, matching the paper's five-round budget.

For the paper's MATH subset, use `test_args.target_data=math_prm800k` and set `paths.dataset_file` to the PRM800K MATH test JSONL file. Set `evaluation.target_data` accordingly when evaluating.

### 2. Evaluate dialogues

After matching the model lists in `configs/analyze.yaml` to the simulation:

```bash
python analyze_simulation_result.py paths.cache_dir=cache run.id=demo-eval
```

The evaluator selects the latest completed run for each configured teacher–student pair. Add `run.resume=true` with the same run ID and settings to resume an interrupted simulation or evaluation.

## Outputs

- `output/from_tutoring_simulator/`: generated dialogues and run metadata.
- `output/from_analyze_simulation_result/`: evaluation annotations and reports in JSON, CSV, and Excel.

OI-filtered ECR and FEI are reported as `real_error_correction_rate` and `real_feedback_efficiency_index`; unfiltered scores are also retained.
