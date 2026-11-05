# Counterfactual Reward Study — code, data and evaluation

Release for the study **"The Right Kind of Wrong: Counterfactual Data for Reward Learning in Manipulation"**.

- Project page: https://rightkindofwrong.github.io/
- Authors: Gokul Kannan Thiruvenkadam, Vishal M. Patel†, Homanga Bharadhwaj† (Johns Hopkins University; † equal advising)

> **This is a study, not a new framework.** We did not build a new reward-model or RL codebase. Every
> result uses existing open-source code, with only small study-specific additions:
>
> | Component | What we used | Link |
> |---|---|---|
> | Reward model (LoRA fine-tuning, inference, evals) | **Robometer** — our fork adds counterfactual-data loaders, LoRA/preprocess configs, the few-shot GVL baseline and paper metrics (branch `counterfactual-study`) | upstream: https://github.com/robometer/robometer · fork: https://github.com/g0kul6/robometer |
> | RL post-training of π₀ with the learned rewards | **robometer-policy-learning** (fork, unmodified pointer) | upstream: https://github.com/robometer/robometer-policy-learning · fork: https://github.com/g0kul6/robometer-policy-learning |
> | Real-robot (DROID/FR3) setup | **droid-fr3** | https://github.com/EmbodiedForge/droid-fr3 |
> | Frozen-VLM baseline | **GVL** (Ma et al., 2024), few-shot variant (below) | — |
>
> Please also cite and respect the licences of those projects.

## What the study asks
Given a limited data-collection budget, which sources of counterfactual (failure) experience are worth
collecting for training a robot reward model? We compare **kinesthetic teaching (teleop), policy rollouts,
human videos and real-to-sim rollouts** across five real-world manipulation tasks, holding the task suite,
successful demonstrations and evaluation fixed. We (i) LoRA-adapt Robometer and (ii) few-shot prompt a frozen
VLM (GVL), evaluate on a held-out benchmark of **250 human-annotated policy rollouts** (trajectory ranking,
partial-completion calibration, success detection), and use selected reward models to guide DSRL post-training
of π₀.

## Repository layout
```
annotations/        human annotations (reward 0–1 partial credit, failure tags, cut-off frames), anonymised
gvl_fewshot/        few-shot GVL: context banks (5 demos/task/source) + runner
eval/               metric + bootstrap code, plotting/aggregation scripts
docs/               full training / GVL hyper-parameter write-ups
```

## Data (Hugging Face)
| Dataset | Contents | Link |
|---|---|---|
| Benchmark | 250 annotated held-out policy rollouts (π₀.₅ and MolmoAct2) in Robometer format | [g0kul6/counterfactual-reward-benchmark](https://huggingface.co/datasets/g0kul6/counterfactual-reward-benchmark) |
| Training sources | teleop (kinesthetic), policy rollouts (π₀), human videos, real-to-sim HiFi/LoFi, with annotations | [g0kul6/counterfactual-reward-train](https://huggingface.co/datasets/g0kul6/counterfactual-reward-train) |

Data licence: CC BY 4.0. Annotations were produced by a single annotator under rubric v3.0.0
(see `docs/` and the paper). Videos of human hands/arms are included; no faces are intended to be visible.

Download and set the video root used by the context banks:
```bash
huggingface-cli download g0kul6/counterfactual-reward-train --repo-type dataset --local-dir data/train
export COUNTERFACTUAL_DATA_ROOT=$PWD/data/train
```

## 1. Fine-tuning Robometer (LoRA) on a counterfactual source
```bash
git clone -b counterfactual-study https://github.com/g0kul6/robometer && cd robometer
uv sync            # see upstream README for environment details
# 1. edit /path/to/counterfactual_study placeholders in robometer/configs/preprocess_counterfactual*.yaml
export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed
uv run python -m robometer.data.scripts.preprocess_datasets --config robometer/configs/preprocess_counterfactual_teleop_left.yaml --cache_dir=$ROBOMETER_PROCESSED_DATASETS_PATH
# 2. LoRA training (base: robometer/Robometer-4B; r=16, alpha=32, dropout 0.05)
bash run_scripts/run_counterfactual_lora_4gpu.sh      # edit paths/GPU ids first
```
Source configs: `*_teleop_left`, `*_rollout_left`, `*_human_left`, plus scaling/mix variants.
All hyper-parameters (LoRA, heads, schedule, scaling and mix experiments) are in
[`docs/TRAINING_AND_GVL_DETAILS.md`](docs/TRAINING_AND_GVL_DETAILS.md).

## 2. Few-shot GVL (frozen VLM, label = whole-trajectory average partial credit)
This is the only GVL variant released. For each benchmark trajectory, GVL sees 30 frames in shuffled order
(first frame as anchor). Few-shot conditions prepend **5 same-task demonstrations** from one source
(`teleop`, `rollout`, `human`, `hifi`), each shown as shuffled frames followed by **one whole-trajectory
average-behaviour partial-credit label** (`label_mode=trajectory_average_reward`; a trajectory-level scalar,
not a per-frame or terminal-state label). Zero-shot has no context. Shuffle seeds 42/43/44; GPT-4o,
`detail=low`.
```bash
export OPENAI_API_KEY=sk-...
export ROBOMETER_REPO=/path/to/robometer COUNTERFACTUAL_DATA_ROOT=/path/to/data/train
export BENCHMARK=/path/to/processed/eval_benchmark OUT=/path/to/gvl_out
bash gvl_fewshot/run_gvl_fewshot.sh
```
Context banks are in `gvl_fewshot/context_banks/*.json` (video paths relative to
`COUNTERFACTUAL_DATA_ROOT`). Details: [`docs/GVL_ZERO_VS_FEW_SHOT.md`](docs/GVL_ZERO_VS_FEW_SHOT.md).
Responses are cached per request, so interrupted runs resume without repeating API calls.

## 3. Evaluation and metrics
Predictions are written as `reward_alignment/*_results.json` (30 values per trajectory). Metrics
(Kendall τ last/mean, MAE last/mean, extreme-reward gap, ROC-AUC, max-progress-minus-partial-credit) with
trajectory-level bootstrap CIs come from `eval/counterfactual_paper_metrics.py` (also in the Robometer fork);
`eval/watch_*_results.py` aggregate seeds. Paths inside those scripts are `/path/to/counterfactual_study`
placeholders — edit before use.

## Reproducibility notes
- Absolute paths in configs are replaced by `/path/to/...` placeholders; set them for your machine.
- RL post-training and real-robot setup follow the linked repos; policy checkpoints are not included.
- GVL requires an OpenAI key and incurs API cost (one call per trajectory per condition).

## Citation
```bibtex
@misc{thiruvenkadam2026rightkindofwrong,
  title  = {The Right Kind of Wrong: Counterfactual Data for Reward Learning in Manipulation},
  author = {Thiruvenkadam, Gokul Kannan and Patel, Vishal M. and Bharadhwaj, Homanga},
  year   = {2026},
  url    = {https://rightkindofwrong.github.io/}
}
```

## Licence
Code in this repository: MIT. Data: CC BY 4.0. Upstream projects retain their own licences.
