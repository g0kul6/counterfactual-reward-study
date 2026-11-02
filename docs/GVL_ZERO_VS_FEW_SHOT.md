# GVL: Zero-shot vs Few-shot

**Paper:** *Vision Language Models are In-Context Value Learners* (Ma et al., 2024)

GVL uses a frozen VLM (Gemini-1.5-Pro in the main experiments) to estimate **task progress** from video frames.

For every video, the paper subsamples **30 frames** and asks the VLM to output a progress value in **0–100%**.

## 1. Zero-shot GVL

No labeled example trajectory is given.

Given a task instruction `g` and target video frames:

```text
frames = [o1, o2, ..., o30]
anchor = o1
shuffled = random_shuffle(frames)

values = VLM(
    task=g,
    anchor=anchor,
    frames=shuffled
)
# -> [v1, v2, ..., v30], each in 0..100
```

Simple math:


\tilde v_1,\ldots,\tilde v_T
= \mathrm{VLM}(\tilde o_1,\ldots,\tilde o_T; g,o_1)


The frames are shuffled so the VLM cannot simply say "later frame = more progress". The original first frame is kept as an **anchor** so the direction of progress is clear.

---



## 2. Terminal-reward few-shot GVL

This study does **not** fabricate framewise demonstration labels. A context
trajectory contributes cutoff-aware shuffled images plus exactly one scalar:
its annotated whole-trajectory terminal partial-credit reward.

```text
demonstration = shuffled(images[raw_start_frame:end_frame])
label = one terminal reward in {0.0, 0.2, ..., 1.0}
```

The target task loads one source-specific bank (`teleop`, `rollout`, or
`human`). Each task uses up to five unique same-task trajectories, exactly one
success when available, and reward/duration-diverse failures. Simulation is
excluded. Teleop tasks with fewer than five eligible unique examples use all
available examples without duplication.

`label_mode=terminal_reward` is the only implemented label mode. The prompt
explicitly states that each scalar applies to the complete demonstration, not
to any individual image.

## 3. Reproducibility and caching

Run commands from `reward_model/robometer`. All target and demonstration
frames are sampled through their annotated cutoff. Shuffles are deterministic
from `shuffle_seed`, task/trajectory identity, and image content.

Responses are cached atomically under `model_config.cache_dir`. The cache key
contains provider, model, task, prompt version, target images, and selected
context. Each valid entry stores both the raw model response and parsed curve.
Rerunning the same command resumes by skipping completed cache entries.
Responses missing any frame ID, containing duplicate IDs, non-numeric or
non-finite values, or values outside 0–100 are rejected and not cached.

Build source-specific banks:

```bash
uv run python scripts/prepare_gvl_context_banks.py \
  --source teleop --num-shots 5 \
  --output ../../artifacts/gvl_context_banks/teleop.json
uv run python scripts/prepare_gvl_context_banks.py \
  --source rollout --num-shots 5 \
  --output ../../artifacts/gvl_context_banks/rollout.json
uv run python scripts/prepare_gvl_context_banks.py \
  --source human --num-shots 5 \
  --output ../../artifacts/gvl_context_banks/human.json
```

Add `--dry-run` to inspect selected rewards and durations without writing a
bank.

Set the benchmark once:

```bash
export ROBOMETER_PROCESSED_DATASETS_PATH='/path/to/counterfactual_study/artifacts/robometer_processed_rollout_sampling_v2'
export GVL_BENCHMARK='/path/to/counterfactual_study/artifacts/robometer_hf/counterfactual_eval_benchmark_left_current/eval_benchmark'
```

Dry-run the full zero-shot benchmark to count cache hits and expected API
calls. No API key is needed and no API call is made. The output
`gvl_dry_run_summary.json` also reports context task, demonstration, and image
counts:

```bash
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=gvl max_frames=30 \
  model_config.provider=openai model_config.model_name=gpt-4o \
  model_config.num_shots=0 model_config.dry_run=true \
  custom_eval.eval_types='[reward_alignment]' \
  custom_eval.reward_alignment="[$GVL_BENCHMARK]" \
  custom_eval.reward_alignment_max_trajectories=null \
  custom_eval.use_frame_steps=false custom_eval.subsample_n_frames=30 \
  output_dir=baseline_eval_output/gvl_openai_zero_dry_run
```

OpenAI smoke test (one trajectory):

```bash
export OPENAI_API_KEY='paste-key-here'
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=gvl max_frames=30 \
  model_config.provider=openai model_config.model_name=gpt-4o \
  model_config.num_shots=0 \
  custom_eval.eval_types='[reward_alignment]' \
  custom_eval.reward_alignment="[$GVL_BENCHMARK]" \
  custom_eval.reward_alignment_max_trajectories=1 \
  custom_eval.use_frame_steps=false custom_eval.subsample_n_frames=30 \
  output_dir=baseline_eval_output/gvl_openai_zero_smoke
```

OpenAI zero-shot full run:

```bash
export OPENAI_API_KEY='paste-key-here'
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=gvl max_frames=30 \
  model_config.provider=openai model_config.model_name=gpt-4o \
  model_config.num_shots=0 \
  custom_eval.eval_types='[reward_alignment]' \
  custom_eval.reward_alignment="[$GVL_BENCHMARK]" \
  custom_eval.reward_alignment_max_trajectories=null \
  custom_eval.use_frame_steps=false custom_eval.subsample_n_frames=30 \
  output_dir=baseline_eval_output/gvl_openai_zero_full
```

Source-specific five-shot full run (set `SOURCE` to `teleop`, `rollout`, or
`human`; use `num_shots=1..5` for scaling):

```bash
export OPENAI_API_KEY='paste-key-here'
SOURCE=teleop
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=gvl max_frames=30 \
  model_config.provider=openai model_config.model_name=gpt-4o \
  model_config.context_bank_path="$PWD/../../artifacts/gvl_context_banks/$SOURCE.json" \
  model_config.context_source=$SOURCE model_config.num_shots=5 \
  model_config.label_mode=terminal_reward \
  custom_eval.eval_types='[reward_alignment]' \
  custom_eval.reward_alignment="[$GVL_BENCHMARK]" \
  custom_eval.reward_alignment_max_trajectories=null \
  custom_eval.use_frame_steps=false custom_eval.subsample_n_frames=30 \
  output_dir="baseline_eval_output/gvl_openai_${SOURCE}_5shot"
```

Append `model_config.dry_run=true` to the few-shot command to inspect expected
calls and context size without setting `OPENAI_API_KEY`. Override the default
model with `model_config.model_name=<openai-model>`.

Normal runs write the existing reward-alignment result JSON and metrics. The
paper metrics helper can consume that result:

```bash
uv run python robometer/evals/counterfactual_paper_metrics.py \
  --results baseline_eval_output/gvl_openai_zero_full/reward_alignment/counterfactual_eval_benchmark_results.json \
  --output baseline_eval_output/gvl_openai_zero_full/paper_metrics.json
```

