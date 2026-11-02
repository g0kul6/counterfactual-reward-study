# Robometer LoRA training and GVL baseline details

This document records the settings used for the original Robometer
counterfactual-data experiments and the GVL baseline. It covers the control,
single-source scaling, mixed-source, and HiFi-versus-LoFi experiments.
Preference-only and successful-progress-only ablations are intentionally
excluded.

## 1. Robometer model and trainable parameters

- Starting checkpoint: `robometer/Robometer-4B`.
- Underlying vision-language backbone recorded by the resolved run
  configuration: `Qwen/Qwen3-VL-4B-Instruct`.
- Trainer: `RBMHeadsTrainer`.
- Precision: bfloat16 (`bf16=true`, `fp16=false`).
- Quantization: disabled.
- Unsloth acceleration: enabled.
- Gradient checkpointing: enabled.
- Vision encoder: frozen.
- Language model: adapted with LoRA; the full language-model weights are not
  directly fine-tuned.
- All three Robometer heads are trained:
  - preference/ranking head;
  - per-frame progress head;
  - binary success head.
- Multi-image input is enabled, with one `<|prog_token|>` after each sampled
  frame.
- Frame pooling is `mean`. Saved YAML files contain
  `average_temporal_patches=true`, while the live experiment-config dump
  records `false`; this inherited field should therefore not be treated as a
  reliable paper hyperparameter.

### LoRA configuration

- Rank: `r=16`.
- Scaling: `lora_alpha=32`.
- Dropout: `0.05`.
- Bias: `none`.
- Language-model target modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  and `down_proj`.
- No LoRA adapters are attached to the vision encoder.

### Prediction-head architecture

Each head uses the backbone hidden representation and has the form:

```text
Linear(hidden_dim, hidden_dim/2)
→ LayerNorm
→ GELU
→ Dropout(0.1)
→ Linear(hidden_dim/2, output_dim)
```

The preference and success heads output one logit. The progress head outputs
10 logits because progress is trained as a 10-bin discrete prediction.

## 2. Shared optimization hyperparameters

- Optimizer: Hugging Face Trainer's default AdamW.
- Adam defaults were not overridden: β₁=0.9, β₂=0.999, ε=1e-8.
- Learning rate: `1e-4`.
- Weight decay: `0.01`.
- Scheduler: cosine decay.
- Warm-up: `warmup_ratio=0.1`; explicit `warmup_steps=0`, so the ratio controls
  warm-up.
- Maximum gradient norm: `1.0`.
- Per-device training batch size: `2`.
- Gradient accumulation: `1`.
- The rerun launcher starts each training process with one visible GPU, so the
  effective optimizer batch is two preference pairs per training process.
  The saved `num_gpus=2` field is inherited configuration metadata and is not
  the runtime world size for these single-process jobs.
- Epoch limit: disabled (`num_train_epochs=-1`).
- Training duration is controlled by `max_steps`.
- Seeds: 42, 43, and 44. Each seed is applied to both data sampling and model
  training.
- Maximum sequence length: 1024.
- Training-time evaluation: disabled.
- Logging: every optimizer step, W&B offline mode.
- Dataloader: four workers, pinned memory, persistent workers.

The step rule is:

```text
max_steps = 3 × number of selected trajectories
```

With batch size two, this gives approximately six effective passes over each
selected trajectory set while keeping exposure comparable across dataset
sizes.

## 3. Frames, targets, sampling, and losses

### Frame processing

- At most 32 frames are retained by preprocessing.
- At most 8 frames are passed to Robometer for a sampled training trajectory.
- Trajectories with fewer than 5 frames are excluded.
- Frames are supplied as multiple images rather than an encoded video.
- The progress convention is `absolute_wrt_total_frames`.
- Dataset order is shuffled.

### Partial-credit supervision

For a trajectory with scalar partial credit \(p<1\):

- the final original frame receives progress target \(p\);
- all other temporary target entries are zero;
- `predict_last_frame_partial_progress=true` creates a mask so progress loss is
  computed only where the final original frame appears.

Therefore, intermediate frames of a partial-credit trajectory are **not**
trained as a linear curve multiplied by \(p\). The scalar partial credit gives
endpoint supervision. Dense progress supervision comes from successful and
synthetically transformed samples such as rewind, reverse-progress, and
different-task constructions.

### Sampling

- Dataset class: `strategy_first`.
- Sample-type ratio: `[1.0, 0.0, 0.0]`.
- Dataset preference ratio: `1.0`.
- Preference strategies are sampled equally:
  rewind, same-task suboptimal, different task, and reverse progress.
- Partial-credit trajectories can be paired only when their rewards differ by
  at least `0.2`.
- Alternative task text is sampled from the same source with probability
  `0.5`.

The `[1,0,0]` ratio does **not** make this a preference-only model. It means
the loader emits preference pairs. Because
`training.predict_pref_progress=true` and all three heads are enabled, one
preference batch contributes:

```text
total loss =
    preference BCE
  + chosen-trajectory progress loss
  + chosen-trajectory success loss
```

There are no explicit coefficients between these three terms; they are added
with unit weight.

### Individual objectives

- Preference: binary cross-entropy on the Bradley–Terry preference logit.
- Progress: cross-entropy over 10 discrete progress bins.
- Success: binary cross-entropy on success logits using valid-frame masks. The
  implementation computes minority-class weights, but divides those weights
  back out before the final mean; consequently the optimized value is
  effectively an unweighted masked BCE in the recorded code.
- Success thresholds: `min_success=0.5`, `max_success=1.0`, with the
  dataset-specific cutoff file
  `robometer/data/dataset_success_cutoff.txt`.
- The counterfactual source names are absent from that cutoff file, so these
  experiments fall back to `max_success=1.0`.
- Failure/suboptimal trajectories are treated as success label zero.

## 4. Experiment matrix

Every number below is **per task**. There are five tasks, so 5 trajectories
per task means 25 total trajectories.

### Control

- Teleop-5: 25 trajectories, 75 steps.
- Rollout-5: 25 trajectories, 75 steps.
- Human-5: 25 trajectories, 75 steps.
- Sim HiFi-5: 25 trajectories, 75 steps.

Each condition is trained independently for seeds 42, 43, and 44.

### Single-source scaling

Rollout:

- 5/task: 25 trajectories, 75 steps.
- 10/task: 50 trajectories, 150 steps.
- 15/task: 75 trajectories, 225 steps.
- 20/task: 100 trajectories, 300 steps.

Human:

- 5/task: 25 trajectories, 75 steps.
- 10/task: 50 trajectories, 150 steps.
- 15/task: 75 trajectories, 225 steps.
- 20/task: 100 trajectories, 300 steps.

Sim HiFi:

- 5/task: 25 trajectories, 75 steps.
- 10/task: 50 trajectories, 150 steps.
- 15/task: 75 trajectories, 225 steps.
- 20/task: 100 trajectories, 300 steps.
- 25/task: 125 trajectories, 375 steps.
- 30/task: 150 trajectories, 450 steps.
- 50/task: 250 trajectories, 750 steps.

The smaller source sets are nested prefixes of the larger sets, so scale
comparisons do not replace all examples when the data count changes.

### HiFi-versus-LoFi simulation ablation

The configured paired fidelity levels are:

- HiFi-5 versus LoFi-5: 25 trajectories each, 75 steps.
- HiFi-10 versus LoFi-10: 50 trajectories each, 150 steps.
- HiFi-20 versus LoFi-20: 100 trajectories each, 300 steps.
- HiFi-50 versus LoFi-50: 250 trajectories each, 750 steps.

LoFi trajectories use the same task/episode keys selected for HiFi. This is a
paired-fidelity comparison rather than two independently sampled trajectory
sets. All optimization settings are identical between HiFi and LoFi.

The aggregate results file records completed three-seed LoFi runs for 5, 10,
and 50 per task. LoFi-20 was configured but intentionally cancelled and has no
completed seeds; it should not be reported as a completed experiment.

### Mixed-source conditions

Control mix:

- Teleop-5 + Rollout-5 + Human-5 + HiFi-5 per task.
- 20 trajectories per task, 100 total.
- 300 steps.

Scaled mix:

- Teleop-5 + Rollout-10 + Human-20 + HiFi-50 per task.
- 85 trajectories per task, 425 total.
- 1,275 steps.

Proportional mix:

- Teleop-5 + Rollout-10 + Human-20 + HiFi-20 per task.
- 55 trajectories per task, 275 total.
- 825 steps.

Control mix and scaled mix each use one pre-merged Hugging Face dataset.
Proportional mix instead supplies four separate source datasets to
`StrategyFirstDataset`; because `data_source_weights=null`, those four source
datasets are sampled uniformly rather than in proportion to their stored
trajectory counts.

## 5. Dataset construction

- Camera view: left camera.
- Five tabletop tasks are used.
- Training and the 250-trajectory held-out benchmark have a checked zero-ID
  overlap.
- Teleop-5 contains the five available selected teleoperation trajectories per
  task.
- Rollout, human, and HiFi subsets are selected to cover reward levels and
  short/middle/long durations.
- Scaling subsets are nested: a smaller condition is retained as the core of
  the next larger condition.
- LoFi subsets are paired exactly to the selected HiFi episode keys.
- Endpoint clipping uses the annotated `raw_start_frame` and `end_frame`.

## 6. Saving and evaluation

- Intermediate Trainer checkpoints are disabled (`save_strategy=no`).
- A final model and processor are saved.
- For the original runs, saving produces a self-contained full model snapshot,
  PEFT adapter files, `custom_heads.safetensors`, and PEFT target metadata.
  Thus the experiment trains LoRA adapters, but the default final artifact is
  larger than a LoRA-only checkpoint.
- Reward-alignment evaluation uses all 250 held-out trajectories.
- Evaluation uses whole trajectories, uniformly subsampled to 30 frames.
- Evaluation batch size is attempted at 4, then retried at 2 if necessary.
- Paper metrics use 10,000 bootstrap samples.

## 7. GVL baseline: model and protocol

GVL (Generative Value Learning) is an API-based vision-language baseline, not
a trained reward-model LoRA.
In the reported study it is used only for the matched-data control comparison:
zero-shot and four 5-shot source conditions. The GVL scaling condition lists
in the launcher are unused scaffolding; scaling, simulation fidelity, mixing,
and downstream RL results are Robometer-only.

- Provider used in the experiment launcher: OpenAI.
- Model: `gpt-4o`.
- Request type: OpenAI chat-completions vision request.
- Message structure: one multimodal user message; there is no separate system
  prompt.
- Image detail: `low`.
- Maximum response tokens: 4096.
- Streaming: enabled.
- Temperature, top-p, and a generation seed are not explicitly set, so API
  defaults apply.
- Prompt version: `gvl-trajectory-average-reward-v2`.
- `offset=0.5` is retained in configuration but is not used by the frame
  extraction implementation.
- Target frames: at most 30 uniformly sampled frames, retaining the temporal
  endpoints.
- Context demonstration frames: at most 30 uniformly sampled frames per
  demonstration.
- Frames are converted to JPEG and sent as base64 images.
- Target and demonstration frames are deterministically shuffled using the
  run seed and image-content hashes.
- Seeds 42, 43, and 44 therefore change frame presentation order; they do not
  train or change model weights.

### GVL context conditions

Control conditions:

- zero-shot: no demonstrations;
- Teleop: 5 same-task demonstrations;
- Rollout: 5 same-task demonstrations;
- Human: 5 same-task demonstrations;
- Sim HiFi: 5 same-task demonstrations.

The launcher also defines these requested scaling conditions:

- Rollout: 5, 10, 15, and 20 shots.
- Human: 5, 10, 15, and 20 shots.
- Sim HiFi: 5, 10, 15, 20, 25, and 30 shots.

These GVL scaling conditions were **not executed** and are not reported in the
paper. The completed GVL artifact consists of 5 control conditions × 3 frame
shuffle seeds = 15 runs.

Every completed run evaluates all 250 held-out trajectories: five tasks with
50 trajectories per task, split between 25 π₀.₅ and 25 MolmoAct2 policy
rollouts. The whole trajectory is evaluated once rather than as growing frame
prefixes.

The context banks contain nested per-task orders:

- Teleop bank: up to 5 demonstrations per task.
- Rollout bank: up to 20 per task.
- Human bank: up to 20 per task.
- HiFi bank: up to 30 per task.

Each context example has one whole-trajectory
**average-behavior partial-credit** label. The prompt explicitly says that this
is neither a per-frame label nor a terminal-state label.

### Exact target prompt template

The target part of the prompt is constructed as follows:

```text
GVL prompt version: gvl-trajectory-average-reward-v2.
Return only the requested JSON array.

[Optional same-task labeled demonstrations appear here.]

You are an expert roboticist tasked to predict task completion percentages
for frames of a robot for the task of {task_description}. The task completion
percentages are between 0 and 100, where 100 corresponds to full task
completion. Note that these frames are in random order, so please pay
attention to the individual frames.

Initial robot scene:
This frame: [INITIAL_FRAME]

shows the initial robot scene, where the task completion percentage is 0.

Now, for the task of *{task_description}*, output the task completion
percentage for the following frames that are presented in random order.
Format your response in JSON as follows, making sure to include all frames:

[
  {
    "frame_number": i,
    "frame_description": "...",
    "task_completion_percentage": 0-100
  }
]

Frame 1: [SHUFFLED_FRAME]
Frame 2: [SHUFFLED_FRAME]
...
```

For few-shot GVL, the preceding context section says:

```text
Here are {N} same-task demonstration trajectories from the {source} source.
Each demonstration has exactly one label: the human-assigned average-behavior
partial-credit reward for the whole trajectory. It summarizes behavior across
the trajectory and is not a terminal-state reward. The label is not a
per-frame progress label. Frames are shuffled.

Demonstration {j} for task *{task_description}*:
[SHUFFLED DEMONSTRATION IMAGES]
Whole-trajectory average-behavior partial-credit label for demonstration {j}:
{100 × reward}%. Do not interpret this as a label for any individual image.
```

### Output parsing and reward curve

- The response must be a JSON array with exactly one unique integer
  `frame_number` for every requested frame.
- Every `task_completion_percentage` must be finite and in [0,100].
- Invalid or incomplete responses are rejected and are not cached.
- Valid percentages are divided by 100 and reordered from shuffled
  presentation order back into chronological frame order.
- This reordered sequence is the predicted progress/reward curve.
- The initial frame is additionally shown as a 0%-completion reference.

### Reliability and caching

- Up to five retries are configured after the first request, giving at most
  six attempts.
- Retry delays use exponential backoff from 1 second, capped at 60 seconds.
- HTTP 429 and server-side 5xx responses are retried.
- Successful responses are cached atomically.
- Cache keys include provider, model, task, prompt version, label mode,
  context source/count, selected context trajectories, and hashes of all
  prompt images and text.

## 8. Source-of-truth files

- Shared Robometer config:
  `reward_model/robometer/robometer/configs/counterfactual_lora.yaml`
- Saved resolved run configs:
  `artifacts/rerun_20260922_latest/lora/*/config.yaml`
- Experiment matrix and step rule:
  `reward_model/robometer/scripts/run_robometer_rerun.py`
- Dataset selection and mix construction:
  `reward_model/robometer/scripts/prepare_counterfactual_experiments.py`
- Partial-credit endpoint target:
  `reward_model/robometer/robometer/data/datasets/helpers.py`
- Training losses:
  `reward_model/robometer/robometer/trainers/rbm_heads_trainer.py`
- Final checkpoint saving:
  `reward_model/robometer/robometer/utils/save.py`
- GVL implementation and prompt:
  `reward_model/robometer/robometer/evals/baselines/gvl.py`
- GVL model/condition launcher:
  `reward_model/robometer/scripts/run_gvl_multiseed.py`
- GVL context selection:
  `reward_model/robometer/scripts/prepare_gvl_context_banks.py`
