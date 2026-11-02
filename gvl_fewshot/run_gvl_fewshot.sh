#!/usr/bin/env bash
# Few-shot GVL control runs: zero-shot + {teleop, rollout, human, hifi} x 5 demonstrations,
# seeds 42/43/44. Uses the robometer fork (branch counterfactual-study).
#
#   export OPENAI_API_KEY=sk-...
#   export ROBOMETER_REPO=/path/to/robometer          # fork checkout
#   export COUNTERFACTUAL_DATA_ROOT=/path/to/train_data  # HF train dataset root (videos)
#   export BENCHMARK=/path/to/processed/eval_benchmark   # preprocessed benchmark dataset
#   export OUT=/path/to/gvl_out
set -euo pipefail
: "${OPENAI_API_KEY:?}" "${ROBOMETER_REPO:?}" "${COUNTERFACTUAL_DATA_ROOT:?}" "${BENCHMARK:?}" "${OUT:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROBOMETER_REPO"
python scripts/run_gvl_multiseed.py \
  --seeds 42 43 44 --no-include-existing-seed42 \
  --provider openai --model-name gpt-4o \
  --benchmark "$BENCHMARK" \
  --context-bank-dir "$HERE/context_banks" \
  --processed-datasets-path "${ROBOMETER_PROCESSED_DATASETS_PATH:-$OUT/processed}" \
  --output-root "$OUT"
