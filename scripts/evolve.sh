#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./evolve.sh --dataset <DATASET> --run_name <RUN_NAME> [--batch_size <N>] [--start <STEP>] [--train_steps <N>] [--merge_policy <POLICY>] [--timeout <SECONDS>]

Supported Datasets:
  HLE
  XBENCH-deepsearch, XBENCH-scienceqa, XBENCH-all
  DEEPSEARCHQA, FINSEARCHCOMP

Examples:
  ./evolve.sh --dataset HLE --run_name hle_run --batch_size 4 --start 10 --train_steps 20 --merge_policy naive
  ./evolve.sh --dataset DEEPSEARCHQA --run_name dsqa_run --batch_size 5 --timeout 600
  ./evolve.sh --dataset FINSEARCHCOMP --run_name finsearchcomp_run --batch_size 5 --timeout 600
EOF
}

DATASET=""
RUN_NAME=""
BATCH_SIZE="5"
START_STEP="0"
DRY_RUN="0"
TRAIN_STEPS=""
MERGE_POLICY="naive"
TIMEOUT=""
ENABLE_LANGFUSE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"; shift 2;;
    --run_name)
      RUN_NAME="${2:-}"; shift 2;;
    --batch_size)
      BATCH_SIZE="${2:-}"; shift 2;;
    --start)
      START_STEP="${2:-}"; shift 2;;
    --train_steps)
      TRAIN_STEPS="${2:-}"; shift 2;;
    --merge_policy)
      MERGE_POLICY="${2:-}"; shift 2;;
    --timeout)
      TIMEOUT="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET" || -z "$RUN_NAME" ]]; then
  usage
  exit 2
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -le 0 ]]; then
  echo "Error: --batch_size must be a positive integer, got: $BATCH_SIZE" >&2
  exit 2
fi

if ! [[ "$START_STEP" =~ ^[0-9]+$ ]]; then
  echo "Error: --start must be a non-negative integer, got: $START_STEP" >&2
  exit 2
fi

if [[ -n "$TRAIN_STEPS" ]]; then
  if ! [[ "$TRAIN_STEPS" =~ ^[0-9]+$ ]] || [[ "$TRAIN_STEPS" -le 0 ]]; then
    echo "Error: --train_steps must be a positive integer, got: $TRAIN_STEPS" >&2
    exit 2
  fi
fi

if [[ -z "$MERGE_POLICY" ]]; then
  echo "Error: --merge_policy cannot be empty" >&2
  exit 2
fi

if [[ -n "$TIMEOUT" ]]; then
  if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT" -le 0 ]]; then
    echo "Error: --timeout must be a positive integer (seconds), got: $TIMEOUT" >&2
    exit 2
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found in PATH" >&2
  exit 127
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_SIZE="$(
python - <<'PY' "$REPO_ROOT" "$DATASET"
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
dataset = sys.argv[2]

def count_hle() -> int:
    p = repo_root / "dataset" / "HLE" / "hle_500.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("data", [])
    # mirror dataloader.load_hle_dataset: skip items without "question"
    return sum(1 for item in items if item.get("question") is not None)

def count_xbench(which: str) -> int:
    paths = []
    if which == "deepsearch":
        paths = [repo_root / "dataset" / "XBENCH" / "DeepSearch-2510.json"]
    elif which == "scienceqa":
        paths = [repo_root / "dataset" / "XBENCH" / "ScienceQA.json"]
    elif which == "all":
        paths = [
            repo_root / "dataset" / "XBENCH" / "DeepSearch-2510.json",
            repo_root / "dataset" / "XBENCH" / "ScienceQA.json",
        ]
    else:
        raise ValueError(which)
    total = 0
    for p in paths:
        total += len(json.loads(p.read_text(encoding="utf-8")))
    return total

def count_deepsearchqa() -> int:
    p = repo_root / "dataset" / "DEEPSEARCHQA" / "DSQA-full.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return len(data)
def count_finsearchcomp() -> int:
    p = repo_root / "dataset" / "FinSearchComp" / "t2_t3_questions.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return len(data)

if dataset == "HLE":
    n = count_hle()
elif dataset == "XBENCH-deepsearch":
    n = count_xbench("deepsearch")
elif dataset == "XBENCH-scienceqa":
    n = count_xbench("scienceqa")
elif dataset == "XBENCH-all":
    n = count_xbench("all")
elif dataset == "DEEPSEARCHQA":
    n = count_deepsearchqa()
elif dataset == "FINSEARCHCOMP":
    n = count_finsearchcomp()
else:
    raise SystemExit(f"Unknown dataset: {dataset}")

print(n)
PY
)"

if ! [[ "$DATASET_SIZE" =~ ^[0-9]+$ ]] || [[ "$DATASET_SIZE" -le 0 ]]; then
  echo "Error: computed dataset_size is invalid: $DATASET_SIZE" >&2
  exit 1
fi

TRAIN_STEPS_TOTAL="$(( (DATASET_SIZE + BATCH_SIZE - 1) / BATCH_SIZE ))"

TRAIN_END="$TRAIN_STEPS_TOTAL"
if [[ -n "$TRAIN_STEPS" ]]; then
  # Run at most TRAIN_STEPS steps starting from START_STEP.
  TRAIN_END="$(( START_STEP + TRAIN_STEPS ))"
  if [[ "$TRAIN_END" -gt "$TRAIN_STEPS_TOTAL" ]]; then
    TRAIN_END="$TRAIN_STEPS_TOTAL"
  fi
fi

if [[ "$START_STEP" -ge "$TRAIN_STEPS_TOTAL" ]]; then
  echo "Error: --start ($START_STEP) must be < train_steps_total ($TRAIN_STEPS_TOTAL)" >&2
  exit 2
fi

echo "dataset=$DATASET  dataset_size=$DATASET_SIZE  batch_size=$BATCH_SIZE  train_steps_total=$TRAIN_STEPS_TOTAL  start=$START_STEP  end=$TRAIN_END  run_name=$RUN_NAME  merge_policy=$MERGE_POLICY  timeout=${TIMEOUT:-none}"

EXTRA_ARGS=()


LOG_DIR="output/${RUN_NAME}/logs"
mkdir -p "$LOG_DIR"

for ((i=START_STEP; i<TRAIN_END; i++)); do
  echo "=== step ${i}/${TRAIN_END} (total=${TRAIN_STEPS_TOTAL}) ==="
  CMD=( uv run evolve.py --dataset "$DATASET" --run_name "$RUN_NAME" --batch_size "$BATCH_SIZE" --train_steps 1 --start "$i" --merge_policy "$MERGE_POLICY" )
  
  if [[ -n "$TIMEOUT" ]]; then
    CMD+=( --timeout "$TIMEOUT" )
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN: '; printf '%q ' "${CMD[@]}"; printf '\n'
    continue
  fi
  
  "${CMD[@]}"

done

echo "Done. Logs: ${LOG_DIR}"


