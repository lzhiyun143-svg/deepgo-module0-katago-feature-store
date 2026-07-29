#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
KATAGO="${KATAGO:-/root/katago_bin/katago}"
MODEL="${MODEL:-/root/katago_bin/model.bin.gz}"
BASE_CONFIG="${BASE_CONFIG:-${CONFIG:-/root/katago_bin/analysis.cfg}}"
SGF_DIR="${SGF_DIR:-/root/deepgo/data/kgs}"
MAX_VISITS="${MAX_VISITS:-25}"
NN_MAX_BATCH_SIZE="${NN_MAX_BATCH_SIZE:-128}"
NUM_ANALYSIS_THREADS="${NUM_ANALYSIS_THREADS:-8}"
NUM_SEARCH_THREADS_PER_ANALYSIS_THREAD="${NUM_SEARCH_THREADS_PER_ANALYSIS_THREAD:-4}"
MAX_BOARD_X_SIZE_FOR_NN_BUFFER="${MAX_BOARD_X_SIZE_FOR_NN_BUFFER:-19}"
MAX_BOARD_Y_SIZE_FOR_NN_BUFFER="${MAX_BOARD_Y_SIZE_FOR_NN_BUFFER:-19}"
REQUIRE_MAX_BOARD_SIZE="${REQUIRE_MAX_BOARD_SIZE:-true}"
MAX_INFLIGHT_POSITIONS="${MAX_INFLIGHT_POSITIONS:-512}"
MAX_POSITIONS_PER_QUERY="${MAX_POSITIONS_PER_QUERY:-64}"
NN_CACHE_SIZE_POWER_OF_TWO="${NN_CACHE_SIZE_POWER_OF_TWO:-22}"
NN_MUTEX_POOL_SIZE_POWER_OF_TWO="${NN_MUTEX_POOL_SIZE_POWER_OF_TWO:-17}"
ANALYSIS_LOGS_ROOT="${ANALYSIS_LOGS_ROOT:-$REPO_ROOT/analysis_logs}"
RUN_ROOT="${RUN_ROOT:-$ANALYSIS_LOGS_ROOT/full_module0_training_$(date +%Y%m%d_%H%M%S)}"
STORE_ROOT="${STORE_ROOT:-$RUN_ROOT/feature_store/v1.0.0}"
REQUESTS_OUT="${REQUESTS_OUT:-$RUN_ROOT/requests.jsonl}"
RESPONSES_OUT="${RESPONSES_OUT:-$RUN_ROOT/raw.responses.jsonl}"
KATAGO_LOG="${KATAGO_LOG:-$RUN_ROOT/katago.analysis.log}"
ANALYSIS_CFG="$RUN_ROOT/analysis.cfg"

mkdir -p "$RUN_ROOT" "$STORE_ROOT"

set_cfg() {
  local file=$1
  local key=$2
  local value=$3
  if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$file"; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]*=.*|${key} = ${value}|" "$file"
  else
    echo "${key} = ${value}" >> "$file"
  fi
}

resolve_cuda_device_ids() {
  if [ -n "${CUDA_DEVICE_IDS:-}" ]; then
    echo "$CUDA_DEVICE_IDS"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [ -n "$detected" ]; then
      echo "$detected"
      return 0
    fi
  fi

  if [ -n "${CUDA_DEVICE_TO_USE:-}" ]; then
    echo "$CUDA_DEVICE_TO_USE"
    return 0
  fi

  echo "0"
}

apply_cuda_settings() {
  local file=$1
  local device_ids=$2
  local threads=${3:-1}

  local IFS=','
  read -r -a devices <<< "$device_ids"
  local device_count=${#devices[@]}

  if [ "$device_count" -gt 1 ]; then
    set_cfg "$file" numNNServerThreadsPerModel "$device_count"
    for i in "${!devices[@]}"; do
      set_cfg "$file" "cudaDeviceToUseThread${i}" "${devices[$i]}"
    done
  else
    set_cfg "$file" numNNServerThreadsPerModel "${threads:-1}"
    set_cfg "$file" cudaDeviceToUse "${devices[0]}"
  fi

  if [ -n "${CUDA_USE_FP16:-}" ]; then
    set_cfg "$file" cudaUseFP16 "$CUDA_USE_FP16"
  fi
}

stop_tree() {
  local pid=$1
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: base config not found: $BASE_CONFIG"
  exit 1
fi

cp "$BASE_CONFIG" "$ANALYSIS_CFG"
set_cfg "$ANALYSIS_CFG" logDir "$RUN_ROOT/katago_engine_logs"
set_cfg "$ANALYSIS_CFG" maxVisits "$MAX_VISITS"
set_cfg "$ANALYSIS_CFG" nnMaxBatchSize "$NN_MAX_BATCH_SIZE"
set_cfg "$ANALYSIS_CFG" numAnalysisThreads "$NUM_ANALYSIS_THREADS"
set_cfg "$ANALYSIS_CFG" numSearchThreadsPerAnalysisThread "$NUM_SEARCH_THREADS_PER_ANALYSIS_THREAD"
set_cfg "$ANALYSIS_CFG" maxBoardXSizeForNNBuffer "$MAX_BOARD_X_SIZE_FOR_NN_BUFFER"
set_cfg "$ANALYSIS_CFG" maxBoardYSizeForNNBuffer "$MAX_BOARD_Y_SIZE_FOR_NN_BUFFER"
set_cfg "$ANALYSIS_CFG" requireMaxBoardSize "$REQUIRE_MAX_BOARD_SIZE"
set_cfg "$ANALYSIS_CFG" reportAnalysisWinratesAs SIDETOMOVE
set_cfg "$ANALYSIS_CFG" nnCacheSizePowerOfTwo "$NN_CACHE_SIZE_POWER_OF_TWO"
set_cfg "$ANALYSIS_CFG" nnMutexPoolSizePowerOfTwo "$NN_MUTEX_POOL_SIZE_POWER_OF_TWO"
set_cfg "$ANALYSIS_CFG" nnRandomize true

local_cuda_device_ids="$(resolve_cuda_device_ids)"
local_cuda_threads="${CUDA_NUM_NN_SERVER_THREADS_PER_MODEL:-1}"
apply_cuda_settings "$ANALYSIS_CFG" "$local_cuda_device_ids" "$local_cuda_threads"

echo "==> Using run root: $RUN_ROOT"
echo "==> Using store root: $STORE_ROOT"
echo "==> Using SGF dir: $SGF_DIR"
echo "==> Using analysis config: $ANALYSIS_CFG"
echo "==> maxVisits=$MAX_VISITS nnMaxBatchSize=$NN_MAX_BATCH_SIZE numAnalysisThreads=$NUM_ANALYSIS_THREADS numSearchThreadsPerAnalysisThread=$NUM_SEARCH_THREADS_PER_ANALYSIS_THREAD"
echo "==> boardBuffer=${MAX_BOARD_X_SIZE_FOR_NN_BUFFER}x${MAX_BOARD_Y_SIZE_FOR_NN_BUFFER} maxInFlightPositions=$MAX_INFLIGHT_POSITIONS maxPositionsPerQuery=$MAX_POSITIONS_PER_QUERY nnCacheSizePowerOfTwo=$NN_CACHE_SIZE_POWER_OF_TWO"

echo "[1/6] Initializing feature store"
"$PYTHON_BIN" -m module0_katago_store.cli init-store --root "$STORE_ROOT" --max-visits "$MAX_VISITS"

echo "[2/6] Building manifests"
"$PYTHON_BIN" -m module0_katago_store.cli build-manifest --sgf-dir "$SGF_DIR" --store-root "$STORE_ROOT"

echo "[3/6] Generating KataGo requests"
"$PYTHON_BIN" -m module0_katago_store.cli make-requests --store-root "$STORE_ROOT" --out "$REQUESTS_OUT"

echo "[4/6] Running KataGo analysis"
"$PYTHON_BIN" -m module0_katago_store.cli run-analysis \
  --katago-bin "$KATAGO" \
  --model "$MODEL" \
  --config "$ANALYSIS_CFG" \
  --requests "$REQUESTS_OUT" \
  --out "$RESPONSES_OUT" \
  --log "$KATAGO_LOG" \
  --max-inflight-positions "$MAX_INFLIGHT_POSITIONS" \
  --max-positions-per-query "$MAX_POSITIONS_PER_QUERY"

echo "[5/6] Normalizing responses"
"$PYTHON_BIN" -m module0_katago_store.cli normalize --store-root "$STORE_ROOT" --responses "$RESPONSES_OUT"

echo "[6/6] Running QA"
"$PYTHON_BIN" -m module0_katago_store.cli qa --store-root "$STORE_ROOT"

echo "Completed full Module0 run."
echo "Outputs written under: $RUN_ROOT"
