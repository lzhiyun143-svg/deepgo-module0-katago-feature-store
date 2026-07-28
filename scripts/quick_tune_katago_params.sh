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

KATAGO="${KATAGO:-/root/katago_bin/katago}"
MODEL="${MODEL:-/root/katago_bin/model.bin.gz}"
BASE_CONFIG="${BASE_CONFIG:-${CONFIG:-/root/katago_bin/analysis.cfg}}"
SECONDS_PER_CASE="${SECONDS_PER_CASE:-300}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/analysis_logs/quick_param_tuning_$(date +%Y%m%d_%H%M%S)}"

REQUESTS="${REQUESTS:-${REQUESTS_FILE:-}}"
if [ -z "$REQUESTS" ]; then
  if [ -f "$REPO_ROOT/example_store/requests.jsonl" ]; then
    REQUESTS="$REPO_ROOT/example_store/requests.jsonl"
  elif [ -f "$REPO_ROOT/requests.jsonl" ]; then
    REQUESTS="$REPO_ROOT/requests.jsonl"
  else
    echo "ERROR: cannot find benchmark requests.jsonl; set REQUESTS in .env"
    exit 1
  fi
fi

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "case,maxVisits,nnMaxBatchSize,numAnalysisThreads,numSearchThreads,run_seconds,response_lines,error_lines,gpu_util_avg,gpu_util_max,mem_used_avg_mb,mem_used_max_mb,lines_per_second" > "$SUMMARY"

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

run_case() {
  local case_name=$1
  local max_visits=$2
  local batch_size=$3
  local analysis_threads=$4
  local search_threads=$5

  local case_dir="$OUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  local cfg="$case_dir/analysis.cfg"
  local responses="$case_dir/responses.jsonl"
  local log="$case_dir/katago.log"
  local gpu_csv="$case_dir/gpu.csv"

  cp "$BASE_CONFIG" "$cfg"
  set_cfg "$cfg" maxVisits "$max_visits"
  set_cfg "$cfg" nnMaxBatchSize "$batch_size"
  set_cfg "$cfg" numAnalysisThreads "$analysis_threads"
  set_cfg "$cfg" numSearchThreadsPerAnalysisThread "$search_threads"
  set_cfg "$cfg" reportAnalysisWinratesAs SIDETOMOVE
  # Make the generated configs more aggressive for a strong GPU/CPU host.
  set_cfg "$cfg" nnCacheSizePowerOfTwo 24
  set_cfg "$cfg" nnMutexPoolSizePowerOfTwo 18
  set_cfg "$cfg" nnRandomize true

  local cuda_device_ids
  cuda_device_ids=$(resolve_cuda_device_ids)
  local cuda_threads
  cuda_threads=${CUDA_NUM_NN_SERVER_THREADS_PER_MODEL:-1}
  apply_cuda_settings "$cfg" "$cuda_device_ids" "$cuda_threads"

  echo
  echo "====== Running $case_name for ${SECONDS_PER_CASE}s ======"
  echo "maxVisits=$max_visits nnMaxBatchSize=$batch_size numAnalysisThreads=$analysis_threads numSearchThreadsPerAnalysisThread=$search_threads"
  echo "output: $case_dir"

  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 2 > "$gpu_csv" &
  local mon_pid=$!

  local start_ts
  start_ts=$(date +%s)

  "$KATAGO" analysis -model "$MODEL" -config "$cfg" < "$REQUESTS" > "$responses" 2> "$log" &
  local katago_pid=$!

  if [ "$SECONDS_PER_CASE" -gt 0 ]; then
    sleep "$SECONDS_PER_CASE"
    stop_tree "$katago_pid"
  else
    wait "$katago_pid"
  fi

  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true

  local end_ts
  end_ts=$(date +%s)
  local elapsed=$((end_ts - start_ts))

  local response_lines error_lines
  response_lines=$(wc -l < "$responses" 2>/dev/null | tr -d ' ' || echo 0)
  error_lines=$(grep -c '"error"' "$responses" 2>/dev/null || true)

  local gpu_avg gpu_max mem_avg mem_max
  gpu_avg=$(awk -F',' '{gsub(/ /,"",$2); if($2!=""){sum+=$2;n++}} END{if(n) printf "%.1f",sum/n; else print "0"}' "$gpu_csv")
  gpu_max=$(awk -F',' '{gsub(/ /,"",$2); if($2>max) max=$2} END{print max+0}' "$gpu_csv")
  mem_avg=$(awk -F',' '{gsub(/ /,"",$3); if($3!=""){sum+=$3;n++}} END{if(n) printf "%.0f",sum/n; else print "0"}' "$gpu_csv")
  mem_max=$(awk -F',' '{gsub(/ /,"",$3); if($3>max) max=$3} END{print max+0}' "$gpu_csv")
  local lps
  lps=$(awk -v lines="$response_lines" -v sec="$elapsed" 'BEGIN{if(sec>0) printf "%.2f",lines/sec; else print "0"}')

  echo "$case_name,$max_visits,$batch_size,$analysis_threads,$search_threads,$elapsed,$response_lines,$error_lines,$gpu_avg,$gpu_max,$mem_avg,$mem_max,$lps" >> "$SUMMARY"
  echo "Done $case_name: elapsed=${elapsed}s, lines=$response_lines, errors=$error_lines, gpu_avg=${gpu_avg}%, gpu_max=${gpu_max}%, mem_max=${mem_max}MB, lps=${lps}"
}

# Use a more aggressive profile for a strong GPU + 10-core CPU setup.
# These cases trade a bit more concurrency and batching for higher throughput.
run_case "v25_batch128_a8_s4" 25 128 8 4
run_case "v25_batch256_a8_s4" 25 256 8 4
run_case "v25_batch256_a10_s4" 25 256 10 4
run_case "v25_batch512_a8_s4" 25 512 8 4

echo
echo "====== Quick Summary ======"
if command -v column >/dev/null 2>&1; then
  column -s, -t "$SUMMARY"
else
  cat "$SUMMARY"
fi

echo
echo "Result file: $SUMMARY"
