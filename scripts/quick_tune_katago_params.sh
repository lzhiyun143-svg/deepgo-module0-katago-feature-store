#!/usr/bin/env bash
set -euo pipefail

KATAGO=${KATAGO:-/root/katago_bin/katago}
MODEL=${MODEL:-/root/katago_bin/model.bin.gz}
BASE_CONFIG=${BASE_CONFIG:-/root/katago_bin/analysis.cfg}
SECONDS_PER_CASE=${SECONDS_PER_CASE:-300}
OUT_ROOT=${OUT_ROOT:-/root/katago_benchmark/quick_param_tuning_$(date +%Y%m%d_%H%M%S)}

if [ -f /root/katago_benchmark/bench_100games/maxvisits_25/requests.jsonl ]; then
  REQUESTS=/root/katago_benchmark/bench_100games/maxvisits_25/requests.jsonl
elif [ -f /root/katago_feature_store/requests_sample.jsonl ]; then
  REQUESTS=/root/katago_feature_store/requests_sample.jsonl
else
  echo "ERROR: cannot find benchmark requests.jsonl"
  exit 1
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

  sleep "$SECONDS_PER_CASE"
  stop_tree "$katago_pid"

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

run_case "v25_batch64_a2_s16_baseline" 25 64 2 16
run_case "v25_batch128_a4_s8" 25 128 4 8
run_case "v25_batch256_a8_s4" 25 256 8 4
run_case "v25_batch128_a8_s4" 25 128 8 4

echo
echo "====== Quick Summary ======"
if command -v column >/dev/null 2>&1; then
  column -s, -t "$SUMMARY"
else
  cat "$SUMMARY"
fi

echo
echo "Result file: $SUMMARY"
