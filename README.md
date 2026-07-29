# Module0 KataGo Feature Store

This repository contains the Module0 implementation for building a shared
KataGo feature store from Go game records. It prepares data for downstream
human/user modeling modules, but it does not contain a model training script.

中文说明：本项目是 Module0 数据与特征生成部分。它把 KGS/SGF 棋谱解析成局面索引，
生成 KataGo analysis 请求，再把 KataGo 输出标准化为后续 Module1-4 可读取的特征库。
仓库只上传代码和小型辅助脚本，不上传大规模棋谱、KataGo 模型、KataGo 可执行文件或
完整 raw responses。

## Module0 在整体项目中的位置

```text
KGS / SGF game records
  -> Module0: build a unified KataGo feature store
  -> Module1: user profile / player representation
  -> Module2: human next-move prediction
  -> Module3: move quality and mistake analysis
  -> Module4: explanation, review, and training recommendation
```

Module0 是后续模块的数据底座。后续模块不需要重复解析 SGF 或直接读 KataGo 原始输出，
只需要通过 `FeatureStore` 读取稳定的局面特征。

## What It Does

1. Parse SGF files and extract game metadata and move sequences.
2. Build `dataset_manifest.jsonl` and `position_manifest.jsonl`.
3. Generate KataGo Parallel Analysis Engine JSONL requests.
4. Normalize KataGo JSONL responses into:
   - `normalized/scalars/<split>/part-00000.jsonl`
   - `normalized/candidates/<split>/part-00000.jsonl`
   - `normalized/spatial/<split>/shard-00000.npz`
   - `index/position_index.jsonl`
5. Provide `FeatureStore.get()`, `FeatureStore.get_many()`, and
   `FeatureStore.get_game()` for downstream modules.
6. Run QA checks and benchmark different KataGo parameters.

Typical features saved for each position include board state, side to move,
human next move, KataGo policy, winrate, score lead, ownership map, candidate
moves, and stable IDs.

## Server Paths Used In The Current Experiment

The current Aliyun A10 server was configured with these paths:

```text
Module0 code:        /root/module0_katago_feature_store
Python environment: /root/module0_env
KGS data:            /root/deepgo/data/kgs
Feature store:       /root/katago_feature_store/v1.0.0
KataGo executable:   /root/katago_bin/katago
KataGo model:        /root/katago_bin/model.bin.gz
KataGo config:       /root/katago_bin/analysis.cfg
```

The KGS data was downloaded from `https://dl.u-go.net/gamerecords/` using the
KGS archive list in the DeepGo data script. The current local GitHub repository
does not include those archives.

## Full Module0 Run Commands

On the server:

```bash
cd /root/module0_katago_feature_store
source /root/module0_env/bin/activate
```

Initialize the feature store:

```bash
python -m module0_katago_store.cli init-store \
  --root /root/katago_feature_store/v1.0.0 \
  --max-visits 25
```

Build manifests from KGS SGF files:

```bash
python -m module0_katago_store.cli build-manifest \
  --sgf-dir /root/deepgo/data/kgs \
  --store-root /root/katago_feature_store/v1.0.0
```

Generate KataGo requests:

```bash
python -m module0_katago_store.cli make-requests \
  --store-root /root/katago_feature_store/v1.0.0 \
  --out /root/katago_feature_store/requests.jsonl
```

Run KataGo analysis as a streaming process. KataGo loads the model once, then
receives all pending game queries through its open standard input. The final
response path remains `/root/katago_feature_store/raw.responses.jsonl`:

```bash
python -m module0_katago_store.cli run-analysis \
  --katago-bin /root/katago_bin/katago \
  --model /root/katago_bin/model.bin.gz \
  --config /root/katago_bin/analysis.cfg \
  --requests /root/katago_feature_store/requests.jsonl \
  --out /root/katago_feature_store/raw.responses.jsonl \
  --log /root/katago_feature_store/katago.analysis.log \
  --max-inflight-positions 512 \
  --max-positions-per-query 64
```

Normalize KataGo responses:

```bash
python -m module0_katago_store.cli normalize \
  --store-root /root/katago_feature_store/v1.0.0 \
  --responses /root/katago_feature_store/raw.responses.jsonl
```

Run QA:

```bash
python -m module0_katago_store.cli qa \
  --store-root /root/katago_feature_store/v1.0.0
```

## Current KataGo Setup

KataGo executable:

```text
KataGo v1.16.5, CUDA backend
Package used: katago-v1.16.5-cuda12.8-cudnn9.8.0-linux-x64.zip
Source: https://github.com/lightvector/KataGo/releases
```

KataGo model:

```text
kata1-b28c512nbt-s13255194368-d5935380940.bin.gz
Source: https://katagotraining.org/networks/
```

Current recommended analysis parameters, based on the quick tuning run on one
NVIDIA A10:

```text
maxVisits = 25
nnMaxBatchSize = 128
numAnalysisThreads = 8
numSearchThreadsPerAnalysisThread = 4
reportAnalysisWinratesAs = SIDETOMOVE
```

These values are written in `/root/katago_bin/analysis.cfg` on the server.
A copy of the tuned config is included in this repository:

```text
configs/analysis.a10.maxvisits25.cfg
```

## Local runtime and benchmark configuration

The repository now uses a repo-scoped `.env` file for local KataGo runtime and
benchmark settings. The tuning script at `scripts/quick_tune_katago_params.sh`
automatically loads `.env` from the repository root, so you no longer need to
hard-code `/root/...` paths into the script.

The following variables are supported:

```bash
KATAGO=/path/to/katago
MODEL=/path/to/model.bin.gz
CONFIG=/path/to/analysis.cfg
BASE_CONFIG=/path/to/analysis.cfg
REQUESTS=/path/to/requests.jsonl
OUT_ROOT=/path/to/analysis_logs/quick_param_tuning
SECONDS_PER_CASE=300

# Optional CUDA tuning
CUDA_DEVICE_IDS=0
CUDA_NUM_NN_SERVER_THREADS_PER_MODEL=1
CUDA_USE_FP16=auto

# Ensure the conda runtime can find cuDNN
LD_LIBRARY_PATH=/path/to/conda/env/lib:${LD_LIBRARY_PATH:-}
```

The script uses these values to:

- choose the KataGo binary, model, config, and request file
- create a per-run output directory under `analysis_logs/`
- write per-case configs with stronger GPU/CPU-oriented defaults for the
  benchmark sweep
- apply optional CUDA device selection and FP16 settings

A ready-to-copy example environment file is available at `.env.example`. After modifying that, you can copy it to `.env` in the repository to make the tuning script work without further edits.

## Quick Parameter Tuning Result

Quick tuning was run on 2026-07-28 with 100 benchmark games. Each parameter
group was tested for about 300 seconds.

| Case | maxVisits | nnMaxBatchSize | numAnalysisThreads | numSearchThreads | GPU avg | GPU max | Lines/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v25_batch64_a2_s16_baseline` | 25 | 64 | 2 | 16 | 43.0% | 48% | 11.91 |
| `v25_batch128_a4_s8` | 25 | 128 | 4 | 8 | 71.1% | 78% | 21.59 |
| `v25_batch256_a8_s4` | 25 | 256 | 8 | 4 | 90.0% | 98% | 28.72 |
| `v25_batch128_a8_s4` | 25 | 128 | 8 | 4 | 92.7% | 98% | 29.07 |

The recommended configuration is `v25_batch128_a8_s4`: it increased GPU average
utilization from 43.0% to 92.7% and improved throughput from 11.91 to 29.07
lines/s compared with the baseline.

Run the quick tuning script on the server:

```bash
nohup bash /root/katago_benchmark/quick_tune_katago_params.sh \
  > /root/katago_benchmark/quick_tune_katago_params.log 2>&1 &
```

Check progress:

```bash
tail -f /root/katago_benchmark/quick_tune_katago_params.log
```

Check the summary:

```bash
cat /root/katago_benchmark/quick_param_tuning_*/summary.csv
```

## Benchmark Commands

The code also supports full benchmark setup/report/compare commands through
the CLI:

```bash
python -m module0_katago_store.cli benchmark-setup --help
python -m module0_katago_store.cli benchmark-report --help
python -m module0_katago_store.cli benchmark-compare --help
```

Earlier benchmark settings:

```text
Games: 100
maxVisits tested: 1, 25, 50, 100
Normalized comparable positions: 16600
```

Main findings from the earlier complete benchmark:

```text
maxVisits=25 vs 100:
  Top-1 same rate: 0.80293
  winrate mean absolute delta: 0.004822
  scoreLead mean absolute delta: 0.400291

maxVisits=50 vs 100:
  Top-1 same rate: 0.832168
  winrate mean absolute delta: 0.003336
  scoreLead mean absolute delta: 0.280941
```

`maxVisits=1` is useful for fast policy-style extraction but does not reliably
produce search Top-1 candidate information.

## Code File Guide

Core package: `module0_katago_store/`

- `cli.py`: command-line entry point for `init-store`, `build-manifest`,
  `make-requests`, `normalize`, `qa`, and benchmark commands.
- `manifest.py`: scans SGF files and builds game-level and position-level
  manifests with stable IDs.
- `sgf.py`: lightweight SGF parser for board size, rules, komi, players,
  result, and moves.
- `coords.py`: Go coordinate conversion utilities, including pass moves and
  the skipped `I` column.
- `ids.py`: stable `game_id` and `position_id` helpers.
- `profile.py`: immutable KataGo analysis profile metadata.
- `requests.py`: converts position manifests into KataGo JSONL requests.
- `normalize.py`: converts raw KataGo responses into scalars, candidate moves,
  spatial `.npz` shards, and position index records.
- `loader.py`: public `FeatureStore` API for downstream modules.
- `schema.py`: shared schema constants and exceptions.
- `qa.py`: feature-store quality checks.
- `benchmark.py`: creates benchmark folders and runner scripts.
- `compare_benchmark.py`: compares benchmark outputs against a baseline.
- `io.py`: JSONL and atomic-write helpers.

Auxiliary scripts:

- `scripts/quick_tune_katago_params.sh`: short 300-second-per-case parameter
  tuning script for checking GPU utilization and throughput.

Tests:

- `tests/test_module0.py`: minimal end-to-end smoke test.

## Loader Example

```python
from module0_katago_store import FeatureStore

store = FeatureStore("/root/katago_feature_store/v1.0.0")
feat = store.get("kgs_ab12cd34__t000087", fields=["policy_map", "root_winrate"])
batch = store.get_many(["kgs_ab12cd34__t000087"], fields=["policy_map"])
meta = store.describe()
```

## Expected Store Layout

```text
katago_feature_store/
  v1.0.0/
    metadata/
      analysis_profile.json
      schema.json
      dataset_manifest.jsonl
      position_manifest.jsonl
      generation_report.json
    raw/
    normalized/
      scalars/train/part-00000.jsonl
      candidates/train/part-00000.jsonl
      spatial/train/shard-00000.npz
    index/
      position_index.jsonl
    logs/
```

## What Should Not Be Committed

Do not commit generated data or heavy binaries to GitHub:

- KGS archives and extracted full data.
- KataGo executable files.
- KataGo model files such as `*.bin.gz`.
- Full `raw.responses*.jsonl`.
- Generated `normalized/` feature stores.
- Large `.npz` or `.npy` arrays.
- Large benchmark request/response/log files.

Recommended GitHub content:

- Source code in `module0_katago_store/`.
- Small scripts in `scripts/`.
- Tests.
- README and project metadata.
- Small benchmark summary tables if needed.
