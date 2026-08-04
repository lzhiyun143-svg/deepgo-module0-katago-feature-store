# DeepGo Module0 + Module1

This repository contains the code for:

- Module0: a unified KataGo feature store built from KGS/SGF games.
- Module1: a multi-step task environment that connects Module0 features with downstream Module2/3/4 experiments.

Large generated data, KataGo binaries, KataGo models, raw responses, and `.npz` feature shards are intentionally not committed.

## Overall Pipeline

```text
KGS / SGF games
  -> Module0: parse games, run KataGo, normalize features
  -> Module1: maintain task state, execute actions, log transitions
  -> Module2: human next-move / user behavior model
  -> Module3: belief, state tracking, and quality signals
  -> Module4: high-level delegation policy, explanation, and training strategy
```

Module0 is the feature database. Module1 is the environment layer and the single source of truth for state transitions. Later modules should not directly rewrite board state.

## Server Paths Used In The Current Experiment

```text
Module0/1 code:      /root/module0_katago_feature_store
Python environment: /root/module0_env
KGS data:            /root/deepgo/data/kgs
Feature store:       /root/katago_feature_store/v1.0.0
KataGo executable:   /root/katago_bin/katago
KataGo model:        /root/katago_bin/model.bin.gz
KataGo config:       /root/katago_bin/analysis.cfg
```

## Module0 Commands

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

Build manifests:

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

Run KataGo analysis:

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

Normalize:

```bash
python -m module0_katago_store.cli normalize \
  --store-root /root/katago_feature_store/v1.0.0 \
  --responses /root/katago_feature_store/raw.responses.jsonl
```

QA:

```bash
python -m module0_katago_store.cli qa \
  --store-root /root/katago_feature_store/v1.0.0
```

## Module1 Usage

Module1 exposes `MultiStepTaskEnv`. It accepts either physical Go actions or high-level Module4 delegation decisions.

```python
from module1_environment import EnvironmentConfig, EnvironmentMode, MultiStepTaskEnv

env = MultiStepTaskEnv(
    store_root="/root/katago_feature_store/v1.0.0",
    transition_log_path="/root/module1_runs/transitions.jsonl",
    config=EnvironmentConfig(
        mode=EnvironmentMode.OFFLINE_REPLAY,
        auto_opponent=True,
    ),
    max_steps=100,
)

obs = env.reset()
next_obs, reward_components, done, info = env.step({"delegation": "ai"})
env.close()
```

Supported action forms:

```python
env.step(288)                         # physical action index
env.step("Q16")                       # physical GTP move
env.step({"type": "katago_best"})     # use Module0/KataGo top candidate
env.step({"delegation": "ai"})        # Module4 delegates to AI
env.step({"delegation": "human"})     # replay or Module2 human model
env.step({"delegation": "physical", "action_index": 288})
```

## Module1 Input And Output Contract

Module1 consumes these Module0 files:

```text
feature_store/v1.0.0/metadata/dataset_manifest.jsonl
feature_store/v1.0.0/index/position_index.jsonl
feature_store/v1.0.0/normalized/scalars/
feature_store/v1.0.0/normalized/candidates/
feature_store/v1.0.0/normalized/spatial/
```

`reset()` and `step()` return `observation` with:

```text
ids                 episode_id, game_id, position_id, turn_number, split
board               17 x 19 x 19 board tensor
legal_mask          362 legal-action mask
task_scalars        task-side scalar features
katago_scalars      winrate, scoreLead, utility, policy entropy, top gap
katago_spatial      ownership map and future spatial features
top_candidates      KataGo candidate moves
history_summary     compact transition history
feature_mask        feature availability flags
belief              Module3 belief state
```

`step()` returns:

```text
next_obs, reward_components, done, info
```

`info` includes:

```text
meta_decision          original Module4 decision
physical_action        executed team-side action
opponent_action        optional auto-opponent action
module3_belief         updated belief state
reward_components      decomposed reward
state_hash_before      board-state hash before action
state_hash_after       board-state hash after action
```

## Downstream Module Alignment

Module2 input:

```text
observation["board"]
observation["legal_mask"]
observation["katago_scalars"]
observation["top_candidates"]
player_profile
```

Module2 output expected by Module1:

```python
{"action_index": 288}
{"move": "Q16"}
{"policy": [362 probabilities]}
```

Module3 input:

```text
observation
transition/info
previous_belief
```

Module3 output expected by Module1:

```python
{
  "coherence": 0.5,
  "understanding": 0.5,
  "readiness": 0.5,
  "agency_alignment": 0.5,
  "trust_alignment": 0.5
}
```

Module4 input:

```text
observation
belief
```

Module4 output accepted by Module1:

```python
{"delegation": "human"}
{"delegation": "ai"}
{"delegation": "physical", "action_index": 288}
```

## Code File Guide

Module0 package: `module0_katago_store/`

- `cli.py`: command-line entry point.
- `manifest.py`: builds game and position manifests.
- `sgf.py`: SGF parser.
- `coords.py`: Go coordinate conversion.
- `ids.py`: stable ID helpers.
- `profile.py`: KataGo analysis profile metadata.
- `requests.py`: builds KataGo JSONL requests.
- `analysis.py`: streams requests through KataGo analysis.
- `normalize.py`: normalizes KataGo responses into store files.
- `loader.py`: public `FeatureStore` API.
- `qa.py`: feature-store QA.
- `benchmark.py`: benchmark setup and reports.
- `compare_benchmark.py`: benchmark comparison.

Module1 package: `module1_environment/`

- `core/environment.py`: main `MultiStepTaskEnv`.
- `core/types.py`: environment modes, config, and model protocols.
- `core/go_state.py`: lightweight Go board/rule state.
- `core/action_resolver.py`: resolves physical, AI, human, and policy actions.
- `core/observation_builder.py`: builds downstream observations.
- `core/opponent_driver.py`: optional opponent move driver.
- `core/reward.py`: decomposed reward calculation.
- `core/transition.py`: transition record schema.
- `adapters/module0_adapter.py`: reads Module0 features.
- `adapters/module2_adapter.py`: wraps a human/user model.
- `adapters/module3_adapter.py`: wraps a belief/state model.
- `adapters/module4_adapter.py`: normalizes delegation decisions.
- `datasets/episode_manifest.py`: creates replayable episodes.
- `logging/transition_writer.py`: writes JSONL transition logs.

## KataGo Setup

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

Recommended A10 analysis config from quick tuning:

```text
maxVisits = 25
nnMaxBatchSize = 128
numAnalysisThreads = 8
numSearchThreadsPerAnalysisThread = 4
reportAnalysisWinratesAs = SIDETOMOVE
```

A copy is included at:

```text
configs/analysis.a10.maxvisits25.cfg
```

## Benchmark Result Summary

Quick tuning on one NVIDIA A10:

| Case | maxVisits | nnMaxBatchSize | numAnalysisThreads | numSearchThreads | GPU avg | GPU max | Lines/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v25_batch64_a2_s16_baseline` | 25 | 64 | 2 | 16 | 43.0% | 48% | 11.91 |
| `v25_batch128_a4_s8` | 25 | 128 | 4 | 8 | 71.1% | 78% | 21.59 |
| `v25_batch256_a8_s4` | 25 | 256 | 8 | 4 | 90.0% | 98% | 28.72 |
| `v25_batch128_a8_s4` | 25 | 128 | 8 | 4 | 92.7% | 98% | 29.07 |

Earlier benchmark:

```text
Games: 100
maxVisits tested: 1, 25, 50, 100
Comparable positions: 16600
```

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

## What Should Not Be Committed

Do not commit:

- KGS archives and extracted full data.
- KataGo executables.
- KataGo model files such as `*.bin.gz`.
- Full `raw.responses*.jsonl`.
- Generated `normalized/` feature stores.
- Large `.npz` or `.npy` arrays.
- Large benchmark request/response/log files.

Recommended GitHub content:

- Source code in `module0_katago_store/` and `module1_environment/`.
- Small configs and scripts.
- Tests.
- README and project metadata.
- Small benchmark summaries.
