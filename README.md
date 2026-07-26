# Module0 KataGo Feature Store

This package is an MVP implementation of the Module0 design: build a shared,
versioned KataGo feature store from SGF/game records and expose a stable Python
loader for downstream modules.

中文说明：这个目录只包含 Module0 的代码，不包含大规模数据、KataGo 模型、
KataGo 可执行文件或已经生成的 raw response。它的作用是把棋谱数据整理成
KataGo analysis 请求，并把 KataGo 输出标准化成后续模型可读取的特征库。

## What It Does

1. Build `dataset_manifest.jsonl` and `position_manifest.jsonl` from SGF files.
2. Generate KataGo Parallel Analysis Engine JSONL requests.
3. Normalize KataGo JSONL responses into:
   - `normalized/scalars/<split>/part-00000.jsonl`
   - `normalized/candidates/<split>/part-00000.jsonl`
   - `normalized/spatial/<split>/shard-00000.npz`
   - `index/position_index.jsonl`
4. Load features by `position_id`, `get_many`, or `get_game`.
5. Run basic QA checks for duplicates, missing files, shapes, and ranges.

The implementation intentionally keeps storage simple and inspectable for the
first version. Parquet/Zarr can be added later without changing the public
loader API.

## Quick Start

```bash
cd module0_katago_feature_store
python -m module0_katago_store.cli init-store --root example_store/v1.0.0
python -m module0_katago_store.cli build-manifest --sgf-dir ../history_data_可读SGF真人棋谱 --store-root example_store/v1.0.0
python -m module0_katago_store.cli make-requests --store-root example_store/v1.0.0 --out example_store/requests.jsonl
```

After running KataGo analysis and saving responses:

```bash
python -m module0_katago_store.cli normalize --store-root example_store/v1.0.0 --responses raw.responses.jsonl
python -m module0_katago_store.cli qa --store-root example_store/v1.0.0
```

Loader example:

```python
from module0_katago_store import FeatureStore

store = FeatureStore("example_store/v1.0.0")
feat = store.get("kgs_ab12cd34__t000087", fields=["policy_map", "root_winrate"])
batch = store.get_many(["kgs_ab12cd34__t000087"], fields=["policy_map"])
meta = store.describe()
```

## Code File Guide

Core package: `module0_katago_store/`

- `cli.py`
  Command-line entry point. All common operations are exposed here, including
  `init-store`, `build-manifest`, `make-requests`, `normalize`, `qa`,
  `benchmark-setup`, `benchmark-report`, and `benchmark-compare`.

- `manifest.py`
  Scans SGF files and builds two manifest files:
  `dataset_manifest.jsonl` for game-level metadata and
  `position_manifest.jsonl` for per-position records. It assigns stable
  `game_id` and `position_id` values.

- `sgf.py`
  Lightweight SGF parser used by `manifest.py`. It extracts board size, rules,
  komi, players, result, and moves, and converts SGF coordinates to KataGo/GTP
  coordinates.

- `coords.py`
  Coordinate utilities for Go boards. It converts between KataGo coordinates
  such as `Q16`, zero-based `(row, col)`, and flat indices. It also handles
  `pass` and the skipped `I` column.

- `ids.py`
  Stable ID helpers. It hashes source paths and file contents to create
  reproducible `game_id` values and formats `position_id` values such as
  `kgs_ab12cd34__t000087`.

- `profile.py`
  Defines the immutable KataGo analysis profile, including `maxVisits`,
  `includePolicy`, `includeOwnership`, `analysisPVLen`, model/config hashes,
  and schema version.

- `requests.py`
  Converts the manifests into KataGo Parallel Analysis Engine JSONL requests.
  Each request usually represents one game with multiple `analyzeTurns`.

- `normalize.py`
  Converts KataGo raw JSONL responses into normalized feature-store outputs:
  scalar JSONL records, candidate-move JSONL records, spatial `.npz` shards,
  and a `position_index.jsonl`.

- `loader.py`
  Public Python API for downstream modules. It provides `FeatureStore.get()`,
  `FeatureStore.get_many()`, and `FeatureStore.get_game()` so Module1-4 do not
  need to parse raw KataGo JSON.

- `schema.py`
  Shared exceptions and the current schema declaration. It is used by the
  loader and initialization code to keep feature names, shapes, and versions
  explicit.

- `qa.py`
  Basic quality checks for generated stores: duplicate `position_id`, missing
  spatial shards, wrong tensor shapes, and ownership values outside `[-1, 1]`.

- `benchmark.py`
  Creates benchmark folders and runner scripts for comparing different
  `maxVisits` values on the same fixed sample of games.

- `compare_benchmark.py`
  Compares benchmark results against a baseline such as `maxVisits=100`. It
  reports Top-1 agreement, winrate differences, scoreLead differences, and
  human-move rank differences.

- `io.py`
  Small JSONL and atomic-write utilities used across the package.

- `__init__.py`
  Exports the public `FeatureStore` class and common exceptions.

Tests:

- `tests/test_module0.py`
  Minimal end-to-end smoke test covering coordinate conversion, manifest
  generation, response normalization, QA, and loader access.

Project metadata:

- `pyproject.toml`
  Python package metadata. The only required runtime dependency is `numpy`.

## What Should Not Be Committed

Do not commit generated data or heavy binaries to GitHub:

- KataGo executable files
- KataGo model files such as `*.bin.gz`
- full `raw.responses*.jsonl`
- generated `normalized/` feature stores
- large `.npz` shards
- full benchmark output files if they are large

Recommended GitHub content:

- source code in `module0_katago_store/`
- `tests/`
- `README.md`
- `pyproject.toml`
- small benchmark summary JSON files, if needed

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
