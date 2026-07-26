from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_jsonl


def run_qa(store_root: Path) -> dict:
    index_rows = list(read_jsonl(store_root / "index" / "position_index.jsonl"))
    ids = [r["position_id"] for r in index_rows]
    duplicate_ids = len(ids) - len(set(ids))
    missing_spatial_files = []
    bad_shapes = []
    ownership_out_of_range = 0

    for shard in sorted((store_root / "normalized" / "spatial").glob("*/*.npz")):
        with np.load(shard, allow_pickle=True) as data:
            if data["policy"].ndim != 3 or data["policy"].shape[1:] != (19, 19):
                bad_shapes.append(str(shard))
            if data["ownership"].ndim != 3 or data["ownership"].shape[1:] != (19, 19):
                bad_shapes.append(str(shard))
            finite = data["ownership"][np.isfinite(data["ownership"])]
            if finite.size and (finite.min() < -1.0001 or finite.max() > 1.0001):
                ownership_out_of_range += 1

    for row in index_rows:
        path = store_root / "normalized" / "spatial" / row["spatial_shard"]
        if not path.exists():
            missing_spatial_files.append(str(path))

    report = {
        "positions": len(index_rows),
        "duplicate_position_ids": duplicate_ids,
        "missing_spatial_files": sorted(set(missing_spatial_files)),
        "bad_shape_files": bad_shapes,
        "ownership_out_of_range_shards": ownership_out_of_range,
        "passed": duplicate_ids == 0 and not missing_spatial_files and not bad_shapes and ownership_out_of_range == 0,
    }
    out = store_root / "metadata" / "generation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
