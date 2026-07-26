from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

from .io import read_jsonl


def _load_scalars(case_dir: Path) -> dict[str, dict]:
    base = case_dir / "v1.0.0" / "normalized" / "scalars"
    rows: dict[str, dict] = {}
    for path in base.glob("*/*.jsonl"):
        for rec in read_jsonl(path):
            rows[rec["position_id"]] = rec
    return rows


def _load_top_moves(case_dir: Path) -> dict[str, str | None]:
    base = case_dir / "v1.0.0" / "normalized" / "candidates"
    top: dict[str, str | None] = {}
    for path in base.glob("*/*.jsonl"):
        for rec in read_jsonl(path):
            if int(rec.get("candidate_rank", 0)) == 1:
                top[rec["position_id"]] = rec.get("move")
    return top


def _finite_delta(a, b) -> float | None:
    if a is None or b is None:
        return None
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(af) or not math.isfinite(bf):
        return None
    return abs(af - bf)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    vals = sorted(values)
    p90_idx = min(len(vals) - 1, int(0.9 * (len(vals) - 1)))
    return {
        "count": len(vals),
        "mean": round(mean(vals), 6),
        "median": round(median(vals), 6),
        "p90": round(vals[p90_idx], 6),
        "max": round(vals[-1], 6),
    }


def compare_benchmark(benchmark_root: Path, baseline: str = "maxvisits_100") -> dict:
    baseline_dir = benchmark_root / baseline
    baseline_scalars = _load_scalars(baseline_dir)
    baseline_top = _load_top_moves(baseline_dir)
    cases = []

    for case_dir in sorted(benchmark_root.glob("maxvisits_*")):
        if not case_dir.is_dir() or case_dir.name == baseline:
            continue
        scalars = _load_scalars(case_dir)
        top = _load_top_moves(case_dir)
        common = sorted(set(baseline_scalars) & set(scalars))
        top_same = 0
        top_compared = 0
        winrate_deltas = []
        score_deltas = []
        entropy_deltas = []
        policy_deltas = []
        rank_deltas = []
        rank_same = 0
        rank_compared = 0

        for pid in common:
            b = baseline_scalars[pid]
            c = scalars[pid]
            if pid in baseline_top and pid in top:
                top_compared += 1
                if baseline_top[pid] == top[pid]:
                    top_same += 1
            for key, bucket in [
                ("root_winrate", winrate_deltas),
                ("root_score_lead", score_deltas),
                ("policy_entropy", entropy_deltas),
                ("human_move_policy", policy_deltas),
            ]:
                delta = _finite_delta(b.get(key), c.get(key))
                if delta is not None:
                    bucket.append(delta)
            br = b.get("human_move_rank_policy")
            cr = c.get("human_move_rank_policy")
            if br is not None and cr is not None:
                rank_compared += 1
                if int(br) == int(cr):
                    rank_same += 1
                rank_deltas.append(abs(int(br) - int(cr)))

        cases.append(
            {
                "case": case_dir.name,
                "baseline": baseline,
                "common_positions": len(common),
                "top1_same_rate": round(top_same / top_compared, 6) if top_compared else None,
                "top1_compared": top_compared,
                "human_rank_same_rate": round(rank_same / rank_compared, 6) if rank_compared else None,
                "human_rank_delta": _summary(rank_deltas),
                "root_winrate_abs_delta": _summary(winrate_deltas),
                "root_score_lead_abs_delta": _summary(score_deltas),
                "policy_entropy_abs_delta": _summary(entropy_deltas),
                "human_move_policy_abs_delta": _summary(policy_deltas),
            }
        )

    report = {"benchmark_root": str(benchmark_root), "baseline": baseline, "cases": cases}
    out = benchmark_root / f"benchmark_compare_vs_{baseline}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

