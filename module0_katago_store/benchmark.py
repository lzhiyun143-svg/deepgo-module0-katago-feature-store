from __future__ import annotations

import json
import shlex
from pathlib import Path

from .manifest import build_manifests
from .profile import AnalysisProfile
from .requests import build_analysis_requests
from .schema import REQUIRED_SCHEMA


def _write_schema(store_root: Path) -> None:
    metadata = store_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "schema.json").write_text(
        json.dumps(REQUIRED_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def setup_benchmark(
    sgf_dir: Path,
    out_root: Path,
    max_games: int = 100,
    visits: list[int] | None = None,
    include_ownership: bool = True,
    analysis_pv_len: int = 10,
) -> dict:
    visits = visits or [1, 25, 50, 100]
    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    first_games = None
    first_positions = None
    for max_visits in visits:
        case_dir = out_root / f"maxvisits_{max_visits}"
        store_root = case_dir / "v1.0.0"
        profile = AnalysisProfile(
            profile_id=f"kg_benchmark_vis{max_visits}_v1",
            max_visits=max_visits,
            analysis_pv_len=analysis_pv_len,
            include_policy=True,
            include_ownership=include_ownership,
        )
        profile.write(store_root)
        _write_schema(store_root)
        games, positions = build_manifests(
            sgf_dir=sgf_dir,
            store_root=store_root,
            dataset_version=f"kgs_benchmark_{max_games}_games",
            max_games=max_games,
        )
        profile_dict = json.loads((store_root / "metadata" / "analysis_profile.json").read_text(encoding="utf-8"))
        requests_path = case_dir / "requests.jsonl"
        requests = build_analysis_requests(store_root, requests_path, profile_dict)
        if first_games is None:
            first_games, first_positions = len(games), len(positions)
        summary.append(
            {
                "max_visits": max_visits,
                "store_root": str(store_root),
                "requests": str(requests_path),
                "games": len(games),
                "positions": len(positions),
                "request_lines": len(requests),
            }
        )

    write_runner(out_root, visits)
    report = {
        "benchmark_root": str(out_root),
        "max_games": max_games,
        "visits": visits,
        "include_ownership": include_ownership,
        "analysis_pv_len": analysis_pv_len,
        "games_per_case": first_games,
        "positions_per_case": first_positions,
        "cases": summary,
    }
    (out_root / "benchmark_setup.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def write_runner(out_root: Path, visits: list[int]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "source /root/module0_env/bin/activate",
        "cd /root/module0_katago_feature_store",
        "KATAGO=${KATAGO:-/root/katago_bin/katago}",
        "MODEL=${MODEL:-/root/katago_bin/model.bin.gz}",
        "CONFIG=${CONFIG:-/root/katago_bin/analysis.cfg}",
        f"BENCH_ROOT={shlex.quote(str(out_root))}",
        "echo \"Benchmark root: $BENCH_ROOT\"",
    ]
    for max_visits in visits:
        case = f"maxvisits_{max_visits}"
        lines.extend(
            [
                f"echo '=== Running {case} ==='",
                f"mkdir -p \"$BENCH_ROOT/{case}/logs\"",
                f"/usr/bin/time -p -o \"$BENCH_ROOT/{case}/time.txt\" "
                f"\"$KATAGO\" analysis -config \"$CONFIG\" -model \"$MODEL\" "
                f"< \"$BENCH_ROOT/{case}/requests.jsonl\" "
                f"> \"$BENCH_ROOT/{case}/raw.responses.jsonl\" "
                f"2> \"$BENCH_ROOT/{case}/katago.log\"",
                f"python -m module0_katago_store.cli normalize "
                f"--store-root \"$BENCH_ROOT/{case}/v1.0.0\" "
                f"--responses \"$BENCH_ROOT/{case}/raw.responses.jsonl\" "
                f"> \"$BENCH_ROOT/{case}/normalize.report.json\"",
                f"python -m module0_katago_store.cli qa "
                f"--store-root \"$BENCH_ROOT/{case}/v1.0.0\" "
                f"> \"$BENCH_ROOT/{case}/qa.report.json\"",
            ]
        )
    lines.append("python -m module0_katago_store.cli benchmark-report --benchmark-root \"$BENCH_ROOT\"")
    runner = out_root / "run_all.sh"
    runner.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runner.chmod(0o755)


def benchmark_report(benchmark_root: Path) -> dict:
    cases = []
    for case_dir in sorted(benchmark_root.glob("maxvisits_*")):
        if not case_dir.is_dir():
            continue
        responses = case_dir / "raw.responses.jsonl"
        requests = case_dir / "requests.jsonl"
        time_file = case_dir / "time.txt"
        qa_file = case_dir / "qa.report.json"
        response_lines = 0
        error_lines = 0
        if responses.exists():
            with responses.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    response_lines += 1
                    if '"error"' in line:
                        error_lines += 1
        request_lines = sum(1 for _ in requests.open("r", encoding="utf-8")) if requests.exists() else 0
        seconds = None
        if time_file.exists():
            for line in time_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("real "):
                    seconds = float(line.split()[1])
        qa = {}
        if qa_file.exists():
            text = qa_file.read_text(encoding="utf-8")
            start = text.find("{")
            if start >= 0:
                qa = json.loads(text[start:])
        positions = qa.get("positions")
        cases.append(
            {
                "case": case_dir.name,
                "request_lines": request_lines,
                "response_lines": response_lines,
                "error_lines": error_lines,
                "seconds": seconds,
                "positions_normalized": positions,
                "positions_per_second": round(positions / seconds, 3) if positions and seconds else None,
            }
        )
    report = {"benchmark_root": str(benchmark_root), "cases": cases}
    (benchmark_root / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

