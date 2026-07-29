from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import run_katago_analysis_chunked
from .manifest import build_manifests
from .benchmark import benchmark_report, setup_benchmark
from .compare_benchmark import compare_benchmark
from .normalize import normalize_responses
from .profile import AnalysisProfile
from .qa import run_qa
from .requests import build_analysis_requests
from .schema import REQUIRED_SCHEMA


def init_store(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for rel in [
        "metadata",
        "raw",
        "normalized/scalars/train",
        "normalized/scalars/valid",
        "normalized/scalars/test",
        "normalized/candidates/train",
        "normalized/candidates/valid",
        "normalized/candidates/test",
        "normalized/spatial/train",
        "normalized/spatial/valid",
        "normalized/spatial/test",
        "index",
        "logs",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    profile = AnalysisProfile(max_visits=args.max_visits)
    profile.write(root)
    (root / "metadata" / "schema.json").write_text(
        json.dumps(REQUIRED_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Initialized store: {root}")


def build_manifest_cmd(args: argparse.Namespace) -> None:
    games, positions = build_manifests(Path(args.sgf_dir), Path(args.store_root), max_games=args.max_games)
    print(f"games={len(games)} positions={len(positions)}")


def make_requests_cmd(args: argparse.Namespace) -> None:
    root = Path(args.store_root)
    profile = json.loads((root / "metadata" / "analysis_profile.json").read_text(encoding="utf-8"))
    requests = build_analysis_requests(root, Path(args.out), profile)
    print(f"requests={len(requests)} out={args.out}")


def normalize_cmd(args: argparse.Namespace) -> None:
    root = Path(args.store_root)
    profile = json.loads((root / "metadata" / "analysis_profile.json").read_text(encoding="utf-8"))
    report = normalize_responses(root, Path(args.responses), profile, top_n=args.top_n)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_analysis_cmd(args: argparse.Namespace) -> None:
    report = run_katago_analysis_chunked(
        katago_bin=Path(args.katago_bin),
        model=Path(args.model),
        config=Path(args.config),
        requests_path=Path(args.requests),
        out_path=Path(args.out),
        log_path=Path(args.log) if args.log else None,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        chunk_lines=args.chunk_lines,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def qa_cmd(args: argparse.Namespace) -> None:
    report = run_qa(Path(args.store_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def benchmark_setup_cmd(args: argparse.Namespace) -> None:
    visits = [int(v) for v in args.visits.split(",") if v.strip()]
    report = setup_benchmark(
        sgf_dir=Path(args.sgf_dir),
        out_root=Path(args.benchmark_root),
        max_games=args.max_games,
        visits=visits,
        include_ownership=not args.no_ownership,
        analysis_pv_len=args.analysis_pv_len,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def benchmark_report_cmd(args: argparse.Namespace) -> None:
    report = benchmark_report(Path(args.benchmark_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def benchmark_compare_cmd(args: argparse.Namespace) -> None:
    report = compare_benchmark(Path(args.benchmark_root), baseline=args.baseline)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="module0-katago-store")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("init-store")
    p.add_argument("--root", required=True)
    p.add_argument("--max-visits", type=int, default=100)
    p.set_defaults(func=init_store)

    p = sub.add_parser("build-manifest")
    p.add_argument("--sgf-dir", required=True)
    p.add_argument("--store-root", required=True)
    p.add_argument("--max-games", type=int)
    p.set_defaults(func=build_manifest_cmd)

    p = sub.add_parser("make-requests")
    p.add_argument("--store-root", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=make_requests_cmd)

    p = sub.add_parser("run-analysis")
    p.add_argument("--katago-bin", default="/root/katago_bin/katago")
    p.add_argument("--model", default="/root/katago_bin/model.bin.gz")
    p.add_argument("--config", default="/root/katago_bin/analysis.cfg")
    p.add_argument("--requests", default="/root/katago_feature_store/requests.jsonl")
    p.add_argument("--out", default="/root/katago_feature_store/raw.responses.jsonl")
    p.add_argument("--log", default="/root/katago_feature_store/katago.analysis.log")
    p.add_argument("--work-dir")
    p.add_argument("--chunk-lines", type=int, default=500)
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(func=run_analysis_cmd)

    p = sub.add_parser("normalize")
    p.add_argument("--store-root", required=True)
    p.add_argument("--responses", required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.set_defaults(func=normalize_cmd)

    p = sub.add_parser("qa")
    p.add_argument("--store-root", required=True)
    p.set_defaults(func=qa_cmd)

    p = sub.add_parser("benchmark-setup")
    p.add_argument("--sgf-dir", required=True)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--max-games", type=int, default=100)
    p.add_argument("--visits", default="1,25,50,100")
    p.add_argument("--analysis-pv-len", type=int, default=10)
    p.add_argument("--no-ownership", action="store_true")
    p.set_defaults(func=benchmark_setup_cmd)

    p = sub.add_parser("benchmark-report")
    p.add_argument("--benchmark-root", required=True)
    p.set_defaults(func=benchmark_report_cmd)

    p = sub.add_parser("benchmark-compare")
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--baseline", default="maxvisits_100")
    p.set_defaults(func=benchmark_compare_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
