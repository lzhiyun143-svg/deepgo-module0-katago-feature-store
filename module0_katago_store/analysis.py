from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _write_chunks(requests_path: Path, chunks_dir: Path, chunk_lines: int) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    chunk_fh = None
    try:
        with requests_path.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src):
                if line_no % chunk_lines == 0:
                    if chunk_fh is not None:
                        chunk_fh.close()
                    chunk_path = chunks_dir / f"requests_part_{len(chunks):05d}.jsonl"
                    chunks.append(chunk_path)
                    chunk_fh = chunk_path.open("w", encoding="utf-8", newline="\n")
                chunk_fh.write(line)
    finally:
        if chunk_fh is not None:
            chunk_fh.close()
    return chunks


def _load_existing_chunks(chunks_dir: Path) -> list[Path]:
    return sorted(chunks_dir.glob("requests_part_*.jsonl"))


def _prepare_chunks(requests_path: Path, work_dir: Path, chunk_lines: int, resume: bool) -> list[Path]:
    chunks_dir = work_dir / "requests"
    manifest_path = work_dir / "chunk_manifest.json"
    expected = {
        "requests_path": str(requests_path),
        "requests_size": requests_path.stat().st_size,
        "requests_mtime_ns": requests_path.stat().st_mtime_ns,
        "chunk_lines": chunk_lines,
    }

    if resume and manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = _load_existing_chunks(chunks_dir)
        if actual == expected and chunks:
            return chunks

    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks = _write_chunks(requests_path, chunks_dir, chunk_lines)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def run_katago_analysis_chunked(
    *,
    katago_bin: Path,
    model: Path,
    config: Path,
    requests_path: Path,
    out_path: Path,
    log_path: Path | None = None,
    work_dir: Path | None = None,
    chunk_lines: int = 500,
    resume: bool = True,
) -> dict:
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if not requests_path.exists():
        raise FileNotFoundError(requests_path)

    work_dir = work_dir or out_path.parent / f"katago_chunks_{chunk_lines}"
    log_path = log_path or out_path.with_suffix(".log")
    chunks = _prepare_chunks(requests_path, work_dir, chunk_lines, resume)
    responses_dir = work_dir / "responses"
    logs_dir = work_dir / "logs"
    responses_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_requests = _count_lines(requests_path)
    completed = 0
    with log_path.open("a", encoding="utf-8", newline="\n") as main_log:
        main_log.write(
            f"chunked analysis start: requests={requests_path} chunks={len(chunks)} "
            f"chunk_lines={chunk_lines}\n"
        )
        main_log.flush()

        for idx, chunk_path in enumerate(chunks, 1):
            response_path = responses_dir / f"{chunk_path.stem}.responses.jsonl"
            chunk_log_path = logs_dir / f"{chunk_path.stem}.katago.log"
            done_path = response_path.with_suffix(response_path.suffix + ".done")
            if resume and done_path.exists() and response_path.exists() and response_path.stat().st_size > 0:
                completed += 1
                main_log.write(f"[{idx}/{len(chunks)}] skip {chunk_path.name}\n")
                main_log.flush()
                continue

            main_log.write(f"[{idx}/{len(chunks)}] run {chunk_path.name}\n")
            main_log.flush()
            if done_path.exists():
                done_path.unlink()
            with chunk_path.open("rb") as stdin, response_path.open("wb") as stdout, chunk_log_path.open("wb") as stderr:
                subprocess.run(
                    [
                        str(katago_bin),
                        "analysis",
                        "-model",
                        str(model),
                        "-config",
                        str(config),
                    ],
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                )
            if response_path.stat().st_size == 0:
                raise RuntimeError(f"KataGo produced an empty response for {chunk_path}")
            done_path.write_text("done\n", encoding="utf-8")
            completed += 1

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_out.open("wb") as merged:
        for chunk_path in chunks:
            response_path = responses_dir / f"{chunk_path.stem}.responses.jsonl"
            with response_path.open("rb") as part:
                shutil.copyfileobj(part, merged)
    tmp_out.replace(out_path)

    return {
        "requests": str(requests_path),
        "output": str(out_path),
        "work_dir": str(work_dir),
        "chunk_lines": chunk_lines,
        "chunks": len(chunks),
        "completed_chunks": completed,
        "request_lines": total_requests,
        "response_lines": _count_lines(out_path),
    }
