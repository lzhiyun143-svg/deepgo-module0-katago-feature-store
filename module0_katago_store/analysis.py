from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from tqdm import tqdm


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _query_turns(query: dict) -> list[int]:
    if "analyzeTurns" in query:
        return [int(turn) for turn in query["analyzeTurns"]]
    return [len(query.get("moves", []))]


def _completed_positions(responses_path: Path) -> set[tuple[str, int]]:
    if not responses_path.exists():
        return set()

    completed = set()
    with responses_path.open("r", encoding="utf-8") as responses:
        for line in responses:
            response = json.loads(line)
            if response.get("isDuringSearch") or "id" not in response or "turnNumber" not in response:
                continue
            game_id = str(response["id"]).split("#", 1)[0]
            completed.add((game_id, int(response["turnNumber"])))
    return completed


def _iter_pending_queries(
    requests_path: Path,
    completed: set[tuple[str, int]],
    max_inflight_positions: int,
    max_positions_per_query: int,
):
    """Yield valid per-game queries with bounded analyzeTurns lists."""
    with requests_path.open("r", encoding="utf-8") as requests:
        for line in requests:
            query = json.loads(line)
            query_id = str(query["id"])
            outstanding_turns = [
                turn for turn in _query_turns(query) if (query_id, turn) not in completed
            ]
            for start in range(0, len(outstanding_turns), max_positions_per_query):
                query_part = query.copy()
                query_part["id"] = f"{query_id}#{start}"
                turns = outstanding_turns[start : start + max_positions_per_query]
                if "analyzeTurns" in query_part:
                    query_part["analyzeTurns"] = turns
                yield json.dumps(query_part, ensure_ascii=False, separators=(",", ":")) + "\n", query_part["id"], len(turns)


def _count_pending_positions(
    requests_path: Path, completed: set[tuple[str, int]]
) -> int:
    expected_responses = 0
    with requests_path.open("r", encoding="utf-8") as requests:
        for line in requests:
            query = json.loads(line)
            query_id = str(query["id"])
            expected_responses += sum(
                (query_id, turn) not in completed for turn in _query_turns(query)
            )
    return expected_responses


def run_katago_analysis_streaming(
    *,
    katago_bin: Path,
    model: Path,
    config: Path,
    requests_path: Path,
    out_path: Path,
    log_path: Path | None = None,
    max_inflight_positions: int = 512,
    max_positions_per_query: int = 64,
    resume: bool = True,
) -> dict:
    """Run one long-lived KataGo analysis process and stream queries to its stdin.

    The model is loaded once and stays resident until all pending queries have
    completed. Only ``max_inflight_positions`` are queued in KataGo at once.
    Long game requests are split into at most ``max_positions_per_query`` turns
    per valid KataGo query so work from multiple games can interleave.
    """
    if not requests_path.exists():
        raise FileNotFoundError(requests_path)
    if max_inflight_positions <= 0:
        raise ValueError("max_inflight_positions must be positive")
    if max_positions_per_query <= 0:
        raise ValueError("max_positions_per_query must be positive")
    if max_positions_per_query > max_inflight_positions:
        raise ValueError("max_positions_per_query cannot exceed max_inflight_positions")

    log_path = log_path or out_path.with_suffix(".log")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _completed_positions(out_path) if resume else set()
    expected_responses = _count_pending_positions(requests_path, completed)
    if not resume and out_path.exists():
        out_path.unlink()

    total_requests = _count_lines(requests_path)
    with log_path.open("a", encoding="utf-8", newline="\n") as main_log:
        main_log.write(
            f"streaming analysis start: requests={requests_path} "
            f"expected_responses={expected_responses} "
            f"max_inflight_positions={max_inflight_positions} "
            f"max_positions_per_query={max_positions_per_query} "
            f"resumed_responses={len(completed)}\n"
        )
        main_log.flush()

        if expected_responses:
            output_mode = "a" if resume else "w"
            with (
                out_path.open(output_mode, encoding="utf-8", newline="\n") as output,
                subprocess.Popen(
                    [str(katago_bin), "analysis", "-model", str(model), "-config", str(config)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=main_log,
                    text=True,
                    encoding="utf-8",
                ) as process,
                tqdm(total=expected_responses, desc="KataGo responses", unit="position") as progress,
            ):
                assert process.stdin is not None
                assert process.stdout is not None
                stdin = process.stdin
                stdout = process.stdout
                condition = threading.Condition()
                in_flight_positions = 0
                chunk_sizes: dict[str, int] = {}

                def drain_responses() -> None:
                    nonlocal in_flight_positions
                    for response_line in stdout:
                        output.write(response_line)
                        output.flush()
                        response = json.loads(response_line)
                        if "error" in response:
                            with condition:
                                in_flight_positions -= chunk_sizes.pop(str(response.get("id")), 0)
                                condition.notify_all()
                            continue
                        if not response.get("isDuringSearch") and "turnNumber" in response:
                            progress.update(1)
                            with condition:
                                in_flight_positions -= 1
                                condition.notify_all()

                reader = threading.Thread(target=drain_responses, daemon=True)
                reader.start()
                try:
                    for submitted_queries, (query_line, chunk_id, position_count) in enumerate(
                        _iter_pending_queries(
                            requests_path,
                            completed,
                            max_inflight_positions,
                            max_positions_per_query,
                        ),
                        1,
                    ):
                        with condition:
                            while in_flight_positions + position_count > max_inflight_positions:
                                condition.wait()
                            in_flight_positions += position_count
                            chunk_sizes[chunk_id] = position_count
                        stdin.write(query_line)
                        stdin.flush()
                        if submitted_queries % 100 == 0:
                            progress.set_postfix(queries=submitted_queries, queued=in_flight_positions)
                except BrokenPipeError:
                    process.wait()
                    raise RuntimeError(f"KataGo exited early with code {process.returncode}") from None
                finally:
                    stdin.close()

                reader.join()
                exit_code = process.wait()
                if exit_code:
                    raise subprocess.CalledProcessError(exit_code, process.args)

    response_lines = _count_lines(out_path) if out_path.exists() else 0
    return {
        "requests": str(requests_path),
        "output": str(out_path),
        "processes_started": 1 if expected_responses else 0,
        "request_lines": total_requests,
        "expected_responses": expected_responses,
        "max_inflight_positions": max_inflight_positions,
        "max_positions_per_query": max_positions_per_query,
        "resumed_responses": len(completed),
        "response_lines": response_lines,
    }
