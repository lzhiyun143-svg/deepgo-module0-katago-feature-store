from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisProfile:
    profile_id: str = "kg_main_vis100_v1"
    katago_version: str = "RECORD_EXACT_VERSION"
    executable_sha256: str = ""
    main_model_filename: str = ""
    main_model_sha256: str = ""
    analysis_config_sha256: str = ""
    max_visits: int = 100
    analysis_pv_len: int = 10
    include_policy: bool = True
    include_ownership: bool = True
    include_ownership_stdev: bool = False
    report_perspective: str = "SIDE_TO_MOVE"
    rules_fallback: str = "chinese"
    komi_fallback: float = 7.5
    schema_version: str = "1.0"

    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def write(self, store_root: Path) -> None:
        metadata = store_root / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["profile_sha256"] = self.content_hash()
        (metadata / "analysis_profile.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

