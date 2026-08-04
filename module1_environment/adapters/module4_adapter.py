from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DelegationDecision:
    delegation: str
    metadata: dict


def normalize_delegation(decision) -> DelegationDecision:
    if isinstance(decision, str):
        return DelegationDecision(decision.lower(), {"raw_decision": decision})
    if not isinstance(decision, dict):
        raise TypeError(f"unsupported Module4 decision type: {type(decision)!r}")
    delegation = decision.get("delegation") or decision.get("type") or decision.get("actor_source")
    if delegation is None:
        raise ValueError("Module4 decision must include delegation/type/actor_source")
    return DelegationDecision(str(delegation).lower(), dict(decision))
