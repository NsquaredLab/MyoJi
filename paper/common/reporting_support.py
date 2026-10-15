"""Report label availability without silently dropping any scorable participant."""

import hashlib
import json
from pathlib import Path

import numpy as np


def scoring_support(users, valid):
    users, valid = np.asarray(users).astype(str), np.asarray(valid)
    if users.ndim != 1 or valid.ndim != 2 or len(users) != len(valid) or valid.dtype != bool:
        raise ValueError("one user per window and a boolean frame mask are required")
    people = {}
    for user in sorted(set(users)):
        selected = valid[users == user]
        people[str(user)] = {"windows": len(selected), "windows_with_scored_labels": int(selected.any(1).sum()),
                            "valid_scored_frames": int(selected.sum())}
    scored = [u for u, value in people.items() if value["valid_scored_frames"] > 0]
    if not scored:
        raise ValueError("no participant has a valid scored label")
    return {"present_participants": len(people), "scored_participants": scored,
            "unscorable_participants": [u for u in people if u not in scored],
            "windows": len(users), "windows_with_scored_labels": int(valid.any(1).sum()),
            "valid_scored_frames": int(valid.sum()), "per_participant": people,
            "rule": "Use the existing validity mask over scored frames. Retain all windows and explicitly identify participants with zero scored labels; no prediction-based exclusion."}


def require_scored_participants(scores, support):
    if set(scores["per_user"]) != set(support["scored_participants"]):
        raise ValueError("reported participants differ from valid-label support")


def recovery_provenance(root, manifest):
    if manifest is None:
        return None
    manifest = Path(manifest)
    value = json.loads(manifest.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if value["protocol_sha256"] != sha(Path(root) / "protocol.json"):
        raise ValueError("reporting recovery uses another protocol")
    for name, expected in value["patched_source_sha256"].items():
        if sha(manifest.parent / name) != expected:
            raise ValueError("reporting recovery source changed")
    return {"manifest": str(manifest), "sha256": sha(manifest), "scope": value["scope"]}
