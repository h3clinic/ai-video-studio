"""Bound Gaussian-part planning using the pinned AgentVideo protocol.

This plans work; it neither edits state nor treats an image as 3D geometry.
Section 14 of upstream's contract is a harness, not a fixed agent count.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

EDIT_KINDS = frozenset(("whole_apple", "apple_slice", "red_peel", "mouth_piece"))
PROTECTED_KINDS = frozenset(("donkey", "bowl", "environment"))
TASK_FIELDS = ("part_id", "kind", "appearance", "geometry", "motion", "preserve_ids")


def validate_manifest(manifest):
    """Bind ownership to a hash-checked state archive, not LLM-invented IDs.

    gaussian_ids must be an explicit persistent identity array in the archive.
    A per-frame pixel index is not silently promoted to material identity.
    Semantic binding_verified is supplied by the trusted segmentation stage;
    this validator proves ID membership/disjointness, not segmentation quality.
    """
    path = Path(manifest["state_path"])
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != manifest["state_sha256"]:
        raise ValueError("State hash mismatch")
    if not isinstance(manifest.get("revision"), str) or not manifest["revision"]:
        raise ValueError("Missing state revision")
    with np.load(path, allow_pickle=False) as state:
        ids = state["gaussian_ids"]
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("State requires unique integer Gaussian IDs")
        n = len(ids)
        for field, shape in (("position", (n, 3)), ("colour", (n, 3)),
                             ("covariance", (n, 3, 3)), ("opacity", (n,))):
            values = state[field]
            if values.shape != shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError("Invalid Gaussian attribute array: " + field)
            if field in ("colour", "opacity") and ((values < 0).any() or (values > 1).any()):
                raise ValueError("Out-of-range Gaussian attribute: " + field)
        cov = state["covariance"]
        if not np.allclose(cov, cov.transpose(0, 2, 1), rtol=1e-6, atol=1e-9):
            raise ValueError("Asymmetric covariance")
        if n and np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ValueError("Non-positive covariance")
        universe = set(map(int, ids))
    parts = manifest["parts"]
    if not isinstance(parts, list) or not 7 <= len(parts) <= 64:
        raise ValueError("Expected bounded edit and protected parts")
    seen, names, edit_kinds, protected_kinds = set(), set(), set(), set()
    for part in parts:
        name, kind = part["part_id"], part["kind"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Invalid or duplicate part ID")
        names.add(name)
        selected = part["gaussian_ids"]
        if not isinstance(selected, list) or not selected or any(type(i) is not int for i in selected):
            raise ValueError("Each part needs nonempty integer Gaussian IDs")
        owned = set(selected)
        if len(owned) != len(selected) or not owned <= universe or owned & seen:
            raise ValueError("Unknown, duplicate, or overlapping Gaussian ownership")
        if part.get("binding_verified") is not True:
            raise ValueError("Unverified semantic binding")
        seen |= owned
        if part.get("protected") is True and kind in PROTECTED_KINDS:
            protected_kinds.add(kind)
        elif part.get("protected") is False and kind in EDIT_KINDS:
            edit_kinds.add(kind)
        else:
            raise ValueError("Unknown kind or invalid protection policy")
    if edit_kinds != EDIT_KINDS or protected_kinds != PROTECTED_KINDS:
        raise ValueError("Missing required fruit or protected region")
    if seen != universe:
        raise ValueError("Unassigned Gaussian state; ownership must be exhaustive")
    return parts


def _issues(proposal, part):
    if not isinstance(proposal, dict) or set(proposal) != set(TASK_FIELDS):
        return ["Return exactly the requested task fields"]
    issues = []
    if proposal["part_id"] != part["part_id"] or proposal["kind"] != part["kind"]:
        issues.append("Part identity and kind cannot change")
    if proposal["preserve_ids"] is not True:
        issues.append("Preserve existing Gaussian identities")
    for field in ("appearance", "geometry", "motion"):
        if not isinstance(proposal[field], str) or not 8 <= len(proposal[field]) <= 1500:
            issues.append(field + " needs a bounded explicit instruction")
    return issues


def run_part_delegation(upstream, client, manifest, parent_proposals=None):
    """Return validated per-part instructions and the actual upstream trace.

    client implements GeminiLLM.chat/model_id. No network client is constructed
    here. Parent proposals are verified via original adopt to avoid redundant
    calls. Strict think has at most two attempts and no fallback.
    """
    parts = validate_manifest(manifest)  # fail before any model call
    if client is None:
        raise ValueError("A real or explicit test client is required; no mock fallback")
    Agent = upstream["Agent"]
    bus = upstream["Bus"](echo=False)
    bus.strict = True
    controller = Agent("video-controller", bus, client, desc="Bounded part-edit planning")
    director = controller.spawn(Agent, "idea-director")
    assigner = director.spawn(Agent, "task-assigner")
    outputs = []
    for part in parts:
        if part["protected"]:
            bus.log(assigner.name, "info", "Protected part: " + part["part_id"])
            continue
        owner = assigner.spawn(Agent, "part:" + part["part_id"], desc=part["kind"])
        verifier = owner.verifier("verify:" + part["part_id"], lambda p, part=part: _issues(p, part))
        proposed = (parent_proposals or {}).get(part["part_id"])
        result = owner.adopt(proposed, TASK_FIELDS, verifier, "Gaussian part instructions")
        if result is None:
            task = {"part_id": part["part_id"], "kind": part["kind"],
                    "required_fields": list(TASK_FIELDS), "preserve_ids": True,
                    "revision": manifest["revision"], "owned_gaussian_count": len(part["gaussian_ids"]),
                    "instruction": "Replace orange semantics with detailed apple material. Whole apple has red skin, stem and natural pores; slices have pale flesh and red skin edges; peel is red thin skin without orange pith; mouth piece follows existing contact/occlusion. Specify appearance, 3D geometry and temporally coherent motion for only this assigned part. Do not change donkey, bowl, environment or other parts. Do not claim image generation proves 3D or motion readiness."}
            result = owner.think(json.dumps(task), verifier, fallback=lambda: {},
                                 max_tries=2, max_tokens=650, what="Gaussian part instructions")
        outputs.append({"owner": owner.name, "gaussian_ids": part["gaussian_ids"],
                        "request": result, "readiness": {"geometry": False, "motion": False, "render": False}})
    return {"revision": manifest["revision"], "state_sha256": manifest["state_sha256"],
            "assignments": outputs, "protected_parts": [p["part_id"] for p in parts if p["protected"]],
            "trace": {"agents": bus.agents, "events": bus.events},
            "upstream": upstream.get("provenance"), "scene_modified": False, "executable": False,
            "limitations": ["Instructions are not generated assets or verified motion.",
                            "Semantic mask correctness needs independent media review.",
                            "Image generation must be fitted to assigned 3D Gaussians before execution."]}
