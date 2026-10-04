import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from real_video.agentvideo_bridge import load_upstream
from real_video.agentvideo_parts import EDIT_KINDS, PROTECTED_KINDS, run_part_delegation, validate_manifest


class Client:
    model_id = "fixture/no-network"
    def __init__(self): self.calls = 0
    def chat(self, messages, **kw):
        self.calls += 1
        task = json.loads(messages[1]["content"])
        return json.dumps(dict(part_id=task["part_id"], kind=task["kind"], preserve_ids=True,
                               appearance="Red apple skin with subtle natural pores",
                               geometry="Preserve contact and existing silhouette",
                               motion="Track source part with stable identity"))


class PartsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name)/"state.npz"
        np.savez(path, gaussian_ids=np.arange(7), position=np.zeros((7,3)),
                 colour=np.ones((7,3))*.5, covariance=np.tile(np.eye(3),(7,1,1)),
                 opacity=np.ones(7))
        kinds = sorted(EDIT_KINDS) + sorted(PROTECTED_KINDS)
        self.manifest = dict(state_path=str(path), state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             revision="v1", parts=[dict(part_id=k, kind=k, gaussian_ids=[i],
                             protected=k in PROTECTED_KINDS, binding_verified=True) for i,k in enumerate(kinds)])

    def upstream(self):
        path = Path(__file__).resolve().parents[3]/"work"/"agentvideo-reference"
        if not path.exists(): self.skipTest("Pinned upstream checkout unavailable")
        return load_upstream(path)

    def test_actual_protocol_and_adopt(self):
        client = Client()
        out = run_part_delegation(self.upstream(), client, self.manifest)
        self.assertEqual(client.calls, 4)
        self.assertFalse(out["executable"])
        self.assertFalse(out["scene_modified"])
        self.assertEqual(len(out["assignments"]),4)
        self.assertEqual(out["trace"]["agents"]["task-assigner"]["parent"], "idea-director")
        proposals = {a["request"]["part_id"]:a["request"] for a in out["assignments"]}
        again = run_part_delegation(self.upstream(), client, self.manifest, proposals)
        self.assertEqual(client.calls, 4)
        self.assertEqual(sum(e["kind"] == "pass" for e in again["trace"]["events"]),4)

    def test_unbound_rejected_before_calls(self):
        client = Client()
        self.manifest["parts"][0]["binding_verified"] = False
        with self.assertRaises(ValueError): run_part_delegation({}, client, self.manifest)
        self.assertEqual(client.calls, 0)

    def test_conflicts_and_unknown_ids(self):
        for ids in ([1], [99], [True], []):
            m = copy.deepcopy(self.manifest)
            m["parts"][0]["gaussian_ids"] = ids
            with self.assertRaises(ValueError): validate_manifest(m)

    def test_wrong_hash(self):
        self.manifest["state_sha256"] = "0"*64
        with self.assertRaises(ValueError): validate_manifest(self.manifest)

    def test_ids_without_attributes_rejected(self):
        path = Path(self.manifest['state_path'])
        np.savez(path, gaussian_ids=np.arange(7))
        self.manifest['state_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(KeyError): validate_manifest(self.manifest)

    def test_strict_failed_proposals_do_not_fallback(self):
        class Invalid(Client):
            def chat(self, *args, **kw): self.calls += 1; return '{}'
        client = Invalid()
        upstream = self.upstream()
        with self.assertRaises(upstream["AgentFailed"]):
            run_part_delegation(upstream, client, self.manifest)
        self.assertEqual(client.calls,2)


if __name__ == "__main__": unittest.main()
