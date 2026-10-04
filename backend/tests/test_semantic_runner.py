import ast
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from cloud.run_semantic_alternative import directory_bootstrap_code, JOB


class SemanticRunnerTests(unittest.TestCase):
    def test_bootstrap_uses_exclusive_creation_never_overwrites(self):
        code=directory_bootstrap_code()
        ast.parse(code)
        self.assertIn('exist_ok=False',code)
        self.assertIn('root.resolve()==root',code)
        self.assertIn('not target.is_symlink()',code)
        for forbidden in ('unlink(', 'rmtree(', 'write_bytes(', 'load_key(', 'environ'):
            self.assertNotIn(forbidden,code)

    def test_parent_preflight_does_not_initialize_cuda(self):
        ast.parse(JOB)
        self.assertNotIn('import torch',JOB)
        self.assertNotIn('torch.cuda',JOB)
        self.assertIn('--query-gpu=name',JOB)

    def test_shutdown_watch_exits_only_after_final_and_provider_agree(self):
        from cloud import run_semantic_fresh as fresh
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'allocation.json').write_text(json.dumps(dict(pod='owned123',name='trial',remote_deadline_epoch=200)))
            (root/'final_status.json').write_text(json.dumps(dict(id='owned123',desiredStatus='EXITED')))
            with patch.object(fresh,'OUT',root),patch.object(fresh,'NAME','trial'), \
                    patch.object(fresh.time,'time',return_value=100),patch.object(fresh.time,'sleep') as sleep, \
                    patch.object(fresh,'rest',return_value={'desiredStatus':'EXITED'}) as provider:
                fresh.watch()
                provider.assert_called_once_with('GET','pods/owned123')
                sleep.assert_not_called()


if __name__=='__main__':unittest.main()
