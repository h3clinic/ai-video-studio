import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from real_video import runway_bootstrap as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_paid_submission_not_retried_and_key_not_in_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'artifacts'/'trial'
            store=Mock();store.load_for_api.return_value='SECRET_DUMMY'
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'CredentialStore',return_value=store),patch.object(bootstrap,'request',side_effect=RuntimeError('Runway HTTP 402')) as request,patch('builtins.print'):
                bootstrap.run(out,'image',True)
                report=(out/'reference/report.json').read_text()
                self.assertNotIn('SECRET_DUMMY',report)
                self.assertEqual(json.loads(report)['estimated_credits'],5)
                with self.assertRaises(FileExistsError):bootstrap.run(out,'image',True)
                self.assertEqual(request.call_count,1)
    def test_cached_success_revalidates_media_without_api_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'artifacts'/'trial';reference=out/'reference'
            reference.mkdir(parents=True)
            (reference/'report.json').write_text(json.dumps({'output_sha256':'a'*64}))
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'CredentialStore'),patch.object(bootstrap,'request') as request:
                with self.assertRaises(ValueError):bootstrap.run(out,'image')
                request.assert_not_called()


if __name__=='__main__':unittest.main()
