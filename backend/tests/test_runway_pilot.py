import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from real_video import runway_pilot as pilot


class PilotTests(unittest.TestCase):
    def test_failed_submission_is_saved_and_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'source.png';source.write_bytes(b'fixture-not-sent')
            out=root/'artifacts'/'pilot'
            store=Mock();store.load_for_api.return_value='SECRET_FIXTURE'
            with patch.object(pilot,'ROOT',root),patch.object(pilot,'SOURCE',source),patch.object(pilot,'CredentialStore',return_value=store),patch.object(pilot,'request',side_effect=RuntimeError('Runway HTTP 402')) as request,patch('builtins.print'):
                pilot.run(out,True)
                self.assertEqual(request.call_count,1)
                report=(out/'report.json').read_text()
                self.assertNotIn('SECRET_FIXTURE',report)
                self.assertEqual(json.loads(report)['status'],'submission_failed_or_unknown')
                with self.assertRaises(FileExistsError):pilot.run(out,True)
                self.assertEqual(request.call_count,1)


if __name__=='__main__':unittest.main()
