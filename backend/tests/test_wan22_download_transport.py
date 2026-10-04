"""Transport entry checks never need live Hub requests or GPU allocation."""
import unittest
from unittest.mock import patch

from real_video import prepare_wan22 as prepare


class Wan22DownloadTransportTests(unittest.TestCase):
    def test_xet_installed_gate_reaches_mocked_metadata(self):
        with patch('huggingface_hub.utils._runtime.is_xet_available', return_value=True), \
                patch.object(prepare, 'HfApi') as api:
            api.return_value.model_info.side_effect = RuntimeError('mock metadata boundary')
            with self.assertRaisesRegex(RuntimeError, 'mock metadata boundary'):
                prepare.run('xet')
            api.return_value.model_info.assert_called_once_with(
                prepare.REPO, revision=prepare.REVISION, files_metadata=True)

    def test_xet_disabled_fails_before_network(self):
        with patch('huggingface_hub.utils._runtime.is_xet_available', return_value=False), \
                patch.object(prepare, 'HfApi') as api:
            with self.assertRaisesRegex(RuntimeError, 'installed and enabled'):
                prepare.run('xet')
            api.assert_not_called()

    def test_unknown_transport_never_starts_network(self):
        with patch.object(prepare, 'HfApi') as api:
            with self.assertRaisesRegex(ValueError, 'Explicit ranges or official Xet'):
                prepare.run('anything_else')
            api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
