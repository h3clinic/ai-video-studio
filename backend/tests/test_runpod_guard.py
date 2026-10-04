from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cloud.runpod_guarded_session import launch, restore
from cloud.runpod_control import request


class GuardTests(unittest.TestCase):
    def test_invalid_bounds_never_contact_provider(self):
        with patch('cloud.runpod_guarded_session.api') as api:
            for seconds,cost in ((0,1),(2200,3),(60,4),(60,0)):
                with self.assertRaises(ValueError): launch(Path('unused'),seconds,cost)
            api.assert_not_called()

    def test_reject_running_or_expensive_or_custom_command(self):
        base=dict(desiredStatus='EXITED',costPerHr=3.49,dockerEntrypoint=[],dockerStartCmd=[])
        for change in (dict(desiredStatus='RUNNING'),dict(costPerHr=5),
                       dict(dockerEntrypoint=['custom']),dict(dockerStartCmd=['custom'])):
            with self.subTest(change=change), patch('cloud.runpod_guarded_session.api',return_value=base|change) as api:
                with self.assertRaises(RuntimeError): launch(Path('unused'),60,.2)
                api.assert_called_once_with()

    def test_computed_cost_bound(self):
        with patch('cloud.runpod_guarded_session.api',return_value=dict(desiredStatus='EXITED',costPerHr=3.49)) as api:
            with self.assertRaises(RuntimeError): launch(Path('unused'),2100,.2)
            api.assert_called_once_with()

    def test_restore_only_stopped(self):
        with patch('cloud.runpod_guarded_session.api',return_value={'desiredStatus':'RUNNING'}) as api:
            with self.assertRaises(RuntimeError):restore()
            api.assert_called_once_with()

    def test_status_redacts_environment(self):
        from cloud.runpod_control import POD
        with patch('cloud.runpod_control.api',return_value=dict(id=POD,env={'SECRET':'do-not-output'})):
            self.assertNotIn('env',request())

    def test_status_is_bound_to_one_pod(self):
        with patch('cloud.runpod_control.api',return_value=dict(id='wrong')):
            with self.assertRaises(RuntimeError):request()

    def test_unsupported_operation(self):
        with self.assertRaises(ValueError):request('delete')


if __name__=='__main__':unittest.main()
