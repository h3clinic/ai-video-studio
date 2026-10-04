"""Diagnostic contracts only; not evidence of perceptual model success."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from real_video.train_latent_appearance import future_metrics, resource_guard


class AppearanceTrainingTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(47)
        clean=torch.randn(1,2,5,4,6)
        noise=torch.randn_like(clean)
        sigma=.6
        return clean,noise,sigma,(1-sigma)*clean+sigma*noise

    def test_correct_flow_recovers_future(self):
        clean,noise,sigma,noisy=self.fixture()
        result=future_metrics(noise-clean,noisy,clean,noise,sigma)
        self.assertTrue(all(value<1e-12 for value in result.values()))

    def test_anchor_error_cannot_count_as_future_improvement(self):
        clean,noise,sigma,noisy=self.fixture()
        pred=noise-clean
        pred[:,:,0]+=100
        result=future_metrics(pred,noisy,clean,noise,sigma)
        self.assertTrue(all(value<1e-12 for value in result.values()))

    def test_future_spatial_error_is_measured(self):
        clean,noise,sigma,noisy=self.fixture()
        pred=noise-clean
        pred[:,:,1:,:,::2]+=1
        result=future_metrics(pred,noisy,clean,noise,sigma)
        self.assertTrue(all(value>0 for value in result.values()))

    def test_resource_preflight_rejects_low_ram(self):
        with patch('psutil.virtual_memory',return_value=SimpleNamespace(available=4*2**30)), \
             patch('psutil.sensors_battery',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'8 GiB'):
                resource_guard(Path('not_created_by_test'),600)

    def test_resource_preflight_rejects_low_unplugged_battery(self):
        with patch('psutil.virtual_memory',return_value=SimpleNamespace(available=12*2**30)), \
             patch('psutil.sensors_battery',return_value=SimpleNamespace(power_plugged=False,percent=10)):
            with self.assertRaisesRegex(RuntimeError,'Low battery'):
                resource_guard(Path('not_created_by_test'),600)


if __name__=='__main__': unittest.main()
