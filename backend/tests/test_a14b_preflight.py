"""Dependency startup contracts; no checkpoint loading, CUDA or network use."""
import types
import unittest
from unittest.mock import patch

from cloud import a14b_preflight as preflight


class A14BPreflightTests(unittest.TestCase):
    def setUp(self):
        self.imports = []
        self.pipeline = types.SimpleNamespace(prompt_clean=lambda text: preflight.PROBE_EXPECTED)
        self.stack = []
        for name, replacement in (
            ('find_spec', lambda name: object()),
            ('version', lambda name: 'test-version'),
            ('import_module', self.importer),
        ):
            patcher = patch.object(preflight, name, replacement)
            patcher.start(); self.stack.append(patcher)
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(self.stack)])

    def importer(self, name):
        self.imports.append(name)
        if name == preflight.PROMPT_MODULE:
            return self.pipeline
        return types.SimpleNamespace()

    def test_success_runs_actual_prompt_probe_without_model_methods(self):
        observed = []
        self.pipeline.prompt_clean = lambda text: observed.append(text) or preflight.PROBE_EXPECTED
        result = preflight.require_dependencies()
        self.assertTrue(result['ready'])
        self.assertEqual(observed, [preflight.PROBE_INPUT])
        self.assertEqual(self.imports, ['ftfy','regex','sentencepiece','imageio_ffmpeg',preflight.PROMPT_MODULE])
        self.assertFalse(result['pretrained_load_called'])
        self.assertFalse(result['downloads_attempted'])
        self.assertFalse(result['gpu_work_attempted'])

    def test_missing_ftfy_fails_before_any_pipeline_import(self):
        with patch.object(preflight,'find_spec',lambda name: None if name == 'ftfy' else object()):
            with self.assertRaises(preflight.DependencyPreflightError) as caught:
                preflight.require_dependencies()
        self.assertEqual(self.imports, [])
        self.assertIn('ftfy',str(caught.exception))
        self.assertFalse(caught.exception.report['ready'])
        self.assertEqual(caught.exception.report['prompt_probe']['status'],'skipped_missing_dependencies')

    def test_collects_multiple_missing_modules(self):
        with patch.object(preflight,'find_spec',lambda name: None if name in ('ftfy','sentencepiece') else object()):
            report = preflight.check_dependencies()
        self.assertEqual([entry['module'] for entry in report['checks'] if not entry['present']],
                         ['ftfy','sentencepiece'])
        self.assertEqual(self.imports, [])

    def test_broken_find_spec_fails_closed(self):
        def broken(name):
            if name == 'diffusers': raise ValueError('broken import spec')
            return object()
        with patch.object(preflight,'find_spec',broken):
            result = preflight.check_dependencies()
        item = next(item for item in result['checks'] if item['module']=='diffusers')
        self.assertFalse(result['ready'])
        self.assertEqual(item['discovery_error_type'],'ValueError')
        self.assertEqual(self.imports, [])

    def test_optional_import_breakage_stops_before_pipeline(self):
        def broken(name):
            self.imports.append(name)
            raise ImportError('private/path/token must not be printed')
        with patch.object(preflight,'import_module',broken):
            result = preflight.check_dependencies()
        self.assertFalse(result['ready'])
        self.assertEqual(result['prompt_probe']['error_type'],'ImportError')
        self.assertEqual(self.imports,['ftfy'])
        self.assertNotIn('private/path',str(result))

    def test_successful_import_is_not_prompt_success(self):
        def missing_ftfy(text): raise NameError("name 'ftfy' is not defined")
        self.pipeline.prompt_clean = missing_ftfy
        with self.assertRaises(preflight.DependencyPreflightError) as caught:
            preflight.require_dependencies()
        self.assertEqual(caught.exception.report['prompt_probe']['error_type'],'NameError')

    def test_changed_prompt_contract_rejected(self):
        self.pipeline.prompt_clean = lambda text: text
        result = preflight.check_dependencies()
        self.assertFalse(result['ready'])
        self.assertEqual(result['prompt_probe']['error_type'],'UnexpectedPromptCleanOutput')

    def test_missing_prompt_function_rejected(self):
        self.pipeline = types.SimpleNamespace()
        result = preflight.check_dependencies()
        self.assertEqual(result['prompt_probe']['error_type'],'AttributeError')

    def test_metadata_absence_does_not_fake_missing_import(self):
        def no_metadata(name): raise preflight.PackageNotFoundError(name)
        with patch.object(preflight,'version',no_metadata):
            result = preflight.check_dependencies()
        self.assertTrue(result['ready'])
        self.assertTrue(all(item['version'] is None and item['present'] for item in result['checks']))

    def test_distribution_names_are_not_import_names(self):
        result = preflight.check_dependencies()
        pillow = next(item for item in result['checks'] if item['module']=='PIL')
        self.assertEqual(pillow['distribution'],'Pillow')
        self.assertEqual(pillow['version'],'test-version')

    def test_runner_checks_dependencies_before_model_validation(self):
        from cloud import wan_a14b_experiment as experiment
        args=types.SimpleNamespace(frames=33,steps=20,train_steps=2,wall_seconds=600)
        with patch.object(preflight,'require_dependencies',side_effect=RuntimeError('dependency sentinel')), \
             patch.object(experiment,'validate_model') as validate:
            with self.assertRaisesRegex(RuntimeError,'dependency sentinel'):
                experiment.run(args)
        validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
