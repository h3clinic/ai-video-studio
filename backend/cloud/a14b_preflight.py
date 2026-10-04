"""CPU-only dependency checks before loading the large Wan A14B checkpoints.

Wan's prompt cleaner imports ftfy conditionally but invokes it during prompt
encoding. A successful pipeline import therefore does not prove prompt encoding
is ready. This check exercises the installed prompt-cleaning function on a fixed
literal before any pretrained loading. It never downloads, installs, loads model
weights, allocates tensors, invokes CUDA, or starts paid compute.

Passing is dependency evidence only: it does not guarantee GPU compatibility,
memory sufficiency, tokenizer/model loading, video generation, or quality.
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json


# Import names are distinct from distribution names (notably PIL/Pillow).
REQUIRED = (
    ('ftfy', 'ftfy'),
    ('regex', 'regex'),
    ('sentencepiece', 'sentencepiece'),
    ('torch', 'torch'),
    ('diffusers', 'diffusers'),
    ('transformers', 'transformers'),
    ('accelerate', 'accelerate'),
    ('huggingface_hub', 'huggingface_hub'),
    ('safetensors', 'safetensors'),
    ('numpy', 'numpy'),
    ('PIL', 'Pillow'),
    ('imageio', 'imageio'),
    ('imageio_ffmpeg', 'imageio-ffmpeg'),
    ('psutil', 'psutil'),
)
PROMPT_MODULE = 'diffusers.pipelines.wan.pipeline_wan_i2v'
PROBE_INPUT = '  A donkey\n eats an &amp;amp; orange.  '
PROBE_EXPECTED = 'A donkey eats an & orange.'


class DependencyPreflightError(RuntimeError):
    def __init__(self, report):
        self.report = report
        failures = [item['module'] for item in report['checks'] if not item['present']]
        probe = report['prompt_probe']
        reason = (', '.join(failures) if failures else
                  'installed Wan prompt cleaning failed (' + probe.get('error_type', 'unexpected output') + ')')
        super().__init__('A14B dependency preflight failed before model loading: ' + reason)


def check_dependencies():
    """Return a structured report; collect all missing names before imports.

    Package versions are recorded, not silently installed or upgraded. A missing
    distribution version alone is not a failed import (editable/vendor installs
    can still be valid). Broken import discovery does fail closed.
    """
    report = dict(schema_version=1, scope='CPU dependency and native prompt-cleaning smoke check',
                  ready=False, checks=[], prompt_probe=dict(status='not_run'),
                  pretrained_load_called=False, downloads_attempted=False,
                  gpu_work_attempted=False)
    for module, distribution in REQUIRED:
        check = dict(module=module, distribution=distribution, present=False, version=None)
        try:
            check['present'] = find_spec(module) is not None
        except Exception as error:
            check['discovery_error_type'] = type(error).__name__
        try:
            check['version'] = version(distribution)
        except PackageNotFoundError:
            check['version_status'] = 'distribution_metadata_unavailable'
        except Exception as error:
            check['version_error_type'] = type(error).__name__
        report['checks'].append(check)
    if any(not item['present'] for item in report['checks']):
        report['prompt_probe']['status'] = 'skipped_missing_dependencies'
        return report
    try:
        # Catch binary/import breakage in optional components that the native
        # pipeline may otherwise defer until prompt or video I/O preparation.
        for name in ('ftfy', 'regex', 'sentencepiece', 'imageio_ffmpeg'):
            import_module(name)
        pipeline_module = import_module(PROMPT_MODULE)
        cleaner = getattr(pipeline_module, 'prompt_clean')
        observed = cleaner(PROBE_INPUT)
        if observed != PROBE_EXPECTED:
            report['prompt_probe'] = dict(status='failed', error_type='UnexpectedPromptCleanOutput')
            return report
        report['prompt_probe'] = dict(status='passed', function=PROMPT_MODULE+'.prompt_clean')
    except Exception as error:
        # Record the exception type, not arbitrary dependency error text which
        # could contain account, environment or path details.
        report['prompt_probe'] = dict(status='failed', error_type=type(error).__name__)
        return report
    report['ready'] = True
    return report


def require_dependencies():
    """Fail before a caller invokes from_pretrained; return evidence on success."""
    report = check_dependencies()
    if not report['ready']:
        raise DependencyPreflightError(report)
    return report


if __name__ == '__main__':
    result = check_dependencies()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
