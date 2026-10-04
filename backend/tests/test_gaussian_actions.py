import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from real_video import generate_gaussian_actions as actions
from real_video.asset_readiness import file_identity
from real_video.generate_gaussian_actions import action_control


class ActionControlTests(unittest.TestCase):
    def test_stroll_stop_schedule(self):
        self.assertEqual(action_control(0,.5),('Walk',.5))
        self.assertEqual(action_control(19,.5),('Walk',.5))
        self.assertEqual(action_control(29,.5),('Walk',0.))
        self.assertEqual(action_control(30,.5),('Idle',0.))
        self.assertEqual(action_control(69,.5),('Idle',0.))
        deceleration=[action_control(i,.5)[1] for i in range(19,31)]
        self.assertTrue(all(a>=b for a,b in zip(deceleration,deceleration[1:])))

    def test_invalid_control(self):
        for index,speed in ((-1,.5),(1,-1),(1,5),(1.5,.5)):
            with self.assertRaises(ValueError): action_control(index,speed)


class RendererDispatchTests(unittest.TestCase):
    def render(self, renderer):
        return actions.render_action_frame(None,None,None,None,None,None,
            height=10,width=20,ground=False,renderer=renderer,radius=5)

    def test_legacy_remains_default_and_receives_radius(self):
        with patch.object(actions,'legacy_render',return_value=('rgb','alpha')) as render:
            self.assertEqual(self.render('legacy'),('rgb','alpha'))
            self.assertEqual(render.call_args.kwargs['radius'],5)
            self.assertNotIn('fragment_budget',render.call_args.kwargs)

    def test_adaptive_receives_bounded_configuration_without_radius(self):
        with patch('real_video.adaptive_gaussian_renderer.render',return_value=('rgb','alpha')) as render:
            self.assertEqual(self.render('adaptive'),('rgb','alpha'))
            kw=render.call_args.kwargs
            self.assertNotIn('radius',kw)
            self.assertNotIn('fixed_radius',kw)
            for key,value in actions.ADAPTIVE_LIMITS.items(): self.assertEqual(kw[key],value)

    def test_budget_failure_is_not_hidden_by_legacy_fallback(self):
        with (patch('real_video.adaptive_gaussian_renderer.render',side_effect=RuntimeError('budget exceeded')),
              patch.object(actions,'legacy_render') as legacy):
            with self.assertRaisesRegex(RuntimeError,'budget exceeded'): self.render('adaptive')
            legacy.assert_not_called()

    def test_unknown_renderer_rejected(self):
        with self.assertRaises(ValueError): self.render('unbounded')


class ActionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_rejected_report_blocks_before_motion_output_or_cuda(self):
        bad=self.root/'bad.json'; bad.write_text('{"schema_version":1}')
        output=self.root/'new_run'
        with (patch.object(actions,'generate_motion') as motion, patch.object(actions,'make_asset') as asset,
              patch('torch.cuda.reset_peak_memory_stats') as cuda):
            with self.assertRaisesRegex(RuntimeError,'preflight blocked'):
                actions.run(output,cat_readiness=bad,dog_readiness=bad)
            motion.assert_not_called(); asset.assert_not_called(); cuda.assert_not_called()
        self.assertFalse(output.exists())

    def fixture(self):
        # Explicit mocks of upstream readiness: test this runner's file-binding
        # and rig-selection logic, not anatomy or checkpoint serialization.
        paths={name:self.root/(name+'.asset') for name in ('cat','dog')}
        mesh=self.root/'cat.mesh'; mesh.write_bytes(b'cat mesh')
        reports={}; rigs={}; rig_paths={}
        for name,path in paths.items():
            path.write_bytes(name.encode()); rig_path=self.root/(name+'.rig')
            rig_path.write_bytes((name+' rig').encode()); rig_paths[name]=rig_path
            parents=actions.CAT_PARENTS if name=='cat' else actions.DOG_PARENTS
            rigs[str(rig_path)]=dict(parents=list(parents),joints=torch.zeros(len(parents),3))
            if name=='dog': rigs[str(rig_path)].update(source_to_asset=torch.eye(3),paw_sole_anchors=torch.zeros(4,3),floor=0.)
            report=dict(inputs={'asset':file_identity(path),'mesh':file_identity(mesh if name=='cat' else path),
                                'rig':file_identity(rig_path)},implementation={'fixture':True})
            reports[name]=self.root/(name+'.json'); reports[name].write_text(json.dumps(report))
        return paths,mesh,reports,rigs,rig_paths

    def test_loads_approved_rigs_instead_of_rebuilding(self):
        paths,mesh,reports,rigs,_=self.fixture()
        with (patch.object(actions,'ASSETS',paths),patch.object(actions,'CAT_MESH',mesh),
             patch.object(actions,'require_motion_ready',return_value=True),
             patch.object(actions,'load_verified',side_effect=lambda p:rigs[p]) as loader):
            approved,provenance=actions.preflight_assets(reports['cat'],reports['dog'])
        self.assertEqual(loader.call_count,2)
        self.assertIs(approved['cat'],rigs[str(self.root/'cat.rig')])
        self.assertEqual(provenance['dog']['inputs']['asset'],file_identity(paths['dog']))

    def test_other_mesh_approval_does_not_approve_runner_mesh(self):
        paths,mesh,reports,rigs,_=self.fixture()
        report=json.loads(reports['cat'].read_text()); report['inputs']['mesh']=file_identity(paths['dog'])
        reports['cat'].write_text(json.dumps(report))
        with (patch.object(actions,'ASSETS',paths),patch.object(actions,'CAT_MESH',mesh),
             patch.object(actions,'require_motion_ready',return_value=True),patch.object(actions,'load_verified') as loader):
            with self.assertRaisesRegex(RuntimeError,'asset/mesh'):
                actions.preflight_assets(reports['cat'],reports['dog'])
            loader.assert_not_called()

    def test_approved_but_incompatible_rig_cannot_reach_retargeter(self):
        paths,mesh,reports,rigs,_=self.fixture()
        rigs[str(self.root/'cat.rig')]['parents']=[-1,0]
        with (patch.object(actions,'ASSETS',paths),patch.object(actions,'CAT_MESH',mesh),
             patch.object(actions,'require_motion_ready',return_value=True),
             patch.object(actions,'load_verified',side_effect=lambda p:rigs[p])):
            with self.assertRaisesRegex(RuntimeError,'topology'):
                actions.preflight_assets(reports['cat'],reports['dog'])

    def test_rig_change_during_load_is_rejected(self):
        paths,mesh,reports,rigs,_=self.fixture()
        def changed(p):
            Path(p).write_bytes(b'changed during load')
            return rigs[p]
        with (patch.object(actions,'ASSETS',paths),patch.object(actions,'CAT_MESH',mesh),
             patch.object(actions,'require_motion_ready',return_value=True),
             patch.object(actions,'load_verified',side_effect=changed)):
            with self.assertRaisesRegex(RuntimeError,'changed during loading'):
                actions.preflight_assets(reports['cat'],reports['dog'])

    def test_dog_adapter_basis_is_validated_on_cpu(self):
        paths,mesh,reports,rigs,_=self.fixture()
        rigs[str(self.root/'dog.rig')]['source_to_asset']=torch.zeros(3,3)
        with (patch.object(actions,'ASSETS',paths),patch.object(actions,'CAT_MESH',mesh),
             patch.object(actions,'require_motion_ready',return_value=True),
             patch.object(actions,'load_verified',side_effect=lambda p:rigs[p])):
            with self.assertRaisesRegex(RuntimeError,'basis is invalid'):
                actions.preflight_assets(reports['cat'],reports['dog'])
