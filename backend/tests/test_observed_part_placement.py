"""Tiny math/ownership diagnostics, not video-generation validation."""
import unittest
import numpy as np
from real_video.observed_part_placement import (associate_observations, partition_replacement_masks,
                                               fit_observed_placement, mask_iou, evaluate_replacement_coverage)


class ObservedPlacementTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((40,60),bool);self.mask[15:30,20:35]=True
        self.depth = np.ones((40,60),np.float32)*3
        self.points = np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.25,.25)])

    def test_fit_follows_observed_image_motion_and_bottom(self):
        first=fit_observed_placement(self.points,self.mask,self.depth,focal=50.,principal=(30.,20.))
        moved=np.zeros_like(self.mask);moved[10:25,30:45]=True
        second=fit_observed_placement(self.points,moved,self.depth,focal=50.,principal=(30.,20.))
        self.assertLess(first['bottom_error_pixels'],.01)
        self.assertLess(second['bottom_error_pixels'],.01)
        self.assertLess(second['translation'][0],first['translation'][0])
        self.assertGreater(second['translation'][1],first['translation'][1])
        self.assertFalse(second['physical_contact_verified'])

    def test_depth_is_front_surface_not_center(self):
        fit=fit_observed_placement(self.points,self.mask,self.depth,focal=50.,principal=(30.,20.))
        xyz=self.points*fit['scale']+np.array(fit['translation'])
        self.assertAlmostEqual(xyz[:,2].min(),3.)
        self.assertGreater(np.mean(xyz[:,2]),3.)

    def test_missing_parts_preserved_and_ambiguous_overlap_withheld(self):
        piece=np.zeros_like(self.mask);piece[28:32,30:37]=True
        protected=np.zeros_like(self.mask);protected[15:17,20:23]=True
        result=partition_replacement_masks({'body':self.mask,'slice':piece}, {'body'}, protected)
        self.assertFalse(np.any(result['remove'] & piece))
        self.assertFalse(np.any(result['remove'] & protected))
        self.assertTrue(np.all(result['preserved'][piece]))
        self.assertEqual(result['deferred_ids'],('slice',))

    def test_association_retains_identity_but_refuses_duplicate_ambiguity(self):
        previous=[dict(part_id='part_7',label='fruit',mask=self.mask)]
        current=[dict(label='fruit',mask=self.mask.copy())]
        self.assertEqual(associate_observations(previous,current),{0:'part_7'})
        self.assertEqual(associate_observations(previous,current+current),{})
        self.assertEqual(associate_observations(previous,[dict(label='peel',mask=self.mask)]),{})
        self.assertEqual(mask_iou(self.mask,self.mask),1.)

    def test_invalid_depth_shape_masks_fail_closed(self):
        for depth in (self.depth*0,self.depth*np.nan):
            with self.assertRaises(ValueError):fit_observed_placement(self.points,self.mask,depth,focal=50.,principal=(30.,20.))
        depth=self.depth.copy();depth[15:23,20:35]=10.
        with self.assertRaises(ValueError):fit_observed_placement(self.points,self.mask,depth,focal=50.,principal=(30.,20.))
        with self.assertRaises(ValueError):partition_replacement_masks({'body':self.mask},{'missing'},np.zeros_like(self.mask))

    def test_empty_or_thin_support_rejected_before_depth_reductions(self):
        import warnings
        from unittest.mock import patch
        empty=np.zeros_like(self.mask)
        single=empty.copy();single[20,25]=True
        thin=empty.copy();thin[20,20:30]=True
        # Regression for remote v3: no warning/median/quantile call on empty
        # eligibility after protected subtraction, and a catchable ValueError.
        for mask in (empty,single,thin):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with patch('numpy.median',side_effect=AssertionError('depth reduction too early')), \
                        patch('numpy.quantile',side_effect=AssertionError('depth reduction too early')):
                    with self.assertRaisesRegex(ValueError,'Insufficient observed mask extent'):
                        fit_observed_placement(self.points,mask,self.depth,focal=50.,principal=(30.,20.))

    def test_fully_protected_or_duplicate_parts_preserve_source_and_report_counts(self):
        protected=partition_replacement_masks({'body':self.mask},{'body'},self.mask)
        self.assertFalse(protected['eligible']['body'].any())
        self.assertEqual(protected['instance_pixel_counts']['body']['eligible_pixels'],0)
        self.assertEqual(protected['instance_pixel_counts']['body']['protected_overlap_pixels'],int(self.mask.sum()))
        duplicates=partition_replacement_masks({'a':self.mask,'b':self.mask},{'a','b'},np.zeros_like(self.mask))
        self.assertFalse(duplicates['remove'].any())
        self.assertEqual(duplicates['instance_pixel_counts']['a']['ambiguous_overlap_pixels'],int(self.mask.sum()))
        with self.assertRaises(ValueError):
            fit_observed_placement(self.points,duplicates['eligible']['a'],self.depth,focal=50.,principal=(30.,20.))

    def test_worker_does_not_equate_generic_orange_with_a_verified_body(self):
        from cloud.object_edit_tracked_worker import label_kind
        self.assertEqual(label_kind('orange fruit.'),'whole_fruit')
        for label in ('orange slice','orange segment','orange peel'):
            self.assertEqual(label_kind(label),'unsupported_piece')
        for label in ('orange','orange fruit orange peel','red apple'):
            self.assertEqual(label_kind(label),'unknown')
        self.assertEqual(label_kind('donkey'),'protected')

    def test_remote_gpu_gate_refuses_busy_lock_and_existing_compute(self):
        from unittest.mock import patch, MagicMock, mock_open
        from types import SimpleNamespace
        from cloud.object_edit_tracked_worker import exclusive_gpu
        fake=SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=MagicMock())
        # No local nvidia-smi invocation; this tests control flow with mocks.
        with patch.dict('sys.modules',{'fcntl':fake}), patch('builtins.open',mock_open()), \
                patch('subprocess.run',return_value=SimpleNamespace(stdout='1234\n')) as run:
            with self.assertRaises(RuntimeError):
                with exclusive_gpu():self.fail('Busy device was entered')
            self.assertEqual(run.call_args.kwargs['timeout'],10)
        fake.flock.side_effect=BlockingIOError
        with patch.dict('sys.modules',{'fcntl':fake}), patch('builtins.open',mock_open()), patch('subprocess.run') as run:
            with self.assertRaises(RuntimeError):
                with exclusive_gpu():self.fail('Busy lock was entered')
            run.assert_not_called()
        fake.flock.side_effect=None
        with patch.dict('sys.modules',{'fcntl':fake}), patch('builtins.open',mock_open()), \
                patch('subprocess.run',return_value=SimpleNamespace(stdout='')):
            with exclusive_gpu():pass

    def test_independent_queries_keep_unknown_protected_and_provenance(self):
        from unittest.mock import MagicMock
        from cloud.object_edit_tracked_worker import independent_grounding, GROUNDING_QUERIES
        class Tensor:
            def __init__(self,value):self.value=np.array(value)
            def cpu(self):return self
            def numpy(self):return self.value
        class Inputs(dict):
            input_ids='tokens'
            def to(self,device):return self
        processor=MagicMock(return_value=Inputs())
        labels=['orange fruit','orange peel','orange slice orange segment','fruit','donkey','wooden bowl']
        processor.post_process_grounded_object_detection.side_effect=[
            [dict(boxes=Tensor([[20,20,30,30]]),scores=Tensor([.5]),text_labels=[label])] for label in labels]
        model=MagicMock()
        rows=independent_grounding('image',processor,model,100,100)
        self.assertEqual([c.kwargs['text'] for c in processor.call_args_list],list(GROUNDING_QUERIES))
        self.assertEqual([r['kind'] for r in rows],['whole_fruit','unsupported_piece','unknown','unknown','protected','protected'])
        self.assertEqual([r['grounding_query'] for r in rows],list(GROUNDING_QUERIES))
        self.assertEqual(model.call_count,6)

    def test_coverage_gate_preserves_source_on_holes_or_spill(self):
        alpha=self.mask.astype(np.float32)
        alpha[15:20,20:35]=0
        result=evaluate_replacement_coverage(alpha,self.mask,np.zeros_like(self.mask))
        self.assertFalse(result['accepted_for_insertion']);self.assertFalse(result['remove_mask'].any())
        alpha=np.ones_like(self.mask,dtype=np.float32)
        result=evaluate_replacement_coverage(alpha,self.mask,np.zeros_like(self.mask))
        self.assertFalse(result['accepted_for_insertion'])
        self.assertGreater(result['outside_alpha_fraction'],.05)

    def test_coverage_gate_protects_actor_and_only_removes_opaque_pixels(self):
        alpha=self.mask.astype(np.float32)
        protected=np.zeros_like(self.mask);protected[20,25]=True
        result=evaluate_replacement_coverage(alpha,self.mask,protected)
        self.assertFalse(result['accepted_for_insertion'])
        alpha[15,20]=.8
        result=evaluate_replacement_coverage(alpha,self.mask,np.zeros_like(self.mask))
        self.assertTrue(result['accepted_for_insertion'])
        self.assertFalse(result['remove_mask'][15,20])
        self.assertEqual(result['retained_target_pixels'],1)
        self.assertFalse(result['quality_accepted'])


if __name__=='__main__':unittest.main()
