import unittest
from real_video.elimination_scope import plan_elimination


def node(parent=None,relation='unrelated',protected=False):
    return dict(parent=parent,relation=relation,protected=protected,binding_verified=True,revision=0)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.inventory={'orange':node(),'peel':node('orange','detached_from'),
            'pith':node('peel','part_of'),'shadow':node('orange','effect_of'),
            'bowl':node('orange','supports',True),'donkey':node('orange','contact_with',True),
            'orange_flower':node(protected=True)}
        self.proposal=dict(candidates=[],unknown_parts=[],reviewed_frames=[0,20,39])
    def plan(self):return plan_elimination('orange',self.inventory,self.proposal,[0,20,39])
    def test_transitive_parts_and_receivers(self):
        result=self.plan()
        self.assertEqual(result['remove'],['orange','peel','pith'])
        self.assertEqual(result['rerender'],['bowl','donkey','shadow'])
        self.assertIn('orange_flower',result['preserve'])
        self.assertTrue(result['scope_verified']);self.assertFalse(result['executable'])
    def test_protected_part_blocks(self):
        self.inventory['peel']['protected']=True
        self.assertFalse(self.plan()['scope_verified'])
    def test_unknown_residue_and_temporal_omission_block(self):
        self.proposal['unknown_parts']=['segment in mouth'];self.proposal['reviewed_frames']=[0]
        self.assertGreaterEqual(len(self.plan()['blockers']),2)
    def test_model_cannot_claim_other_object(self):
        self.proposal['candidates']=[dict(entity_id='orange_flower',relation='part_of',confidence=.99,frames=[0],reason='Orange color')]
        result=self.plan();self.assertNotIn('orange_flower',result['remove']);self.assertFalse(result['scope_verified'])
    def test_nan_unknown_fields_and_uncertainty(self):
        item=dict(entity_id='peel',relation='part_of',confidence=.5,frames=[0],reason='Ambiguous')
        self.proposal['candidates']=[item];self.assertFalse(self.plan()['scope_verified'])
        item['confidence']=float('nan')
        with self.assertRaises(ValueError):self.plan()
        self.proposal['candidates']=[];self.proposal['execute']='delete everything'
        with self.assertRaises(ValueError):self.plan()


if __name__=='__main__':unittest.main()
