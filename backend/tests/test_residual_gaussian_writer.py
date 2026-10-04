"""CPU operator/appearance-fit contracts, not generative-quality tests."""
import unittest

import torch

from real_video.gaussian_latent_memory import GaussianLatentMemory
from real_video.residual_gaussian_writer import SparseAlphaOperator, write_residual


def cache_from_dense(matrix):
    pixels,rows = torch.nonzero(matrix,as_tuple=True)
    return dict(pixel=pixels,ids=rows,weight=matrix[pixels,rows])


class ResidualGaussianWriterTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([71,19,400])
        self.prior = torch.tensor([[.4,-.7],[1.2,.3],[3.,-2.]])
        self.memory = GaussianLatentMemory(self.ids,2,'a'*64,initial_features=self.prior)
        self.matrix = torch.tensor([[.7,.2,0.],[.1,.8,0.]])
        self.cache = cache_from_dense(self.matrix)
        self.target = torch.tensor([[[.8,-.2]],[[.1,.7]]])
        self.q = torch.ones(1,2)

    def write(self,**kwargs):
        return write_residual(self.memory,self.target,self.cache,
            ids=self.ids,confidence=self.q,**kwargs)

    def test_adjoint_and_duplicate_fragment_diagonal(self):
        cache = dict(pixel=torch.tensor([0,0,1,0]),ids=torch.tensor([0,0,1,1]),
                     weight=torch.tensor([.2,.3,.7,.1]))
        op = SparseAlphaOperator(cache,2,2,chunk_size=1)
        x=torch.tensor([[.4,-.2],[1.1,.9]],dtype=torch.float64)
        y=torch.tensor([[.3,.6],[-.8,1.4]],dtype=torch.float64)
        torch.testing.assert_close((op.forward(x)*y).sum(),(x*op.adjoint(y)).sum())
        dense=torch.tensor([[.5,.1],[0.,.7]],dtype=torch.float64)
        torch.testing.assert_close(op.forward(x),dense@x)
        torch.testing.assert_close(op.diagonal(torch.ones(2)),dense.square().sum(0))

    def test_matches_dense_regularized_solution(self):
        report=self.write(regularization=.05,trust_radius=8.,max_iterations=12,tolerance=1e-10)
        a=self.matrix.double(); prior=self.prior.double()
        target=self.target.permute(1,2,0).reshape(2,2).double()
        delta=torch.linalg.solve(a.T@a+.05*torch.eye(3,dtype=torch.float64),a.T@(target-a@prior))
        torch.testing.assert_close(self.memory.snapshot()['features'].double(),prior+delta,rtol=1e-5,atol=1e-6)
        self.assertTrue(report['committed'])
        self.assertFalse(report['generator_weights_trained'])
        history=report['objective_history']
        self.assertTrue(all(b<=a for a,b in zip(history,history[1:])))

    def test_zero_residual_repeated_writes_are_exact_state_noops(self):
        self.target=self.memory.read(self.cache,1,2,ids=self.ids)['features']
        old=self.memory.snapshot()
        for _ in range(8):
            report=self.write()
            self.assertFalse(report['committed'])
        now=self.memory.snapshot()
        for key in ('features','observation_mass','ids'):
            self.assertTrue(torch.equal(old[key],now[key]))
        self.assertEqual(old['writes'],now['writes'])

    def test_hidden_and_excluded_ids_remain_exact(self):
        self.write(writable=torch.tensor([True,False,True]))
        after=self.memory.snapshot()
        self.assertTrue(torch.equal(after['features'][1:],self.prior[1:]))
        self.assertTrue(torch.equal(after['observation_mass'][1:],torch.zeros(2)))

    def test_composed_background_readback_is_exact_noop(self):
        background=torch.ones_like(self.target)
        self.target=self.memory.read(self.cache,1,2,ids=self.ids)['features']+background
        before=self.memory.snapshot()
        for _ in range(20):
            self.assertFalse(self.write(background=background)['committed'])
        after=self.memory.snapshot()
        for key in ('features','observation_mass','ids'):
            self.assertTrue(torch.equal(before[key],after[key]))
        self.assertEqual(before['writes'],after['writes'])

    def test_zero_confidence_and_validity_do_not_write(self):
        for use_validity in (False,True):
            self.q=torch.ones(1,2) if use_validity else torch.zeros(1,2)
            report=self.write(validity=torch.zeros(1,2,dtype=torch.bool) if use_validity else None)
            self.assertFalse(report['committed'])
            self.assertTrue(torch.equal(self.memory.snapshot()['features'],self.prior))

    def test_explicit_background_and_trust_limit(self):
        report=self.write(background=torch.full_like(self.target,.1),trust_radius=.03)
        change=(self.memory.snapshot()['features']-self.prior).norm(dim=1).max()
        self.assertLessEqual(float(change),.0300001)
        self.assertLessEqual(report['final_objective'],report['initial_objective'])

    def test_low_opacity_is_regularized_not_an_unbounded_inverse(self):
        self.matrix=torch.tensor([[.0001,0.,0.],[0.,.0001,0.]])
        self.cache=cache_from_dense(self.matrix)
        report=self.write(regularization=.001,trust_radius=.05)
        self.assertLessEqual(report['max_feature_update_norm'],.0500001)
        self.assertTrue(torch.isfinite(self.memory.snapshot()['features']).all())

    def test_invalid_inputs_rejected_before_state_mutation(self):
        for kwargs in ({'regularization':0.},{'trust_radius':float('nan')},
                       {'max_iterations':100},{'writable':torch.ones(3)}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.write(**kwargs)
        with self.assertRaises(ValueError):
            write_residual(self.memory,self.target,self.cache,ids=self.ids.flip(0),confidence=self.q)
        bad=dict(self.cache);bad['weight']=torch.ones_like(self.cache['weight'])
        with self.assertRaises(ValueError):
            write_residual(self.memory,self.target,bad,ids=self.ids,confidence=self.q)
        self.assertTrue(torch.equal(self.memory.snapshot()['features'],self.prior))
        self.assertEqual(self.memory.writes,0)


if __name__=='__main__':
    unittest.main()
