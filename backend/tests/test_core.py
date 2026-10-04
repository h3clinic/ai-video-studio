import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from gv.geometry import exp_so3,frame_error
from gv.data import make_batch
from gv.model import Prototype,state_bytes
from gv.render import render
import recall


class GeometryTests(unittest.TestCase):
    def test_so3_and_gradient(self):
        v=torch.zeros(7,3,requires_grad=True)
        R=exp_so3(v)
        R.sum().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        R=exp_so3(torch.randn(100,3)*4)
        self.assertLess(frame_error(R)['orthogonality_max'],2e-6)
        self.assertLess(frame_error(R)['determinant_error_max'],2e-6)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=Prototype(hidden=12,width=32,neighbors=3).eval()
        self.z,self.c,self.seq=make_batch(17,batch=2,n=9,steps=2)

    def test_global_rigid_equivariance(self):
        s=self.seq[0]
        h=self.model.dynamics.initial_memory(s)
        a,ha=self.model.dynamics(s,h,self.c)
        Q=exp_so3(torch.tensor([0.3,-0.5,0.8]))
        shift=torch.tensor([1.,-2.,0.3])
        moved=dict(s,p=s['p']@Q.T+shift,R=Q@s['R'],velocity=s['velocity']@Q.T)
        c=dict(self.c,axis=self.c['axis']@Q.T,center=self.c['center']@Q.T+shift)
        b,hb=self.model.dynamics(moved,h,c)
        torch.testing.assert_close(b['p'],a['p']@Q.T+shift,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(b['R'],Q@a['R'],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(hb,ha,atol=2e-6,rtol=2e-6)

    def test_permutation_equivariance(self):
        s=self.seq[0]
        h=self.model.dynamics.initial_memory(s)
        a,ha=self.model.dynamics(s,h,self.c)
        perm=torch.randperm(9)
        b,hb=self.model.dynamics({k:v[:,perm] for k,v in s.items()},h[:,perm],self.c)
        torch.testing.assert_close(b['p'],a['p'][:,perm])
        torch.testing.assert_close(hb,ha[:,perm])

    def test_state_size_independent_of_horizon(self):
        sizes=[]
        with torch.no_grad():
            for s,h in self.model.generate(self.z,self.c,steps=100):
                sizes.append(state_bytes(s,h))
                self.assertFalse(h.requires_grad)
        self.assertEqual(len(set(sizes)),1)
        self.assertLess(frame_error(s['R'])['orthogonality_max'],1e-5)

    def test_all_learned_components_receive_gradients(self):
        s=self.model.initializer(self.z,self.c)
        h=self.model.dynamics.initial_memory(s)
        out,h=self.model.dynamics(s,h,self.c)
        loss=out['p'].square().mean()+out['R'].square().mean()+h.square().mean()
        loss.backward()
        for name,module in [('init',self.model.initializer),('dynamics',self.model.dynamics)]:
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads,name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0,name)

    def test_checkpoint_and_memory_resume_without_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pt'
            torch.save({'config':self.model.config,'model':self.model.state_dict()},path)
            from generate import load_weights,make_input
            restored=load_weights(path,'cpu')
            with patch('gv.data.synthetic_sequence',side_effect=AssertionError('Teacher called at inference')):
                z,c=make_input(5,9,'orbit',0.8,'cpu')
                states=list(self.model.generate(z,c,steps=8))
                other=list(restored.generate(z,c,steps=8))
                torch.testing.assert_close(states[-1][0]['R'],other[-1][0]['R'])
                state,h=states[4]
                torch.save({'state':state,'h':h},Path(directory)/'memory.pt')
                saved=torch.load(Path(directory)/'memory.pt',weights_only=True)
                state,h=saved['state'],saved['h']
                for _ in range(4): state,h=restored.dynamics(state,h,c)
                torch.testing.assert_close(state['p'],states[-1][0]['p'])
                torch.testing.assert_close(h,states[-1][1])


class RenderTests(unittest.TestCase):
    def test_orientation_changes_pixels(self):
        s={'p':torch.zeros(1,3),'R':torch.eye(3)[None],
           'scale':torch.tensor([[0.4,0.04,0.04]]),'color':torch.ones(1,3),'alpha':torch.ones(1,1)*0.9}
        a=render(s,size=32)
        b=render(dict(s,R=exp_so3(torch.tensor([[0.,0.,1.5707963]]))),size=32)
        self.assertGreater((a-b).abs().mean().item(),0.005)
        self.assertTrue(torch.isfinite(a).all())


class RecallTests(unittest.TestCase):
    def test_mathml_preserves_latex_and_discards_script(self):
        p=recall.PaperHTML()
        p.feed('<p>rotation</p><math alttext="R^T R=I"><mi>duplicate</mi></math><script>untrusted()</script>')
        self.assertIn('R^T R=I',p.text())
        self.assertNotIn('untrusted',p.text())
        self.assertNotIn('duplicate',p.text())

    def test_offline_search_provenance_and_query_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            db=recall.connect(directory)
            db.execute('INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?)',('2104.12229v1','Vectors','https://arxiv.org/html/2104.12229v1','html','hash','now','vectors','local',1))
            db.execute('INSERT INTO chunks VALUES (?,?,?,?,?)',('2104.12229v1','section-2','Vectors','equivariance','orthogonal vector rotations'))
            db.commit(); db.close()
            hits=recall.search('"orthogonal" OR )',directory)
            self.assertEqual(hits[0]['id'],'2104.12229v1')
            self.assertEqual(hits[0]['sha256'],'hash')
            self.assertEqual(hits[0]['equation_status'],'unverified extraction')
            self.assertEqual(recall.search('!!!',directory),[])


if __name__=='__main__': unittest.main()
