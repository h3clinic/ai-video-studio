import unittest
import torch
from real_video.animal_control_graph import AnimalControlGraph,bind_points
from real_video.train_animal_visual import render_loss


class AnimalVisualTests(unittest.TestCase):
    def setup_case(self):
        p=torch.tensor([[.3,.3],[.7,.3],[.3,.7],[.7,.7]])
        sample=dict(position=p[None,None].repeat(1,4,1,1),angle=torch.zeros(1,4,4),
                    visibility=torch.ones(1,4,4),confidence=torch.ones(1,4,4),colour=torch.zeros(1,4,3),adjacency=torch.eye(4)[None])
        points=torch.zeros(4,10); points[:,:2]=p; points[:,2:4]=-2; points[:,4]=1; points[:,9]=1
        index,weight=bind_points(p,p)
        target=dict(points=points,index=index,weight=weight,frames=torch.full((4,8,8,3),128,dtype=torch.uint8),masks=torch.ones(4,8,8,dtype=torch.uint8))
        model=AnimalControlGraph()
        state=model.initialize(sample['position'][:,:3],sample['angle'][:,:3],sample['visibility'][:,:3],sample['colour'],sample['adjacency'])
        return model,state,sample,target

    def test_render_supervision_has_finite_model_gradients(self):
        model,state,sample,target=self.setup_case(); state=model.step(state)
        loss,*_=render_loss(state,sample,target,3); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_target_does_not_mutate_predicted_state(self):
        model,state,sample,target=self.setup_case(); state=model.step(state)
        saved={k:v.clone() for k,v in state.items()}; first=render_loss(state,sample,target,3)[0]
        target['frames'][3].fill_(0); second=render_loss(state,sample,target,3)[0]
        self.assertNotEqual(float(first.detach()),float(second.detach()))
        self.assertTrue(all(torch.equal(saved[k],state[k]) for k in state))

    def test_silhouette_loss_penalizes_disappearance(self):
        _,state,sample,target=self.setup_case(); first=render_loss(state,sample,target,3)[2]
        hidden=dict(state,visibility=torch.zeros_like(state['visibility']))
        second=render_loss(hidden,sample,target,3)[2]
        self.assertGreater(float(second),float(first))


if __name__=='__main__': unittest.main()
