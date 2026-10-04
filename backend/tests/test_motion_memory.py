import unittest
import numpy as np
import torch
from real_video.gaussian_motion_memory import MotionReplay, transport_geometry
from real_video.keyframed_gaussian_motion import frame_points


class MotionMemoryTests(unittest.TestCase):
    def packet(self):
        base=torch.zeros(4,10); base[:,4]=1; base[:,9]=1
        base[:,6:9]=torch.arange(12).reshape(4,3).float()/12
        motion=torch.zeros(3,4,2)
        motion[:,:,0]=torch.tensor([.01,.02,-.01,-.02])[None]
        return dict(base=base,displacements=motion,logscale_deltas=torch.zeros(3,4,2),
                    angle_deltas=torch.ones(3,4)*.1,visibility=torch.ones(4,4))

    def test_independent_motion_and_constant_colours(self):
        packet=self.packet(); s=MotionReplay(packet,'cpu')
        s.advance()
        self.assertTrue(torch.equal(s.points[:,:2],packet['displacements'][0]))
        self.assertTrue(torch.equal(s.points[:,6:9],packet['base'][:,6:9]))
        self.assertFalse(torch.equal(s.points[0,:2],s.points[1,:2]))
        self.assertTrue(torch.allclose(s.points[:,4:6].norm(dim=-1),torch.ones(4)))

    def test_replay_restore_end_and_source_ownership(self):
        packet=self.packet(); saved=packet['base'].clone(); s=MotionReplay(packet,'cpu')
        s.advance(); snap=s.snapshot(); s.advance(); expected=s.points.clone()
        s.restore(snap); s.advance(); self.assertTrue(torch.equal(expected,s.points))
        s.advance()
        with self.assertRaises(StopIteration): s.advance()
        self.assertTrue(torch.equal(packet['base'],saved))

    def test_vectors_only_and_frozen_ablations(self):
        packet=self.packet(); s=MotionReplay(packet,'cpu','vectors_only'); s.advance()
        self.assertTrue(torch.equal(s.points[:,2:],packet['base'][:,2:]))
        frozen=MotionReplay(packet,'cpu','frozen'); frozen.advance()
        self.assertTrue(torch.equal(frozen.points,packet['base']))

    def test_covariance_transport_rotation(self):
        logs=np.log(np.array([[3/480,2/480]],np.float32)); axes=np.array([[1.,0.]],np.float32)
        rotation=np.array([[[0.,-1.],[1.,0.]]],np.float32)
        newlogs,newaxes=transport_geometry(logs,axes,rotation)
        self.assertTrue(np.allclose(newlogs,logs))
        self.assertTrue(np.allclose(newaxes,np.array([[0.,1.]])))

    def test_bidirectional_anchors_and_vector_ablation(self):
        anchors=[]
        for i in range(9):
            p=torch.zeros(2,10); p[:,:2]=i*.04; anchors.append(p)
        forward=torch.ones(4,2,2)*.01
        packet=dict(anchors=anchors,segments=[dict(forward=forward,backward=-forward) for _ in range(8)])
        for t in [0,1,2,4,15,31,32]:
            left,right,weight=frame_points(packet,t,'cpu')
            self.assertTrue(torch.allclose(left[:,:2],torch.full((2,2),t*.01),atol=1e-6))
            self.assertTrue(torch.allclose(right[:,:2],left[:,:2],atol=1e-6))
            self.assertTrue(0<=weight<=1)
        left,right,_=frame_points(packet,2,'cpu',motion=False)
        self.assertTrue(torch.equal(left,anchors[0]))
        self.assertTrue(torch.equal(right,anchors[1]))
        with self.assertRaises(ValueError): frame_points(packet,33,'cpu')


if __name__=='__main__': unittest.main()
