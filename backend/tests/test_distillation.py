import unittest
import torch
from torch import nn
from real_video.model import guided_velocity,ddim_step,sample,schedule,GaussianVideoDenoiser


class Toy(nn.Module):
    def __init__(self,student=False):
        super().__init__(); self.weight=nn.Parameter(torch.zeros(())); self.classes=2
        self.config={'channels':1}; self.student=student; self.batches=[]
    def forward(self,x,t,label):
        self.batches.append(len(x))
        cond=label.float().reshape(-1,1,1,1,1)
        if self.student: cond=2+1.5*(cond-2)
        return x*.1+cond*.03+self.weight


class DistillationTests(unittest.TestCase):
    def test_precasting_keeps_normalization_and_embeddings_fp32(self):
        from real_video.inference_precision import precast_autocast_weights
        model=GaussianVideoDenoiser(classes=2,width=8,channels=16)
        with self.assertRaisesRegex(ValueError,'inference-only'): precast_autocast_weights(model)
        precast_autocast_weights(model.eval())
        self.assertEqual(model.input.weight.dtype,torch.bfloat16)
        self.assertEqual(model.time[0].weight.dtype,torch.bfloat16)
        self.assertEqual(model.label.weight.dtype,torch.float32)
        self.assertEqual(model.r1.norm1.weight.dtype,torch.float32)
    def test_runner_guidance_mismatch_is_rejected(self):
        model=Toy()
        class WrongRunner:
            guidance=2.
        with self.assertRaisesRegex(ValueError,'runner does not match'):
            sample(model,torch.tensor([0]),17,steps=2,shape=(2,4,4),velocity_fn=WrongRunner())
    def test_forward_noise_uses_declared_alpha(self):
        from real_video.distill_guidance import forward_noised
        z=torch.ones(2,1,1,2,2)*3; noise=torch.ones_like(z)*2
        actual=forward_noised(z,torch.tensor([0,1]),noise,torch.tensor([1.,0.]))
        torch.testing.assert_close(actual[0],z[0]); torch.testing.assert_close(actual[1],noise[1])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA graph requires CUDA')
    def test_cuda_graph_accepts_fresh_states_times_labels_and_preserves_outputs(self):
        from real_video.graph_sampling import GraphedVelocity
        torch.manual_seed(93)
        model=GaussianVideoDenoiser(classes=2,width=8,channels=16).cuda().eval()
        runner=GraphedVelocity(model,1,(4,8,8))
        with torch.no_grad():
            first=None
            for seed,label,time in ((3,0,999),(4,1,200)):
                torch.manual_seed(seed); x=torch.randn(1,16,4,8,8,device='cuda')
                t=torch.tensor([time],device='cuda'); y=torch.tensor([label],device='cuda')
                actual=runner(x,t,y); expected=guided_velocity(model,x,t,y,1.5)
                torch.testing.assert_close(actual,expected,rtol=0,atol=1e-5)
                if first is None: first=actual; snapshot=actual.clone()
            torch.testing.assert_close(first,snapshot,rtol=0,atol=0)
            labels=torch.tensor([1],device='cuda')
            expected=sample(model,labels,17,steps=3,shape=(4,8,8))
            actual=sample(model,labels,17,steps=3,shape=(4,8,8),velocity_fn=runner)
            torch.testing.assert_close(actual,expected,rtol=0,atol=1e-5)
            with self.assertRaisesRegex(ValueError,'shape mismatch'):
                runner(torch.zeros(2,16,4,8,8,device='cuda'),t,y)
    def test_fixed_guidance_matches_teacher_with_half_batch(self):
        teacher=Toy(); student=Toy(True); x=torch.randn(2,1,2,4,4)
        t=torch.tensor([5,5]); labels=torch.tensor([0,1])
        expected=guided_velocity(teacher,x,t,labels,1.5)
        actual=guided_velocity(student,x,t,labels,1.5,1.5)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(teacher.batches,[4]); self.assertEqual(student.batches,[2])
    def test_reject_untrained_guidance(self):
        with self.assertRaisesRegex(ValueError,'Guidance differs'):
            sample(Toy(True),torch.tensor([0]),42,steps=2,shape=(2,4,4),guidance=3,distilled_guidance=1.5)
    def test_rollout_equivalent_and_observer_does_not_change_sampling(self):
        seen=[]; labels=torch.tensor([0,1]); teacher=Toy(); student=Toy(True)
        result=sample(teacher,labels,42,steps=5,shape=(2,4,4),observer=lambda x,t,v:seen.append((x.clone(),t.clone(),v.clone())))
        actual=sample(student,labels,42,steps=5,shape=(2,4,4),distilled_guidance=1.5)
        torch.testing.assert_close(actual,result)
        self.assertEqual(len(seen),5)
        torch.testing.assert_close(result,sample(teacher,labels,42,steps=5,shape=(2,4,4)))
    def test_ddim_preserves_original_formula(self):
        x=torch.randn(2,3); v=torch.randn_like(x); alpha=schedule('cpu'); a=alpha[500]; b=alpha[300]
        predicted=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-4,4)
        noise=(1-a).sqrt()*x+a.sqrt()*v
        torch.testing.assert_close(ddim_step(x,v,a,b),b.sqrt()*predicted+(1-b).sqrt()*noise)
        torch.testing.assert_close(ddim_step(x,v,a),predicted)
    def test_ddim_local_error_bound_including_clipping(self):
        torch.manual_seed(52)
        x=torch.randn(2,3)*8; v=torch.randn_like(x)*5
        dx=torch.randn_like(x)*.3; dv=torch.randn_like(x)*.3
        alpha=torch.tensor(.4); next_alpha=torch.tensor(.7)
        a=alpha.sqrt(); s=(1-alpha).sqrt(); b=next_alpha.sqrt(); q=(1-next_alpha).sqrt()
        actual=(ddim_step(x+dx,v+dv,alpha,next_alpha)-ddim_step(x,v,alpha,next_alpha)).norm()
        bound=(b*a+q*s)*dx.norm()+(b*s+q*a)*dv.norm()
        self.assertLessEqual(actual.item(),bound.item()+1e-6)


if __name__=='__main__': unittest.main()
