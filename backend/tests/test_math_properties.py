"""Numerical checks of the mathematical note, not empirical physics claims."""
import unittest
import torch
from real_video.replay_fit import temporal_basis,temporal_basis_at,physical_fields,expand_coefficients
from real_video.representation import decode_state,render_fields


class MathPropertyTests(unittest.TestCase):
    def test_fractional_basis_matches_discrete_and_analytic_derivative(self):
        torch.testing.assert_close(temporal_basis(16,8),temporal_basis_at(16,8,torch.arange(16)),rtol=0,atol=0)
        t=torch.tensor([0.7,5.2,13.1]); h=0.001
        finite=(temporal_basis_at(16,8,t+h)-temporal_basis_at(16,8,t-h))/(2*h)
        analytic=temporal_basis_at(16,8,t,derivative=True)
        torch.testing.assert_close(finite,analytic,atol=0.0003,rtol=0.002)

    def test_covariance_is_positive_and_sign_gauge_is_unobservable(self):
        torch.manual_seed(72)
        fields=torch.randn(1,9,2,8,8)*0.2
        state=decode_state(fields[:,:,0],min_log_scale=-2.)
        Q=state['R'][...,:2,:2]; scales=state['scale'][...,:2]
        covariance=Q@torch.diag_embed(scales.square())@Q.transpose(-1,-2)
        self.assertGreater(torch.linalg.eigvalsh(covariance).min().item(),0)
        changed=fields.clone(); changed[:,4:6]*=-1
        torch.testing.assert_close(render_fields(fields),render_fields(changed),rtol=1e-6,atol=1e-6)

    def test_anchor_bound_in_native_pixel_coordinates(self):
        g=8; S=256; coefficients=torch.randn(9,4,g,g)*100
        fields=physical_fields(coefficients,temporal_basis(4,4),2/g)
        native=(decode_state(fields[:,0][None])['p'][0,:,:2]+0.5)*(S/64)-0.5
        grid=(torch.arange(g)+0.5)*S/g-0.5
        yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        anchors=torch.stack((xx,yy),-1).reshape(-1,2)
        self.assertLessEqual((native-anchors).abs().max().item(),S/(2*g)+1e-5)

    def test_anchored_velocity_matches_finite_difference(self):
        torch.manual_seed(39); c=torch.randn(9,4,8,8)*0.2
        t=torch.tensor([5.2]); h=0.001; L=0.25; S=256; fps=8
        basis=temporal_basis_at(16,4,t)
        z=expand_coefficients(c,basis)[:2]
        dz=expand_coefficients(c,temporal_basis_at(16,4,t,derivative=True))[:2]
        analytic=fps*(S/4)*L*(1-z.tanh().square())*dz
        plus=physical_fields(c,temporal_basis_at(16,4,t+h),L)[:2]
        minus=physical_fields(c,temporal_basis_at(16,4,t-h),L)[:2]
        finite=fps*(S/4)*(plus-minus)/(2*h)
        torch.testing.assert_close(finite,analytic,atol=0.01,rtol=0.01)

    def test_orthonormal_coefficient_error_energy(self):
        basis=temporal_basis(16,7); errors=torch.randn(7,19)
        torch.testing.assert_close((basis@errors).square().sum(),errors.square().sum(),rtol=1e-6,atol=1e-5)


if __name__=='__main__': unittest.main()
