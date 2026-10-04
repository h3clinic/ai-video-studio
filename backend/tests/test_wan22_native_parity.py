"""CPU native/manual TI2V parity, executing the actual sampler's loop AST.

No pretrained weights, downloads, VAE inference or GPU. This verifies protocol,
not real-model quality. The reference promotes model outputs before CFG to
match our intentional FP32 guidance arithmetic; all transformer calls use BF16.
"""
import ast
from pathlib import Path
import types
import unittest

import torch


SAMPLER = Path(__file__).resolve().parents[1]/'real_video/sample_wan22_memory.py'


def production_denoising_block():
    """Extract, without rewriting, the unique inference block with the loop.

    Importing/running the sampler would trigger unrelated resource/download
    setup. AST extraction binds this test to production rather than copying
    its equations. Missing or ambiguous blocks fail instead of silently using
    an old duplicate. Its real final anchor clamp is included in the block.
    """
    tree = ast.parse(SAMPLER.read_text(encoding='utf-8'), filename=str(SAMPLER))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    expected = ast.dump(ast.parse('enumerate(scheduler.timesteps)', mode='eval').body)
    matches = [node for node in ast.walk(run) if isinstance(node, ast.With)
               and any(isinstance(child, ast.For) and ast.dump(child.iter) == expected
                       for child in node.body)]
    if len(matches) != 1:
        raise AssertionError('Expected one production denoising block; update explicit binding after refactor')
    block = matches[0]
    if len(block.body) != 2 or not isinstance(block.body[-1], ast.Assign):
        raise AssertionError('Production inference block changed; audit its final anchor clamp')
    return compile(ast.Module(body=[block], type_ignores=[]), str(SAMPLER), 'exec')


class Wan22NativeParityTests(unittest.TestCase):
    def test_actual_production_loop_matches_native_i2v_with_same_cfg_precision(self):
        from diffusers import (AutoencoderKLWan, UniPCMultistepScheduler,
                               WanImageToVideoPipeline, WanTransformer3DModel)
        from PIL import Image

        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(601)
            model = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8,
                in_channels=48, out_channels=48, text_dim=16, freq_dim=8, ffn_dim=32,
                num_layers=1, patch_size=(1,2,2), image_dim=None).to(torch.bfloat16)
            # Mirror mixed-precision exceptions used by from_pretrained.
            for name, module in model.named_modules():
                if any(part in model._keep_in_fp32_modules for part in name.split('.')):
                    module.float()
            for name, parameter in model.named_parameters():
                if any(part in model._keep_in_fp32_modules for part in name.split('.')):
                    parameter.data = parameter.data.float()
            model.eval().requires_grad_(False)
            vae = AutoencoderKLWan(base_dim=8, decoder_base_dim=8, dim_mult=[1,2,4,4],
                num_res_blocks=1, z_dim=48, in_channels=12, out_channels=12,
                is_residual=True, patch_size=2, scale_factor_spatial=16,
                latents_mean=[0.]*48, latents_std=[1.]*48).eval()
            noise = torch.randn(1,48,9,2,4)
            anchor = torch.randn(1,48,1,2,4).to(torch.bfloat16)
            mask = torch.ones(1,1,9,2,4); mask[:,:,0] = 0
            positive = torch.randn(1,4,16).to(torch.bfloat16)
            negative = torch.randn(1,4,16).to(torch.bfloat16)

        def scheduler():
            return UniPCMultistepScheduler(prediction_type='flow_prediction',
                use_flow_sigmas=True, flow_shift=3., use_dynamic_shifting=False,
                solver_order=2, solver_type='bh2', predict_x0=True,
                lower_order_final=True, final_sigmas_type='zero')

        pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae,
            transformer=model, scheduler=scheduler(), expand_timesteps=True)
        pipe.set_progress_bar_config(disable=True)

        def supplied_latents(self, *args, **kwargs):
            return noise.clone(), anchor.float(), mask.clone()

        pipe.prepare_latents = types.MethodType(supplied_latents, pipe)
        self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
        self.assertTrue(all(p.device.type == 'cpu' for p in vae.parameters()))
        # Explicitly match our FP32 CFG convention, not native BF16 CFG.
        hook = model.register_forward_hook(lambda module, args, result: (result[0].float(),))
        try:
            native = pipe(image=Image.new('RGB',(64,32)), height=32, width=64,
                num_frames=33, num_inference_steps=20, guidance_scale=5.,
                prompt_embeds=positive, negative_prompt_embeds=negative,
                output_type='latent').frames
        finally:
            hook.remove()

        manual_scheduler = scheduler()
        manual_scheduler.set_timesteps(20, device='cpu')
        namespace = dict(torch=torch, model=model, scheduler=manual_scheduler,
            latents=noise.clone(), anchor=anchor, first_mask=mask.clone(),
            positive=positive, negative=negative, report={'guidance_scale':5.},
            mode='cpu_parity', steps=20, status=lambda message: None)
        exec(production_denoising_block(), namespace)
        actual = namespace['latents']
        self.assertEqual(actual.shape, (1,48,9,2,4))
        self.assertTrue(torch.equal(actual[:,:,:1], anchor.float()))
        torch.testing.assert_close(actual, native, rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main()
