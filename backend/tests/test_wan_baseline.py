import unittest
from argparse import Namespace

from real_video.wan_baseline import validate_args, MODEL, REVISION


class WanBaselineTests(unittest.TestCase):
    def test_fixed_supported_shape(self):
        validate_args(Namespace(height=480, width=832, frames=33, steps=50))
        self.assertEqual(MODEL, 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
        self.assertEqual(len(REVISION), 40)

    def test_invalid_shape_rejected_instead_of_silently_resized(self):
        for field, value in [('height', 481), ('width', 0), ('frames', 32), ('steps', 0)]:
            args = Namespace(height=480, width=832, frames=33, steps=50)
            setattr(args, field, value)
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_latent_only_pipeline_does_not_require_rgb_decoder(self):
        # Random miniature model verifies plumbing only, never a video-quality result.
        import torch
        from diffusers import WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler
        model = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=16,
                    in_channels=16, out_channels=16, text_dim=32, freq_dim=16,
                    ffn_dim=64, num_layers=1).eval()
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction',
                                            use_flow_sigmas=True, flow_shift=8.0)
        pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=None,
                           transformer=model, scheduler=scheduler)
        embeds = torch.zeros(1, 4, 32)
        output = pipe(prompt_embeds=embeds, negative_prompt_embeds=embeds,
                      height=64, width=96, num_frames=5, num_inference_steps=2,
                      guidance_scale=6.0, output_type='latent',
                      generator=torch.Generator().manual_seed(71)).frames
        self.assertEqual(tuple(output.shape), (1, 16, 2, 8, 12))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
