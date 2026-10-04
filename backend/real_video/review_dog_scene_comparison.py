"""Evidence-linked review and full-cost summary for the two requested workflows."""
import json
from pathlib import Path
import statistics
from real_video.articulation_quality import create_review,decide_review
from real_video.dog_scene_experiment import OUT


def review(folder,video_name,claim,observations):
    video=folder/video_name
    result=create_review(video,[dict(path=folder/f'frame_{i:03d}.png',frame_index=i) for i in [0,7,23,32]])
    result['claim_type']=claim
    result['reviewer']=dict(kind='agent',name='Codex sampled-frame source/mask/composition inspection',observed_video=False)
    result['provenance']=dict(future_conditioned=claim=='reconstruction',test_used_for_training=False,test_used_for_selection=False)
    for frame in result['frames']: frame['observation']=observations['frames']
    for name,(status,description) in observations['dimensions'].items():
        result['dimensions'][name]=dict(status=status,observation=description,evidence_frame_ids=[f['id'] for f in result['frames']])
    result['limitations']=['Continuous playback not attested. Sampled frames include pre-refresh frame 7/23, not just sharper bank resets.',observations['limits']]
    result['decision']=decide_review(result,video)
    (folder/'visual_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result['decision']


def main():
    a=review(OUT/'composition_v2','gaussian_cat_plus_dog.mp4','reconstruction',dict(
        frames='Original highly saturated cat is retained as the dominant animal; small separately generated dog overlays its foreground. Before bank refreshes there are black missing-coverage patches around the cat paw and tail and loss of dog boundary/face detail.',
        limits='2D composition with captured independent motion, no 3D scene or learned interaction. Cat saturation was already present in the exact requested source. The initial dog mask mistakenly excluded dark eyes/nose; enclosed holes were filled in v2 without another Wan generation, but tracking/visibility errors remain.',
        dimensions=dict(paw_placement=('fail','Foreground dog scale/contact/shadows do not match a believable shared environment.'),
                        limb_bending=('uncertain','Recorded leg movement is present; physical gait not established from samples.'),
                        body_shape=('fail','Relative animal scale is implausible and outlines degrade between refreshes.'),
                        tearing=('fail','Black coverage holes and fragmented outlines visible near the cat paw/tail and dog mouth/legs.'),
                        temporal_coherence=('fail','Marked appearance degradation before appearance-bank resets; continuity not validated.'))))
    b=review(OUT/'joint','video.mp4','conditional_motion_generation',dict(
        frames='Cat and dog appear together initially with more consistent photographic appearance. Dog is cropped and progresses off the right edge by later sampled frames, despite prompt asking both animals fully visible.',
        limits='Fresh joint generation, not preservation of exact previous cat/environment. Dog framing requirement failed; this is not a quality-matched comparator to A.',
        dimensions=dict(paw_placement=('uncertain','Paws and forward steps are visible but contacts partially occluded and continuous gait not reviewed.'),
                        limb_bending=('uncertain','Different leg extensions seen; exact physical articulation unverified.'),
                        body_shape=('uncertain','Plausible visible silhouettes, but dog is incomplete/cropped in the image.'),
                        tearing=('pass','No comparable black coverage holes seen in the sampled RGB frames.'),
                        temporal_coherence=('uncertain','Cat persists across samples; dog leaves the field of view rather than remaining fully visible.'))))
    generation=json.loads((OUT/'generation_report.json').read_text())
    composition=json.loads((OUT/'composition_v2/metrics.json').read_text())
    cat=json.loads((OUT/'cat_capture/capture_report.json').read_text())
    dog=json.loads((OUT/'dog_capture_v2/capture_report.json').read_text())
    summary=dict(status='Completed fixed-camera diagnostic; neither result approved as high-quality 3D generative scene memory',
        A=dict(workflow='Existing exact cat video -> Gaussian motion banks; new independent Wan dog -> Gaussian motion banks; composite and replay without Wan',
               fresh_wan_denoise=generation['cases']['dog']['denoise'],prompt_encoding=generation['cases']['dog']['text_encode'],
               cat_capture_seconds=cat['capture_seconds'],dog_capture_seconds=dog['capture_seconds'],
               replay=composition['stages']['streaming_replay_without_rgb_history'],
               full_uncompressed_gaussian_tensor_bytes=composition['storage']['all_cpu_gaussian_tensor_bytes'],
               lossless_compressed_editable_bank_bytes=composition['storage']['cat_lossless_gzip_bytes']+composition['storage']['dog_lossless_gzip_bytes'],
               maximum_live_gaussian_state_bytes=composition['storage']['maximum_live_gaussian_state_bytes'],
               rendered_mp4_bytes=composition['storage']['A_encoded_video_file_bytes'],quality=a),
        B=dict(workflow='Original cat prompt plus dog request -> fresh whole-scene Wan generation',
               fresh_wan_denoise=generation['cases']['joint']['denoise'],prompt_encoding=generation['cases']['joint']['text_encode'],
               saved_mp4_bytes=composition['storage']['B_encoded_video_file_bytes'],
               cached_mp4_cpu_decode_median_seconds=statistics.median(composition['stages']['B_encoded_video_cpu_decode']['seconds_each']),quality=b),
        shared_setup=generation['common'],
        conclusions=['No measured denoising GPU memory reduction: same peak allocated bytes in both fresh generations.',
                     'Same frame dimensions, number of frames, denoising steps and guidance passes; no denoiser-work reduction implemented.',
                     'Gaussian-only replay avoids Wan but cached MP4 playback also avoids Wan and is faster here.',
                     'Editable Gaussian banks are larger than a saved output MP4; they provide different editing capabilities, not better video compression.',
                     'Observed generation times are thermally confounded: 87C, active NVIDIA software thermal slowdown during B. No algorithmic speedup inferred.',
                     'A is captured planar motion, not persistent learned 3D dynamics; bank IDs reset every 8 frames.',
                     'Original cat generation/training is a pre-existing sunk cost, excluded from incremental A timings; cold-start acquisition would add it.'],
        accounting_notes=['Shared model loads measured once and reported separately, not duplicated into each case.',
                          'Source/latent/prompt/debug artifacts and installed model checkpoints are retained but are not counted as minimal replay payload.',
                          'All Gaussian future states and appearance banks ARE counted. Live state alone is not total memory.',
                          'CPU RSS includes runtime/framework overhead and allocator retention. GPU figures are PyTorch allocated tensors, not total GPU usage.',
                          'Initial segmentation trial and v2 correction both retained. Figures for final A refer to v2; full research-development cost exceeds a single production run.',
                          'Gaussian cat capture briefly ran on CPU during dog inference; timings are diagnostics, not a controlled replicated performance study.'])
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
