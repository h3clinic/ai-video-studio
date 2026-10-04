"""Record actual recovery outcomes and stage timings, without model/API work."""
import json
from real_video.gaussian_program import Program,evidence,ROOT

def main():
    out=ROOT/'artifacts/cloud/object_edit_fresh4090_v4'
    report=json.loads((out/'experiment/results/output/report.json').read_text())
    frames=report['frames']
    metrics=dict(frames=len(frames),duration_seconds=121/24,
        model_load_seconds=report['stages']['detector_segmenter_load_seconds'],
        segmentation_seconds=sum(f['semantic_seconds'] for f in frames),
        repair_seconds=sum(f['repair_seconds'] for f in frames),
        raster_seconds=sum(f['raster_seconds'] for f in frames),
        encoding_seconds=report['stages']['encoding_seconds'],
        peak_gpu_bytes=report['peak_gpu_bytes'],training_seconds=0,
        all_unselected_colour_fields_exact=all(f['outside_colour_exact'] for f in frames),
        savings_proven=False,accepted=False,
        cost_note='GPU rate 0.74 USD/hour; timing above excludes startup, transfer, setup, and failed trials. Not an invoice or quality-matched comparison.')
    (out/'metrics_summary.json').write_text(json.dumps(metrics,indent=2))
    p=Program()
    try:
        p.record_experiment(dict(role='rendering',issue_key='rendering.related_part_elimination',outcome='partial',
            hypothesis='Text-grounded DINO/SAM masks select fruit residuals outside the old fixed ROI and permit local Gaussian colour repair.',
            observation='Existing 4090 resume reported gpu_capacity_unavailable. Fresh-host allocation succeeded. Two loading failures were preserved: AutoProcessor resolution and missing hf_transfer. Explicit snapshots and standard downloader enabled a completed 40-frame run in 28.57 seconds remote experiment wall time. Sampled review finds improved residue removal, but dark static apple and blurred repaired bowl fail overall acceptance. All remote workers stopped.',
            next_action='Improve temporally consistent local repair and physically meaningful apple-mouth contact; no autonomous generation or savings claim.',
            inputs=[evidence(ROOT,ROOT/'cloud/object_edit_semantic_worker.py'),evidence(ROOT,ROOT/'research/prior_manifest.json')],
            outputs=[evidence(ROOT,out/'experiment/results/output/gaussian_apple.mp4'),evidence(ROOT,out/'visual_review.json'),evidence(ROOT,out/'metrics_summary.json'),evidence(ROOT,out/'final_status.json')]))
        p.export()
    finally:p.db.close()

if __name__=='__main__':main()
