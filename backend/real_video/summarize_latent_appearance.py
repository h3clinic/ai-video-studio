"""Revalidate the measured reader pilot and record its scoped failure."""
import json
from pathlib import Path

from .articulation_quality import decide_review
from .checkpoint_io import load_verified
from .gaussian_program import Program, ROOT, evidence, write_json, verify_evidence


def summarize():
    out=Path('artifacts/real_video/detail_memory/reader_v2')
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    decisions={}
    for mode in ('v3_reader_disabled','memory','flipped_features'):
        review=json.loads((out/f'{mode}_visual_review.json').read_text(encoding='utf-8'))
        decisions[mode]=decide_review(review,out/f'{mode}.mp4')
    base=report['validation']['after']['v3_reader_disabled']
    actual=report['validation']['after']['memory']
    wrong=report['validation']['after']['flipped_features']
    prepared=load_verified(out/'prepared.pt')
    bank=prepared['anchors_only'][-1]
    old_paths=list(report['sources'])
    retained=all(verify_evidence(ROOT,dict(path=path,sha256=report['sources'][path])) for path in old_paths)
    record=dict(schema_version=1,finished=report.get('finished') is True,
        quality_accepted=False,native_gaussian_generation_achieved=False,checkpoint_promoted=False,
        model_change='675,200 newly trained appearance-reader parameters alongside frozen Wan and v3 adapter',
        trained_parameters=report['new_trainable_parameters'],training_steps=report['training_steps'],
        frozen_parameters_unchanged=report['frozen_parameters_unchanged'],old_artifact_hashes_unchanged=retained,
        serialization_exact=report['saved_reader_reload_exact'],
        evidence=[evidence(ROOT,out/p) for p in ('report.json','latent_reader.pt','prepared.pt','comparison.png')],
        appearance_protocol='First-image latent write -> fixed-ID planar Gaussian recall -> new Wan residual reader; full future 2D tracks are supplied',
        memory=dict(gaussian_count_per_development_bank=bank['ids'].numel(),latent_channels=bank['channels'],
            writes_per_bank=bank['writes'],canonical_snapshot_has_future_tracks='tracks' in bank,
            resident_feature_mass_id_bytes=sum(bank[k].numel()*bank[k].element_size() for k in ('ids','features','observation_mass')),
            note='This 1,792-dot planar training bank is not the 400,000-dot 3D cat; 3D writer/readout integration remains incomplete'),
        measurements=dict(training_seconds=report['training_seconds'],training_peak_cuda_bytes=report['training_peak_cuda_bytes'],
            peak_process_rss_bytes=report['resources']['peak_process_rss_bytes'],
            new_reader_checkpoint_bytes=(out/'latent_reader.pt').stat().st_size,
            prepared_bank_condition_target_archive_bytes=(out/'prepared.pt').stat().st_size,
            denoising=report['samples'],end_to_end_wan_savings_established=False),
        diagnostic_changes=dict(
            flow_mse_relative_improvement_percent=100*(base['flow_mse']-actual['flow_mse'])/base['flow_mse'],
            latent_edge_error_relative_improvement_percent=100*(base['future_edge_error']-actual['future_edge_error'])/base['future_edge_error'],
            correct_vs_flipped_flow_gap_percent=100*(wrong['flow_mse']-actual['flow_mse'])/wrong['flow_mse']),
        independent_visual_decisions=decisions,
        conclusion='Rejected: a different brown antelope-like animal replaces the source patterned cow; flipped appearance gives nearly the same sampled output. Small loss changes do not demonstrate identity/detail retention.',
        limitations=['17-frame, 1.417-second development diagnostic, not a seven-second accepted 3D action video',
            'Training/development reuse, no independent final test and only 32 optimization steps',
            'No learned geometry/correspondence writer, native Gaussian updates or generated appearance writeback',
            'Existing malformed 3D cat/dog assets unchanged; no new fine fur in those assets',
            'Spatially flipped features stress alignment; this single trial does not prove all feature conditioning is ignored',
            'Sampled decoded poses inspected, not continuous playback'],
        next_action='Keep the learned reader experimental. Test source-identity-sensitive training/conditioning on real development data with fixed-backbone controls and wrong-identity ablations before 3D transfer; train reliable geometry/correspondence before generated memory commits. Do not claim that more splats or smaller raster footprints can supply missing appearance.')
    write_json(out/'outcome.json',record)
    program=Program()
    try:
        outputs=[evidence(ROOT,out/p) for p in ('outcome.json','report.json','latent_reader.pt','prepared.pt','memory_visual_review.json')]
        inputs=[evidence(ROOT,path) for path in old_paths]
        program.record_experiment(dict(role='generation',issue_key='generation.latent_appearance',
            hypothesis='A patch-preserving learned reader can use anchor-only Gaussian latent features to retain appearance better than five-channel RGB conditioning',
            observation=record['conclusion'],next_action=record['next_action'],outcome='rejected',inputs=inputs,outputs=outputs))
        program.issue('generation.latent_appearance','generation','Trained latent reader does not yet retain source identity/detail',
            outputs+[evidence(ROOT,'real_video/wan_latent_appearance.py'),evidence(ROOT,'real_video/latent_track_memory.py')],record['next_action'])
        program.export()
    finally: program.db.close()
    print(json.dumps({k:record[k] for k in ('quality_accepted','measurements','diagnostic_changes','conclusion')},indent=2))


if __name__=='__main__': summarize()
