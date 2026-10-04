"""Collate immutable experimental evidence; never tune or relabel failed gates."""
import json
from pathlib import Path
from .checkpoint_io import digest


def main():
    root=Path('artifacts/real_video/true3d')
    output=root/'vector_self_eval_summary.json'
    if output.exists(): raise FileExistsError('Preserve completed self-evaluation summary')
    inputs={}
    def read(relative):
        path=root/relative
        inputs[str(path)]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    fit=read('dense_fit_loop/v2/loop_report.json')
    fit_visual=read('dense_fit_loop/v2/visual_audit.json')
    fit_record=read('dense_fit_loop/v2/gaussian_vectors.audit.json')
    learned_record=read('attached_vector_weights/v2/recording_audit.json')
    inference=read('attached_vector_weights/v2/inference_audit.json')
    resume=read('attached_vector_weights/v2/resume_audit.json')
    visual=read('attached_vector_weights/v2/evaluation/visual_review.json')
    read('attached_vector_weights/v2/independent_audit.json')
    selections=[read(f'learned_motion_loop/v{version}/selection.json') for version in [2,3,4]]
    protocols=[read(f'learned_motion_loop/v{version}/protocol.json') for version in [2,3,4]]
    final=selections[-1]
    means=final['best_trained_selection']['scores']['means']
    report=dict(date='2026-09-30',scope='Completed bounded self-evaluation; genuine model improvements are not assumed.',
        fit=dict(completed_candidates=6,classification='full-source-conditioned reconstruction',
            baseline_tracking_epe_source_px=fit['baseline_lk_epe_source_px'],selected_tracking_epe_source_px=fit['selected_lk_epe_source_px'],
            surviving_track_error_reduction=fit['improvement_ratio'],numerical_strain_pass=fit['accepted'],
            natural_motion_pass=fit_visual['natural_gait_passed'],
            warning='Only about5%oflower-limb/foottracks survivefinalframe; correspondenceerror doesnotmeasureallmotion/hidden3Daccuracy.'),
        learning=dict(completed_candidates=sum(len(p['candidates']) for p in protocols),
            completed_training_updates=sum(len(p['candidates'])*p['steps_per_candidate'] for p in protocols),
            loop_seconds=sum(s['loop_seconds'] for s in selections),
            best_trained_ade=means['learned']['ade'],best_trained_fde=means['learned']['fde'],
            disabled_ade=means['zero_weights']['ade'],disabled_fde=means['zero_weights']['fde'],
            best_trained_ade_ratio=means['learned']['ade']/means['zero_weights']['ade'],
            last_selected_is_trained=final['selected_is_trained'],learned_benefit=final['learned_benefit'],
            ordinary_acceptance_pass=final['ordinary_gates_pass'],tenfold_achieved=False,
            metric_units='32-unit observedobjectspan; notsourcepixels',validation_is_adaptively_reused=True,
            attached_diagnostic_weights=inference['model_sha256'],
            attached_model_note='v3 realtrainedweights; failedvalidationandvisualgates. v4fallbacknotmisrepresented aslearned.'),
        memory=dict(gaussians=fit_record['gaussians'],fitted_states=fit_record['states'],forecast_states=learned_record['states'],
            fitted_record_bytes=fit_record['packet_bytes'],forecast_record_bytes=learned_record['packet_bytes'],
            fitted_position_replay_error=fit_record['position_replay_max_error'],forecast_position_replay_error=learned_record['position_replay_max_error'],
            recurrent_tensor_bytes=resume['recurrent_state_tensor_bytes'],continuation_file_bytes=resume['checkpoint_bytes'],
            resume_new_steps=resume['smoke_new_steps'],resume_mesh_error=resume['resume_vertices_max_error'],
            resume_seed_reads=resume['resume_seed_reads'],resume_image_reads=resume['resume_image_reads'],
            warning='Expandedrecordsarenotcompression. Fixedasset/weights, solver, renderer andruntime excluded fromcurrentstatecounts.'),
        visual_quality_pass=visual['accepted'],quality_matched_efficiency_demonstrated=False,
        videos=dict(reconstruction='dense_fit_loop/v2/source_fit_comparison_7s.mp4',
                    learned_diagnostic='attached_vector_weights/v2/evaluation/conditional_comparison_7s_held.mp4',
                    temporal_scope='Seven-second held inspections of33fitted or13forecaststates; notsevensecondsnewgeneration.'),
        evidence_sha256=inputs)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(output),sha256=digest(output),learning=report['learning']),indent=2))


if __name__=='__main__': main()
