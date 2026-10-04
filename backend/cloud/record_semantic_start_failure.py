"""Record preserved provider failures; no API calls or GPU work."""
from real_video.gaussian_program import Program,evidence,ROOT

def main():
    program=Program()
    try:
        program.record_experiment(dict(role='rendering',issue_key='rendering.related_part_elimination',
            outcome='failed_execution',
            hypothesis='Grounded fruit-part segmentation can select residual Gaussian samples outside the old fixed ellipse, enabling local repair without regenerating the full video.',
            observation='Implemented remote-only DINO/SAM diagnostic and delta state persistence. Python compilation and 13 existing focused regression tests passed. Neither remote launch reached model execution: initial start and one retry returned HTTP 500. Pod EXITED and default startup restored on both attempts. Temporary key disabled afterward. No new masks, generated video, quality result, or compute savings measured.',
            next_action='Resolve RunPod start failure before another launch; then validate segmentation coverage and protected receiver preservation on actual frames. Do not treat this unexecuted worker as a successful edit.',
            inputs=[evidence(ROOT,ROOT/p) for p in ['cloud/object_edit_semantic_worker.py','cloud/run_semantic_edit.py','research/prior_manifest.json','research/sweep_2026-10-04_semantic_elimination.json']],
            outputs=[evidence(ROOT,ROOT/p) for p in ['artifacts/cloud/object_edit_semantic_v1/control.json','artifacts/cloud/object_edit_semantic_v1_start_retry/control.json']]))
        program.export()
    finally:program.db.close()

if __name__=='__main__':main()
