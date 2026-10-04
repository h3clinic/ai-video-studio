import unittest
from cloud.record_apple_repair import completed_report


class RepairReportingTests(unittest.TestCase):
    def test_loading_report_is_not_completed_model_run(self):
        for report in (None,{},dict(status='loading',frames=[]),dict(status='rendering',frames=[{}])):
            self.assertFalse(completed_report(report))

    def test_only_terminal_report_with_real_frame_evidence_counts(self):
        report=dict(status='blocked_no_safe_insertion',output_frames=5,output_sha256='a'*64)
        self.assertTrue(completed_report(report))
        self.assertFalse(completed_report(dict(report,output_frames=0)))
        self.assertFalse(completed_report(dict(report,status='failed')))


if __name__=='__main__':unittest.main()
