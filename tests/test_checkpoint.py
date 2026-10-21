import csv
import tempfile
import unittest
from pathlib import Path
from evaluation_checkpoint import open_run, save_run, update_row, rate_limited


class CheckpointTests(unittest.TestCase):
    def test_completed_rows_resume_and_config_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            report = open_run(directory, {'model': 'a'}, [{'id': 'q1'}])
            report['rows'] = [dict(case_id='q1', mode='hybrid', status='answered')]
            save_run(directory, report)
            self.assertEqual(open_run(directory, {'model': 'a'}, [{'id': 'q1'}])['rows'], report['rows'])
            with self.assertRaises(ValueError):
                open_run(directory, {'model': 'b'}, [{'id': 'q1'}])

    def test_review_edits_and_previous_attempts_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            report = open_run(directory, {}, [])
            failed = dict(case_id='q1', mode='hybrid', status='api_error', human_correct='')
            report['rows'] = [failed]
            save_run(directory, report)
            path = Path(directory) / 'review.csv'
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            rows[0]['human_correct'] = '1'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader(); writer.writerows(rows)
            update_row(report['rows'], dict(case_id='q1', mode='hybrid', status='answered', human_correct=''))
            save_run(directory, report)
            self.assertEqual(report['rows'][0]['human_correct'], '1')
            self.assertEqual(report['rows'][0]['prior_attempts'][0]['status'], 'api_error')

    def test_rate_limit_diagnostic(self):
        self.assertTrue(rate_limited({'diagnostics': ['verification.error: RateLimitError']}))
        self.assertFalse(rate_limited({'diagnostics': ['draft.quote_mismatch']}))
