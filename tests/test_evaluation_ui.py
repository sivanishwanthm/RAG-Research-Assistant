import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

SCRIPT = '''from app_workspaces import evaluation_view
evaluation_view(lambda: None)
'''

class EvaluationUITests(unittest.TestCase):
    def test_empty_state_and_new_check(self):
        with tempfile.TemporaryDirectory() as directory, patch('app_workspaces.ROOT', Path(directory)):
            app = AppTest.from_string(SCRIPT).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No evaluations yet' in i.value for i in app.info))
            app.radio(key='evaluation_section').set_value('Run a new check').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.button(key='start_evaluation').label, 'Start four-question check')

    def test_review_updates_scores_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as directory, patch('app_workspaces.ROOT', Path(directory)):
            folder = Path(directory) / 'evaluation_results' / 'example'
            folder.mkdir(parents=True)
            report = {'cases':[{'id':'q1'}, {'id':'q2'}], 'results':[dict(id='q1',question='What is attention?',
                should_answer=True, reviewed=True, reference_answer='Reference', answer='Answer [1]',
                status='answered', evidence=[], evidence_recall_at_5=1, seconds=2, diagnostics=[])]}
            path = folder / 'results.json'
            path.write_text(json.dumps(report))
            app = AppTest.from_string(SCRIPT).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('stopped early' in w.value for w in app.warning))
            self.assertTrue(any(m.value == 'Not reviewed' for m in app.metric))
            for label in ['Is the answer correct?', 'Are the claims supported?', 'Do the citations support their claims?']:
                next(s for s in app.selectbox if s.label == label).set_value(1.)
            next(b for b in app.button if b.label == 'Save review').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(json.loads((folder/'reviews.json').read_text())['q1']['correct'], 1.)
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertTrue(app.success)
            self.assertTrue(any('1 of 1 answered questions fully reviewed' in c.value for c in app.caption))

    def test_new_check_can_be_viewed_without_browser_refresh(self):
        script = '''from app_workspaces import evaluation_view
from types import SimpleNamespace
def ask(question, **kwargs):
    return dict(question=question, answer='Answer', evidence=[], diagnostics=[], status='answered', seconds=1)
evaluation_view(lambda: SimpleNamespace(ask=ask))
'''
        with tempfile.TemporaryDirectory() as directory, patch('app_workspaces.ROOT', Path(directory)):
            (Path(directory)/'evaluation_reference_labels.json').write_text(json.dumps({
                key: {'reviewed':True,'expected_evidence':[]} for key in ('q01','q02','q05')}))
            app = AppTest.from_string(script).run()
            app.radio(key='evaluation_section').set_value('Run a new check').run()
            app.button(key='start_evaluation').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
            app.radio(key='evaluation_section').set_value('View results').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('4 of 4 questions attempted' in c.value for c in app.caption))
