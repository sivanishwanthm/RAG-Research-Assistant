import unittest
from types import SimpleNamespace
from rag_evaluation import evaluate, evidence_recall, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(id='q1', question='Question?', should_answer=True, reviewed=True,
                         reference_answer='Answer', expected_evidence=[dict(source='paper',quote='supporting evidence')])
        self.docs = [SimpleNamespace(page_content='Some supporting evidence.',metadata={'source':'paper'})]

    def test_recall_requires_passage_not_just_source(self):
        self.assertEqual(evidence_recall(self.case,[dict(source='paper',passage='irrelevant')]),0)
        self.case['reviewed']=False
        self.assertIsNone(evidence_recall(self.case,[]))

    def test_answer_records_and_missing_review(self):
        rows=evaluate([self.case],lambda q:self.docs,lambda *args:'Answer [1]','REFUSE','FAIL')
        self.assertEqual(rows[0]['evidence_recall_at_5'],1)
        metrics={r['metric']:r for r in summarize(rows,{})}
        self.assertIsNone(metrics['Answer correctness']['value'])
        self.assertEqual(metrics['Answer correctness']['evaluated'],0)
        metrics={r['metric']:r for r in summarize(rows,{'q1':dict(correct=1,grounded=0,citation_accuracy=0.5)})}
        self.assertEqual(metrics['Answer correctness']['value'],1)
        self.assertEqual(metrics['Citation accuracy']['value'],0.5)

    def test_refusal_and_error_are_separate(self):
        cases=[self.case,dict(self.case,id='q2',should_answer=False)]
        rows=evaluate(cases,lambda q:self.docs,lambda *args:'REFUSE','REFUSE','FAIL')
        m={x['metric']:x['value'] for x in summarize(rows,{})}
        self.assertEqual(m['False refusal rate (lower is better)'],1)
        self.assertEqual(m['Unsupported answer rate (lower is better)'],0)
        rows[0]['status']='error'
        m={x['metric']:x for x in summarize(rows,{})}
        self.assertIsNone(m['False refusal rate (lower is better)']['value'])
        self.assertEqual(m['Error rate (lower is better)']['value'],0.5)

    def test_rate_limit_preserves_partial_results(self):
        def answer(q,e,d):
            d.append('verification.error: RateLimitError')
            return 'FAIL'
        rows=evaluate([self.case,self.case],lambda q:self.docs,answer,'REFUSE','FAIL')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'error')

    def test_invalid_review_rejected(self):
        rows=evaluate([self.case],lambda q:self.docs,lambda *a:'Answer','REFUSE','FAIL')
        with self.assertRaises(ValueError):
            summarize(rows,{'q1':{'correct':2}})
