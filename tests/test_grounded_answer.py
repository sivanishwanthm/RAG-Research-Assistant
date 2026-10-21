import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from grounded_answer import grounded_answer, REFUSAL, CHECK_FAILED


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(content=value if isinstance(value, str) else json.dumps(value))


class GroundingTests(unittest.TestCase):
    def setUp(self):
        sleeper = patch("grounded_answer.time.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)
        self.evidence = [dict(citation='[1]', source='https://paper.test/a',
                             passage='The architecture uses attention mechanisms.')]
        self.draft = {'claims': [{'text': 'The architecture uses attention.',
                       'supports': [{'citation': '[1]', 'quote': 'uses attention mechanisms'}]}]}

    def test_out_of_scope_stops_before_generation(self):
        llm = FakeLLM({'answerable': False})
        self.assertEqual(grounded_answer(llm, 'Explain human physiology', self.evidence), REFUSAL)
        self.assertEqual(llm.calls, 1)

    def test_valid_answer_and_unique_source(self):
        llm = FakeLLM({'answerable': True}, self.draft,
                      {'supported': True, 'answers_question': True})
        answer = grounded_answer(llm, 'What does the architecture use?', self.evidence * 2)
        self.assertIn('The architecture uses attention. [1]', answer)
        self.assertEqual(answer.count('https://paper.test/a'), 1)

    def test_fabricated_quote_or_label_rejected(self):
        for field, value in [('quote', 'The human heart pumps blood'), ('citation', '[99]')]:
            draft = json.loads(json.dumps(self.draft))
            draft['claims'][0]['supports'][0][field] = value
            llm = FakeLLM({'answerable': True}, draft, draft)
            self.assertEqual(grounded_answer(llm, 'Question', self.evidence), REFUSAL)
            self.assertEqual(llm.calls, 3)

    def test_real_quote_does_not_bypass_entailment_check(self):
        self.draft['claims'][0]['text'] = 'The architecture cures disease.'
        llm = FakeLLM({'answerable': True}, self.draft,
                      {'supported': False, 'answers_question': True}, {'claims': []})
        self.assertEqual(grounded_answer(llm, 'Question', self.evidence), REFUSAL)

    def test_specific_diagnostics(self):
        self.draft['claims'][0]['supports'][0]['quote'] = 'invented quote'
        llm = FakeLLM({'answerable': True}, self.draft, self.draft)
        diagnostics = []
        result = grounded_answer(llm, 'Question', self.evidence, diagnostics)
        self.assertEqual(result, REFUSAL)
        self.assertIn('draft.quote_mismatch', diagnostics[0])

    def test_malformed_or_failed_calls_withhold_draft(self):
        for bad in ['not json', RuntimeError('API unavailable')]:
            llm = FakeLLM({'answerable': True}, self.draft, bad)
            self.assertEqual(grounded_answer(llm, 'Question', self.evidence), CHECK_FAILED)

    def test_personal_fridge_refuses_even_with_matching_example(self):
        llm = FakeLLM()
        evidence = [dict(citation='[1]', source='paper',
                        passage='In the refrigerator you see a cup and an egg.')]
        answer = grounded_answer(llm, 'What ingredients are in my refrigerator?', evidence)
        self.assertEqual(answer, REFUSAL)
        self.assertEqual(llm.calls, 0)

    def test_single_repair_is_validated_then_audited(self):
        bad = json.loads(json.dumps(self.draft))
        bad['claims'][0]['supports'][0]['quote'] = 'not present'
        llm = FakeLLM({'answerable': True}, bad, self.draft,
                      {'supported': True, 'answers_question': True})
        answer = grounded_answer(llm, 'What does the architecture use?', self.evidence)
        self.assertIn('The architecture uses attention. [1]', answer)
        self.assertEqual(llm.calls, 4)

    def test_repaired_quote_cannot_bypass_audit(self):
        bad = json.loads(json.dumps(self.draft))
        bad['claims'][0]['supports'][0]['quote'] = 'not present'
        llm = FakeLLM({'answerable': True}, bad, self.draft,
                      {'supported': False, 'answers_question': True}, {'claims': []})
        self.assertEqual(grounded_answer(llm, 'Question', self.evidence), REFUSAL)

    def test_ablation_cannot_support_original_model(self):
        from grounded_answer import _draft_error
        quote = "No NSP: A model trained with MLM but without NSP."
        evidence = [dict(citation='[1]', source='paper', passage=quote)]
        draft = {'claims': [{'text': 'BERT uses MLM and NSP.',
                            'supports': [{'citation': '[1]', 'quote': quote}]}]}
        self.assertIn('draft.variant_evidence', _draft_error(draft, evidence))
        draft['claims'][0]['text'] = 'The ablation variant trains without NSP.'
        self.assertIsNone(_draft_error(draft, evidence))

    def test_rate_limit_retry_is_bounded(self):
        from grounded_answer import invoke_with_retries
        class RateLimitError(Exception):
            status_code = 429
        model = FakeLLM(RateLimitError(), {'ok': True})
        response = invoke_with_retries(model, [])
        self.assertIn('true', response.content)
        self.assertEqual(model.calls, 2)
        model = FakeLLM(RateLimitError(), RateLimitError(), RateLimitError())
        with self.assertRaises(RateLimitError):
            invoke_with_retries(model, [])
        self.assertEqual(model.calls, 3)

    def test_non_rate_limit_is_not_retried(self):
        from grounded_answer import invoke_with_retries
        model = FakeLLM(ValueError('bad input'))
        with self.assertRaises(ValueError):
            invoke_with_retries(model, [])
        self.assertEqual(model.calls, 1)

    def test_authentication_has_specific_diagnostic(self):
        class AuthenticationError(Exception):
            status_code = 401
        diagnostics = []
        model = FakeLLM(AuthenticationError())
        self.assertEqual(grounded_answer(model, 'Question', self.evidence, diagnostics), CHECK_FAILED)
        self.assertIn('authentication.failed', diagnostics[0])
        self.assertEqual(model.calls, 1)

    def test_bert_selects_evidence_without_copying_text(self):
        evidence = [dict(citation='[2]', source='bert-paper', passage=
            "BERT uses a masked language model. Inaddition we use next sentence prediction.")]
        draft = {'claims': [{'text': 'BERT uses masked language modeling and next sentence prediction.',
                            'supports': [{'evidence_id': 'E1'}]}]}
        model = FakeLLM({'answerable': True}, draft,
                        {'supported': True, 'answers_question': True})
        answer = grounded_answer(model, 'What pretraining objectives does BERT use?', evidence)
        self.assertIn('next sentence prediction. [2]', answer)
        self.assertEqual(model.calls, 3)

    def test_invalid_evidence_id_cannot_pass(self):
        draft = {'claims': [{'text': 'Claim', 'supports': [{'evidence_id':'E99'}]}]}
        model = FakeLLM({'answerable':True}, draft, draft)
        self.assertEqual(grounded_answer(model,'Question',self.evidence), REFUSAL)

    def test_selected_ids_cannot_override_source_text(self):
        from grounded_answer import _resolve_evidence_ids
        passages = [dict(evidence_id='E1',citation='[1]',passage='Real text')]
        result = _resolve_evidence_ids({'claims':[{'supports':[{
            'evidence_id':'E1','quote':'invented','citation':'[99]'}]}]},passages)
        self.assertEqual(result['claims'][0]['supports'][0]['quote'],'Real text')
        self.assertEqual(result['claims'][0]['supports'][0]['citation'],'[1]')

    def test_missing_evidence_and_non_boolean_approval(self):
        llm = FakeLLM()
        self.assertEqual(grounded_answer(llm, 'Question', []), REFUSAL)
        self.assertEqual(llm.calls, 0)
        llm = FakeLLM({'answerable': 'true'})
        self.assertEqual(grounded_answer(llm, 'Question', self.evidence), REFUSAL)


if __name__ == '__main__':
    unittest.main()

class AuditRepairTests(unittest.TestCase):
    @patch('grounded_answer.time.sleep')
    def test_repaired_claims_must_pass_second_audit(self, sleep):
        evidence = [dict(citation='[1]',source='paper',passage='The architecture uses attention mechanisms.')]
        draft = {'claims':[{'text':'The architecture uses attention mechanisms.', 'supports':[{'evidence_id':'E1'}]}]}
        for supported in (True, False):
            llm = FakeLLM({'answerable':True}, draft,
                          {'supported':False,'answers_question':True,'reason':'Check support.'},
                          draft, {'supported':supported,'answers_question':True})
            answer = grounded_answer(llm, 'What does the architecture use?', evidence)
            self.assertEqual(answer == REFUSAL, not supported)
            self.assertEqual(llm.calls, 5)

class ProviderDiagnosticTests(unittest.TestCase):
    @patch('grounded_answer.time.sleep')
    def test_configuration_errors_withhold_answer_without_raw_body(self, sleep):
        error = RuntimeError('sensitive provider response')
        error.status_code = 404
        diagnostics = []
        answer = grounded_answer(FakeLLM(error), 'Question',
            [dict(citation='[1]',source='paper',passage='Evidence')], diagnostics)
        self.assertEqual(answer, CHECK_FAILED)
        self.assertIn('provider_configuration', diagnostics[0])
        self.assertNotIn('sensitive', str(diagnostics))
