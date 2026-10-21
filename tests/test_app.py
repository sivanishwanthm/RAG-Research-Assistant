import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()
        default_service = patch("app_backend.ResearchService", return_value=SimpleNamespace(library=lambda: [], paper_count=0))
        default_service.start()
        self.addCleanup(default_service.stop)

    def test_initial_page_and_blank_question(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        self.assertFalse(app.exception)
        app.button(key='submit').click().run()
        self.assertTrue(app.warning)
        self.assertFalse(app.exception)

    def test_question_answer_and_evidence(self):
        result=dict(question='Explain attention',answer='Supported answer [1]',
                    evidence=[dict(citation='[1]',source='https://arxiv.org/pdf/1706.03762',passage='Evidence text')],
                    diagnostics=[],status='answered',seconds=1.0)
        service=SimpleNamespace(ask=lambda q, **kwargs:result,paper_count=30)
        with patch('app_backend.ResearchService',return_value=service):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            app.text_input(key='personal_api_key').set_value('test-key').run()
            app.text_area[0].set_value('Explain attention')
            app.button(key='submit').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Supported answer' in x.value for x in app.markdown))
            self.assertEqual(sum('Passage' in e.label for e in app.expander),1)

    def test_new_question_returns_from_library_and_clears_answer(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.session_state['result'] = {'question': 'Old answer'}
        app.radio(key='workspace').set_value('Library').run()
        app.button(key='clear').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key='workspace').value, 'Research')
        self.assertEqual(app.text_area(key='question').value, '')
        self.assertNotIn('result', app.session_state)

    def test_example_returns_from_summary_and_clears_stale_result(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.radio(key='workspace').set_value('Paper Summary').run()
        app.session_state['result'] = {'question': 'Old answer'}
        app.button(key='example1').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key='workspace').value, 'Research')
        self.assertEqual(app.text_area(key='question').value, 'What pretraining objectives does BERT use?')
        self.assertNotIn('result', app.session_state)

    def test_clear_unsent_draft(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.text_area(key='question').set_value('Unsent draft')
        app.button(key='clear').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_area(key='question').value, '')

    def test_personal_key_is_forwarded_and_can_be_cleared(self):
        from unittest.mock import Mock
        service = SimpleNamespace(ask=Mock(return_value=dict(question='Test', answer='No evidence',
            evidence=[], diagnostics=[], status='refused', seconds=0)), paper_count=0)
        with patch('app_backend.ResearchService', return_value=service):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            app.text_input(key='personal_api_key').set_value('test-key-not-real').run()
            app.text_area(key='question').set_value('Test')
            app.button(key='submit').click().run()
            service.ask.assert_called_once_with('Test', api_key='test-key-not-real', provider='Groq', model=None, base_url=None)
            app.button(key='clear_api_key').click().run()
            self.assertEqual(app.text_input(key='personal_api_key').value, '')
            app.button(key='submit').click().run()
            self.assertEqual(service.ask.call_count, 1)
            self.assertTrue(app.warning)
            self.assertFalse(app.exception)

    def test_provider_switch_clears_credentials_and_test_uses_selection(self):
        from unittest.mock import Mock
        client = Mock()
        client.invoke.return_value = SimpleNamespace(content='{"ok":true}')
        with patch('llm_provider.create_client', return_value=client) as create:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            app.text_input(key='personal_api_key').set_value('old-test-key').run()
            app.selectbox(key='llm_provider').set_value('Anthropic').run()
            self.assertEqual(app.text_input(key='personal_api_key').value, '')
            app.text_input(key='llm_model').set_value('test-model').run()
            app.text_input(key='personal_api_key').set_value('new-test-key').run()
            app.button(key='test_llm').click().run()
            create.assert_called_once_with('Anthropic','test-model','new-test-key',None)
            self.assertTrue(app.success)
            self.assertFalse(app.exception)

    def test_missing_user_key_does_not_use_server_key(self):
        from unittest.mock import Mock
        service = SimpleNamespace(ask=Mock(), paper_count=0)
        with patch.dict('os.environ', {'GROQ_API_KEY':'server-test-key'}), patch('app_backend.ResearchService', return_value=service):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            self.assertFalse(any(r.key == 'credential_mode' for r in app.radio))
            app.text_area(key='question').set_value('Explain attention')
            app.button(key='submit').click().run()
            self.assertTrue(app.warning)
            service.ask.assert_not_called()
            self.assertFalse(app.exception)

    def test_research_shortcuts_and_provider_state(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Add your own API key' in c.value for c in app.caption))
        app.button(key='research_prompt_Understand a method').click().run()
        self.assertEqual(app.text_area(key='question').value, 'What is query key and value in transformers')
        app.button(key='research_library').click().run()
        self.assertEqual(app.radio(key='workspace').value, 'Library')
        app.button(key='clear').click().run()
        app.button(key='research_summaries').click().run()
        self.assertEqual(app.radio(key='workspace').value, 'Paper Summary')
        self.assertFalse(app.exception)
