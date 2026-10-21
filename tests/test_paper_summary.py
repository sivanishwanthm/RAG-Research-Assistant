import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from langchain_core.documents import Document
from app_backend import ResearchService

class SummaryTests(unittest.TestCase):
    def test_passages_are_restricted_to_selected_paper(self):
        service = ResearchService.__new__(ResearchService)
        service.docs = [Document(page_content=f'Paper {i} content.', metadata={'source':str(i)}) for i in range(2)]
        service.positions = {service.key(d):i for i,d in enumerate(service.docs)}
        service.reranker = Mock()
        service.reranker.predict.side_effect = lambda pairs, **kwargs: [1.] * len(pairs)
        result = service.summary_passages('0')
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].metadata['source'],'0')
        for call in service.reranker.predict.call_args_list:
            self.assertTrue(all('Paper 0' in text for _, text in call.args[0]))
        with self.assertRaises(ValueError): service.summary_passages('missing')

    def test_selected_paper_and_credentials_forwarded(self):
        service = SimpleNamespace(library=lambda:[dict(source='paper-a',title='Paper A',chunks=3),dict(source='paper-b',title='Paper B',chunks=2)],
            summarize_paper=Mock(return_value=dict(answer='Summary [1]', status='answered', evidence=[], diagnostics=[])))
        script = '''from app_workspaces import paper_summary_view
from summary_test_service import service
paper_summary_view(lambda: service, api_key='test-key', llm_options={'provider':'OpenAI','model':'test-model'})
'''
        with patch.dict('sys.modules', {'summary_test_service':SimpleNamespace(service=service)}):
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            service.summarize_paper.assert_not_called()
            app.button(key='summarize_title_paper-a').click().run()
            service.summarize_paper.assert_called_once_with('paper-a',api_key='test-key',provider='OpenAI',model='test-model')
            self.assertTrue(any('Summary [1]' in m.value for m in app.markdown))
            app.selectbox(key='summary_paper').set_value('paper-b').run()
            self.assertEqual(service.summarize_paper.call_count, 2)
            self.assertEqual(service.summarize_paper.call_args.args[0], 'paper-b')
            app.run()
            self.assertEqual(service.summarize_paper.call_count, 2)
            self.assertFalse(app.exception)

    def test_summary_keeps_opening_and_excludes_short_figure_pointers(self):
        service = ResearchService.__new__(ResearchService)
        abstract = Document(page_content='We propose a new adaptation method. ' * 14, metadata={'source':'a','title':'Paper A'})
        pointer = Document(page_content='H.1 RESULTS\nSee Figure 6 for results.', metadata={'source':'a'})
        method = Document(page_content='1 INTRODUCTION\n' + 'This method freezes model weights. ' * 14, metadata={'source':'a'})
        service.docs = [abstract, method, pointer]
        service.positions = {service.key(d):i for i,d in enumerate(service.docs)}
        service.reranker = Mock()
        service.reranker.predict.side_effect = lambda pairs, **kwargs: [1.] * len(pairs)
        selected = service.summary_passages('a')
        self.assertEqual(selected[0].page_content, abstract.page_content)
        self.assertFalse(any(d.page_content == pointer.page_content for d in selected))

    def test_library_title_opens_summary_and_generates_once(self):
        import streamlit as st
        from pathlib import Path
        st.cache_resource.clear()
        service = SimpleNamespace(library=lambda:[dict(source='paper-a',title='Paper A',chunks=3)],
            summarize_paper=Mock(return_value=dict(answer='Summary [1]',status='answered',evidence=[],diagnostics=[])))
        with patch('app_backend.ResearchService',return_value=service):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
            app.text_input(key='personal_api_key').set_value('test-key').run()
            app.radio(key='workspace').set_value('Library').run()
            app.button(key='library_title_paper-a').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='workspace').value,'Paper Summary')
            self.assertEqual(service.summarize_paper.call_count,1)
            app.run()
            self.assertEqual(service.summarize_paper.call_count,1)
