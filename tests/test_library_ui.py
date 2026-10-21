import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest

SCRIPT = '''from app_workspaces import library_view
from library_test_service import service
library_view(lambda: service)
'''

class LibraryUITests(unittest.TestCase):
    def test_empty_library_explains_next_step(self):
        service = SimpleNamespace(library=lambda: [], paper_count=0)
        with patch.dict('sys.modules', {'library_test_service':SimpleNamespace(service=service)}):
            app=AppTest.from_string(SCRIPT).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No papers indexed yet' in x.value for x in app.info))
            self.assertEqual(app.session_state['paper_count'], 0)

    def test_failed_download_does_not_block_next_paper(self):
        service=SimpleNamespace(library=lambda: [], paper_count=0, ingest=Mock(return_value=4))
        with patch.dict('sys.modules', {'library_test_service':SimpleNamespace(service=service)}), \
             patch('app_workspaces.download_arxiv', side_effect=[ValueError('Unavailable'),(b'data','https://arxiv.org/pdf/1810.04805')]):
            app=AppTest.from_string(SCRIPT).run()
            app.multiselect[0].set_value(app.multiselect[0].options[:2]).run()
            next(b for b in app.button if b.label=='Index arXiv papers').click().run()
            self.assertFalse(app.exception)
            service.ingest.assert_called_once_with(b'data','1810.04805.pdf','https://arxiv.org/pdf/1810.04805')
            self.assertTrue(any('Unavailable' in w.value for w in app.warning))
            self.assertTrue(any(s.label=='1 of 2 papers processed' for s in app.status))
