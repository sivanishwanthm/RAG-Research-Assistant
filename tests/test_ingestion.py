import unittest
from unittest.mock import patch, MagicMock
from paper_ingestion import arxiv_url, parse_pdf, MAX_BYTES
from app_backend import ResearchService
from rag_evaluation import make_evidence

class IngestionTests(unittest.TestCase):
    def test_arxiv_allowlist(self):
        self.assertEqual(arxiv_url('https://arxiv.org/abs/1706.03762'), 'https://arxiv.org/pdf/1706.03762')
        for url in ['http://arxiv.org/pdf/1706.03762', 'https://localhost/pdf/1706.03762',
                    'https://arxiv.org.evil.com/pdf/1706.03762', 'https://arxiv.org/pdf/1706.03762?next=x']:
            with self.assertRaises(ValueError): arxiv_url(url)

    def test_bad_pdf(self):
        with self.assertRaises(ValueError): parse_pdf(b'not a pdf', 'paper.pdf')
        with self.assertRaises(ValueError): parse_pdf(b'%PDF-' + b'x' * MAX_BYTES, 'paper.pdf')

    def test_page_metadata_and_stable_ids(self):
        reader = MagicMock()
        reader.is_encrypted = False
        reader.metadata = {'/Title': 'Research paper'}
        reader.pages = [MagicMock(), MagicMock()]
        reader.pages[0].extract_text.return_value = '1 Introduction\n' + 'Attention models use learned projections. ' * 12
        reader.pages[1].extract_text.return_value = '2 Method\n' + 'Evidence supports this research method. ' * 12
        with patch('pypdf.PdfReader', return_value=reader):
            docs, ids = parse_pdf(b'%PDF-mock', '../paper.pdf')
            self.assertEqual(ids, parse_pdf(b'%PDF-mock', '../paper.pdf')[1])
            self.assertEqual(docs[-1].metadata['page'], 2)
            self.assertEqual(docs[-1].metadata['section'], '2 Method')
            self.assertEqual(make_evidence(docs)[0]['title'], 'Research paper')

    def test_rewrite_preserves_authorship(self):
        self.assertEqual(ResearchService.rewrite('Who contributed to BERT?'), 'Who contributed to BERT?')
        self.assertIn('new method', ResearchService.rewrite('Main contribution of BERT?'))

    def test_ingestion_retry_only_adds_missing_chunks(self):
        import threading
        service = ResearchService.__new__(ResearchService)
        service.lock = threading.RLock()
        service.store = MagicMock()
        service.refresh = MagicMock()
        service.store.get.return_value = {'ids': ['existing']}
        with patch('paper_ingestion.parse_pdf', return_value=(['doc1','doc2'], ['existing','new'])):
            self.assertEqual(service.ingest(b'data','paper.pdf'), 1)
        service.store.add_documents.assert_called_once_with(['doc2'], ids=['new'])
        service.refresh.assert_called_once()

    def test_failed_ingestion_refreshes_partial_index(self):
        import threading
        service = ResearchService.__new__(ResearchService)
        service.lock = threading.RLock()
        service.store = MagicMock()
        service.refresh = MagicMock()
        service.store.get.return_value = {'ids': []}
        service.store.add_documents.side_effect = RuntimeError('embedding failure')
        with patch('paper_ingestion.parse_pdf', return_value=(['doc'], ['new'])):
            with self.assertRaises(RuntimeError): service.ingest(b'data','paper.pdf')
        service.refresh.assert_called_once()
