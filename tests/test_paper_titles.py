import unittest
from paper_ingestion import paper_title

class PaperTitleTests(unittest.TestCase):
    def test_catalog_covers_versioned_arxiv_sources(self):
        self.assertEqual(paper_title('https://arxiv.org/pdf/1706.03762v7.pdf'), 'Attention Is All You Need')
        self.assertIn('ColBERT', paper_title('https://arxiv.org/pdf/2004.12832'))

    def test_uploaded_metadata_and_title_page(self):
        self.assertEqual(paper_title('upload:abc/paper.pdf','A custom paper'), 'A custom paper')
        self.assertEqual(paper_title('upload:abc/attention.pdf','attention.pdf',
                         'Notice\nAttention Is All You Need\nAuthors'), 'Attention Is All You Need')
        self.assertEqual(paper_title('upload:abc/paper.pdf','paper.pdf',
                         'We compare to Attention Is All You Need.'), 'paper.pdf')

    def test_unknown_paper_does_not_borrow_cited_title(self):
        self.assertEqual(paper_title('https://arxiv.org/pdf/9999.99999', None,
                         'Attention Is All You Need'), 'Title unavailable')
