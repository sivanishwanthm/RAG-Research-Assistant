import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from app_backend import ResearchService

class RetrievalIntentTests(unittest.TestCase):
    def test_combined_question_preserves_subject(self):
        for subject in ('colbert', 'BERT', 'a diffusion model'):
            queries = ResearchService.retrieval_queries(f'What is {subject} and how does it work?')
            self.assertEqual(len(queries), 2)
            self.assertEqual(queries[1], f'{subject} architecture mechanism computation steps')
        self.assertEqual(ResearchService.retrieval_queries('Who invented ColBERT?'), ['Who invented ColBERT?'])

    def test_overview_and_mechanism_are_retained_without_duplicates(self):
        docs = [SimpleNamespace(page_content=str(i), metadata={'source':'paper'}) for i in range(7)]
        service = ResearchService.__new__(ResearchService)
        service._search_one = Mock(side_effect=[docs[:5], [docs[0], *docs[3:]]])
        result = service.search('What is a model and how does it work')
        self.assertEqual([d.page_content for d in result], ['0','3','4','5','6'])
        self.assertEqual(service._search_one.call_count, 2)

    def test_attention_terminology_preserves_intent(self):
        self.assertEqual(ResearchService.rewrite('What is query key and value in transformers'),
                         'What are queries, keys, and values in Transformer attention?')
        self.assertIn('Why', ResearchService.rewrite('Why use query key and value in transformers?'))
        unrelated = 'What is query key and value in a database?'
        self.assertEqual(ResearchService.rewrite(unrelated), unrelated)
        single = 'What is a key in transformers?'
        self.assertEqual(ResearchService.rewrite(single), single)

    def test_incomplete_context_stays_in_same_source(self):
        from langchain_core.documents import Document
        service = ResearchService.__new__(ResearchService)
        first = Document(page_content='Output is computed as a weighted sum', metadata={'source':'a','page':1})
        second = Document(page_content='of the values.', metadata={'source':'a','page':2})
        service.docs = [first, second]
        service.positions = {service.key(first):0}
        expanded = service.complete_short_passage(first)
        self.assertIn('of the values.', expanded.page_content)
        self.assertNotIn('page', expanded.metadata)
        service.docs[1] = Document(page_content='Unrelated.', metadata={'source':'b'})
        self.assertEqual(service.complete_short_passage(first).page_content, first.page_content)
