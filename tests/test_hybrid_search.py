"""Exercise notebook retrieval without downloads, embeddings, or LLM calls."""
import contextlib
import io
import json
from pathlib import Path
import unittest

from langchain_core.documents import Document

NOTEBOOK = Path(__file__).resolve().parents[1] / 'RAG Implementation.ipynb'


class Store:
    def __init__(self, docs):
        self.docs = docs
        self.semantic_docs = docs[:1]

    def get(self, limit, offset, include):
        docs = self.docs[offset:offset + limit]
        return dict(ids=[str(offset + i) for i in range(len(docs))],
                    documents=[d.page_content for d in docs],
                    metadatas=[d.metadata for d in docs])

    def as_retriever(self, search_kwargs):
        return self

    def invoke(self, query):
        return self.semantic_docs


def initialize(docs):
    notebook = json.loads(NOTEBOOK.read_text())
    source = ''.join(notebook['cells'][11]['source'])
    source = source[source.index('# Build BM25'):]
    env = {'vectorstore': Store(docs)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, 'hybrid notebook cell', 'exec'), env)
    env['_reranker'] = type('Reranker', (), {'predict': lambda self, pairs, **kwargs: list(range(len(pairs), 0, -1))})()
    return env


def doc(text, source='paper-a'):
    return Document(page_content=text, metadata={'source': source})


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.a = doc('Semantic language architecture')
        self.b = doc('REALM retrieval pretraining', 'paper-b')
        self.c = doc('Image classification convolution', 'paper-c')
        self.env = initialize([self.a, self.b, self.c, self.b])

    def test_keyword_adds_result_missed_by_semantic_search(self):
        self.assertEqual(self.env['keyword_search']('realm!'), [self.b])
        results = self.env['hybrid_search']('realm!')
        self.assertIn(self.a, results)
        self.assertIn(self.b, results)
        self.assertEqual(len(self.env['bm25_docs']), 3)

    def test_no_keyword_match_falls_back_to_semantic(self):
        self.assertEqual(self.env['keyword_search']('unknownterm'), [])
        self.assertEqual(self.env['hybrid_search']('unknownterm'), [self.a])

    def test_fusion_rewards_agreement_without_duplicate_votes(self):
        result = self.env['fuse_rankings']([self.a, self.a, self.b], [self.b, self.c])
        self.assertEqual(result[0], self.b)
        self.assertEqual(len(result), 3)
        other_source = doc(self.a.page_content, 'another-paper')
        self.assertEqual(len(self.env['fuse_rankings']([self.a], [other_source])), 2)

    def test_result_limit_and_empty_query(self):
        docs = [doc(f'Passage {i}') for i in range(12)]
        self.assertEqual(len(self.env['fuse_rankings'](docs, docs[::-1])), 5)
        with self.assertRaises(ValueError):
            self.env['hybrid_search'](' ')
        with self.assertRaises(ValueError):
            initialize([doc('!!!')])

    def test_bibliography_filter_preserves_explicit_reference_queries(self):
        reference = doc('References\n- Smith et al. Transformer')
        body = doc('The Transformer uses attention.')
        filtered = self.env['useful_candidates']([reference, body], 'Explain Transformer')
        self.assertEqual(filtered, [body])
        self.assertEqual(self.env['useful_candidates']([reference], 'List references'), [reference])

    def test_reranker_reorders_candidates(self):
        self.env['_reranker'] = type('Reranker', (), {
            'predict': lambda self, pairs, **kwargs: [0.1, 0.9]})()
        self.assertEqual(self.env['rerank_candidates']('query', [self.a, self.b]), [self.b, self.a])

    def test_contribution_query_clarification(self):
        clarify = self.env['reranking_query']
        self.assertIn('What new method', clarify('What is the contribution of a new model?'))
        self.assertEqual(clarify('Who contributed to the paper?'), 'Who contributed to the paper?')
        self.assertEqual(clarify('Explain attention'), 'Explain attention')

    def test_pagination(self):
        env = initialize([doc(f'Token {i}') for i in range(1002)])
        self.assertEqual(len(env['bm25_docs']), 1002)

    def test_citations_keep_source_metadata(self):
        notebook = json.loads(NOTEBOOK.read_text())
        env = dict(self.env, question='realm', search_keywords=None)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][17]['source']), env)
        self.assertIn('paper-b', env['source_labels'])
        self.assertEqual(env['context'].count('paper-b'), 1)


if __name__ == '__main__':
    unittest.main()
