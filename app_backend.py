"""Python app service using the notebook's retrieval settings and shared guardrails."""
import os
import re
import threading
from pathlib import Path
from time import perf_counter

from dotenv import dotenv_values
from grounded_answer import grounded_answer, REFUSAL, CHECK_FAILED
from rag_evaluation import make_evidence

ROOT = Path(__file__).resolve().parent


class ResearchService:
    def __init__(self, database=None):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from rank_bm25 import BM25Okapi

        database = Path(database or os.getenv('RAG_DATABASE', ROOT / 'chroma_db'))
        database.mkdir(parents=True, exist_ok=True)
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})
        self.store = Chroma(collection_name='langchain', persist_directory=str(database),
                            embedding_function=embedding)
        self.reranker = None
        self.lock = threading.RLock()
        self.refresh()

    def refresh(self):
        from langchain_core.documents import Document
        from rank_bm25 import BM25Okapi
        from paper_ingestion import paper_title
        unique, offset = {}, 0
        while True:
            batch = self.store.get(limit=1000, offset=offset, include=['documents', 'metadatas'])
            if not batch['ids']:
                break
            for text, metadata in zip(batch['documents'], batch['metadatas']):
                if text and self.tokenize(text):
                    metadata = dict(metadata or {})
                    metadata['title'] = paper_title(metadata.get('source', ''), metadata.get('title'), text)
                    doc = Document(page_content=text.strip(), metadata=metadata)
                    unique.setdefault(self.key(doc), doc)
            offset += len(batch['ids'])
        self.docs = list(unique.values())
        self.positions = {self.key(doc): i for i, doc in enumerate(self.docs)}
        tokens = [self.tokenize(d.page_content) for d in self.docs]
        self.terms = [set(t) for t in tokens]
        self.bm25 = BM25Okapi(tokens) if tokens else None
        self.semantic = self.store.as_retriever(search_kwargs={'k':150})
        self.paper_count = len({self.key(d)[0] for d in self.docs})

    @staticmethod
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.casefold())

    @staticmethod
    def key(doc):
        return str(doc.metadata.get('source') or 'Source unavailable'), doc.page_content.strip()

    def ingest(self, data, name, source=None):
        from paper_ingestion import parse_pdf
        documents, ids = parse_pdf(data, name, source)
        with self.lock:
            existing = set(self.store.get(ids=ids, include=[])['ids'])
            pending = [(doc, ident) for doc, ident in zip(documents, ids) if ident not in existing]
            try:
                for start in range(0, len(pending), 64):
                    batch = pending[start:start + 64]
                    self.store.add_documents([d for d, _ in batch], ids=[i for _, i in batch])
            finally:
                self.refresh()
            return len(pending)

    def library(self):
        papers = {}
        with self.lock:
            for doc in self.docs:
                source = self.key(doc)[0]
                item = papers.setdefault(source, dict(source=source, title=doc.metadata.get('title') or 'Title unavailable', chunks=0))
                item['chunks'] += 1
        return list(papers.values())

    @staticmethod
    def rewrite(query):
        # Deterministic intent expansion preserves the user's subject and avoids another API call.
        # Normalize common attention terminology without adding factual answer content.
        # Restrict this to attention/Transformer questions so database keys and values
        # retain their original meaning in other domains.
        if (re.search(r'\b(transformers?|attention)\b', query, re.I)
                and all(re.search(pattern, query, re.I) for pattern in
                        (r'\bquer(?:y|ies)\b', r'\bkeys?\b', r'\bvalues?\b'))):
            query = re.sub(r'\bquery\b', 'queries', query, flags=re.I)
            query = re.sub(r'\bkey\b', 'keys', query, flags=re.I)
            query = re.sub(r'\bvalue\b', 'values', query, flags=re.I)
            query = re.sub(r'\bwhat is queries\b', 'What are queries', query, flags=re.I)
            query = re.sub(r'\bqueries\s*,?\s*keys\s*,?\s*and\s+values\b',
                           'queries, keys, and values', query, flags=re.I)
            query = re.sub(r'\btransformers\b', 'Transformer', query, flags=re.I)
            if not re.search(r'\battention\b', query, re.I):
                query = query.rstrip(' ?.！!') + ' attention?'
        if (re.search(r'\b(contributions?|novelty)\b', query, re.I)
                and not re.search(r'\b(authors?|contributors?|who|acknowledg\w*)\b', query, re.I)):
            return query + ' What new method does the paper propose?'
        return query

    @staticmethod
    def retrieval_queries(query):
        original = ResearchService.rewrite(query)
        combined = re.fullmatch(
            r'\s*what\s+is\s+(.{1,120}?)\s+and\s+how\s+does\s+it\s+work[?.!\s]*',
            query, re.I)
        if combined:
            subject = combined[1].strip(' ?,.')
            return [original, f'{subject} architecture mechanism computation steps']
        return [original]

    def search(self, query):
        queries = self.retrieval_queries(query)
        if len(queries) == 1:
            return self._search_one(queries[0])
        # Reserve one overview, then four mechanism passages for the second intent.
        overview = self._search_one(queries[0])
        mechanism = self._search_one(queries[1])
        selected, seen = [], set()
        for doc in overview[:1] + mechanism + overview[1:]:
            if self.key(doc) not in seen:
                selected.append(doc)
                seen.add(self.key(doc))
            if len(selected) == 5:
                break
        return selected

    def complete_short_passage(self, doc):
        """Attach bounded same-source context when an indexed fragment ends mid-sentence.

        Index order is only a retrieval hint: label the extra text as related context,
        rather than asserting that it is the next sentence or the same PDF page.
        """
        text = doc.page_content.rstrip()
        if not text or len(text) >= 400 or text[-1] in '.!?':
            return doc
        position = self.positions.get(self.key(doc))
        if position is None or position + 1 >= len(self.docs):
            return doc
        following = self.docs[position + 1]
        if self.key(following)[0] != self.key(doc)[0] or len(following.page_content) > 1600:
            return doc
        from langchain_core.documents import Document
        metadata = dict(doc.metadata)
        metadata.pop('page', None)  # Combined text can span pages.
        metadata['context_expanded'] = True
        return Document(page_content=text + '\n\n[Related passage from the same paper]\n' + following.page_content,
                        metadata=metadata)

    def _search_one(self, query):
        if not self.docs:
            return []
        semantic = self.semantic.invoke(query)
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        matching = [i for i, terms in enumerate(self.terms) if set(tokens) & terms]
        matching.sort(key=lambda i: (-float(scores[i]), i))
        lexical = [self.docs[i] for i in matching[:150]]
        fused, documents = {}, {}
        for results in (semantic, lexical):
            seen, rank = set(), 0
            for doc in results:
                if (not re.search(r'\b(references|bibliography|citations|cites)\b',query,re.I)
                    and re.match(r'^\s*(?:references|bibliography)\s*(?:\n|$)',doc.page_content,re.I)):
                    continue
                key = self.key(doc)
                if key in seen or not key[1]:
                    continue
                seen.add(key); rank += 1
                documents[key] = doc
                fused[key] = fused.get(key,0) + .5/(60+rank)
        candidates = [documents[k] for k in sorted(fused,key=lambda k:-fused[k])[:300]]
        if not candidates:
            return []
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',device='cpu')
        scores = self.reranker.predict([(query,d.page_content) for d in candidates],
                                       batch_size=16,show_progress_bar=False)
        from langchain_core.documents import Document
        return [self.complete_short_passage(Document(page_content=d.page_content, metadata={**d.metadata,
                    'rerank_score': float(score), 'fusion_score': fused[self.key(d)]}))
                for d, score in sorted(zip(candidates,scores),key=lambda x:-float(x[1]))[:5]]

    def summary_passages(self, source):
        """Select complementary passages exclusively from one indexed paper."""
        candidates = [d for d in self.docs if self.key(d)[0] == source]
        if not candidates:
            raise ValueError('This paper is no longer indexed. Refresh the paper list.')
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        # Preserve the opening explanation: generic QA reranking can otherwise
        # prefer appendices and figure pointers over a paper's own contribution.
        opening = []
        for doc in candidates[:12]:
            if re.match(r'^\s*1[.\s]+INTRODUCTION', doc.page_content, re.I):
                break
            text = doc.page_content
            if len(text) >= 350 and '@' not in text and not re.match(r'^\s*(?:Figure|Table)\s*\d', text, re.I):
                opening.append(doc)
            if len(opening) == 2:
                break
        substantive = [d for d in candidates if len(d.page_content) >= 200
                       and not re.match(r'^\s*(?:references|bibliography)\b', d.page_content, re.I)]
        candidates = substantive or candidates
        aspects = ['What method or architecture does this paper propose?',
                   'What are the main experimental findings, datasets, and results?',
                   'What limitations or future work does the paper describe?',
                   'What problem and main contribution does this paper present?',
                   'What experiments evaluate the method?']
        selected = [self.complete_short_passage(doc) for doc in opening]
        seen = {self.key(doc) for doc in opening}
        title = candidates[0].metadata.get('title', '')
        for aspect in aspects:
            if len(selected) >= 5:
                break
            scores = self.reranker.predict([(title + ': ' + aspect, d.page_content) for d in candidates],
                                           batch_size=16, show_progress_bar=False)
            for doc, score in sorted(zip(candidates, scores), key=lambda item: -float(item[1])):
                if self.key(doc) not in seen:
                    seen.add(self.key(doc))
                    selected.append(self.complete_short_passage(doc))
                    break
        return selected

    def summarize_paper(self, source, **settings):
        question = ('Summarize this paper using only the supplied passages. Describe its problem, '
                    'contribution, method, experiments, findings, and limitations only where explicitly '
                    'supported. Omit aspects absent from the evidence; do not infer missing details. '
                    'A concise summary of the supported aspects is a complete answer.')
        return self.ask(question, _summary_source=source, **settings)

    def ask(self, question, api_key=None, provider='Groq', model=None, base_url=None, _summary_source=None):
        from llm_provider import create_client
        if not question.strip() or len(question) > 2000:
            raise ValueError('Enter a question between 1 and 2,000 characters.')
        # A supplied key belongs only to this request, never to the cached service.
        # An explicit blank key must not silently charge the server account.
        if api_key is None and provider != 'Groq':
            raise ValueError('Select My API key and enter credentials for this provider.')
        key = (api_key if api_key is not None else
               os.getenv('GROQ_API_KEY') or dotenv_values(ROOT / '.env').get('GROQ_API_KEY') or '').strip()
        if not key:
            raise ValueError('Enter your API key in LLM settings, or configure a server key.')
        selected_model = model if model is not None else (os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b') if provider == 'Groq' else '')
        llm = create_client(provider, selected_model, key, base_url)
        # Serialize shared model use and API pacing across browser sessions.
        with self.lock:
            started = perf_counter()
            evidence = make_evidence(self.summary_passages(_summary_source) if _summary_source is not None else self.search(question.strip()))
            diagnostics = []
            answer = grounded_answer(llm,question.strip(),evidence,diagnostics)
            status = 'refused' if answer == REFUSAL else 'error' if answer == CHECK_FAILED else 'answered'
            return dict(question=question.strip(),answer=answer,evidence=evidence,
                        diagnostics=diagnostics,status=status,provider=provider,model=selected_model,seconds=perf_counter()-started,
                        retrieval_query='Paper-specific summary' if _summary_source is not None else ' | '.join(self.retrieval_queries(question.strip())),
                        summary_source=_summary_source)
