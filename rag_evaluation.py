"""Small, single-pipeline RAG evaluation with explicit reference and review scores."""
from time import perf_counter


def make_evidence(documents):
    evidence, labels, seen = [], {}, set()
    for doc in documents:
        source = str(doc.metadata.get('source') or 'Source unavailable')
        passage = doc.page_content.strip()
        if not passage or (source, passage) in seen:
            continue
        seen.add((source, passage))
        labels.setdefault(source, f'[{len(labels) + 1}]')
        item = dict(citation=labels[source], source=source, passage=passage)
        for field in ('title', 'page', 'section', 'section_inferred', 'rerank_score', 'fusion_score'):
            if field in doc.metadata:
                item[field] = doc.metadata[field]
        evidence.append(item)
        if len(evidence) == 5:
            break
    return evidence


def evidence_recall(case, evidence):
    """Exact reference-excerpt coverage, not a measure of answer quality."""
    refs = case.get('expected_evidence', [])
    if not case.get('reviewed') or not case['should_answer'] or not refs:
        return None
    normalize = lambda text: ' '.join(text.split())
    hits = [any(e['source'] == ref['source'] and bool(normalize(ref['quote']))
                and normalize(ref['quote']) in normalize(e['passage'])
                for e in evidence) for ref in refs]
    return sum(hits) / len(hits)


def evaluate(cases, search, answer, refusal_text, failure_text):
    """Use the normal guarded pipeline. No judge-based grading or extra LLM calls."""
    rows = []
    for case in cases:
        started = perf_counter()
        row = dict(id=case['id'], question=case['question'],
                   should_answer=case['should_answer'], reviewed=case.get('reviewed', False),
                   reference_answer=case.get('reference_answer', ''), evidence=[],
                   evidence_recall_at_5=None, answer='', status='error', diagnostics=[],
                   seconds=0.0)
        try:
            row['evidence'] = make_evidence(search(case['question']))
            row['evidence_recall_at_5'] = evidence_recall(case, row['evidence'])
            row['answer'] = answer(case['question'], row['evidence'], row['diagnostics'])
            row['status'] = ('refused' if row['answer'] == refusal_text else
                             'error' if row['answer'] == failure_text else 'answered')
        except Exception as error:
            row['diagnostics'].append(type(error).__name__)
        row['seconds'] = perf_counter() - started
        rows.append(row)
        if any('RateLimitError' in str(d) for d in row['diagnostics']):
            break  # Do not spend more requests after exhausted retries.
    return rows


def summarize(rows, reviews):
    """Return metrics with denominators; missing reviews never count as zero."""
    metrics = []
    def add(name, values):
        values = [v for v in values if v is not None]
        metrics.append(dict(metric=name, value=sum(values)/len(values) if values else None,
                            evaluated=len(values)))
    add('Reference evidence recall@5', [r['evidence_recall_at_5'] for r in rows])
    for field, title in [('correct', 'Answer correctness'), ('grounded', 'Groundedness'),
                         ('citation_accuracy', 'Citation accuracy')]:
        scores = []
        for row in rows:
            if row['status'] != 'answered':
                continue
            value = reviews.get(row['id'], {}).get(field)
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError(f'{field} for {row["id"]} must be 0–1 or None')
            scores.append(value)
        add(title, scores)
    supported = [r for r in rows if r['reviewed'] and r['should_answer'] and r['status'] != 'error']
    unsupported = [r for r in rows if r['reviewed'] and not r['should_answer'] and r['status'] != 'error']
    add('False refusal rate (lower is better)', [int(r['status']=='refused') for r in supported])
    add('Unsupported answer rate (lower is better)', [int(r['status']=='answered') for r in unsupported])
    add('Mean response time (seconds)', [r['seconds'] for r in rows])
    add('Error rate (lower is better)', [int(r['status']=='error') for r in rows])
    return metrics
