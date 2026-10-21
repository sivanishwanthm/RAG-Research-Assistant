"""Atomic evaluation checkpoints; retain completed rows and human review edits."""
import csv
import json
from pathlib import Path

ERROR_STATUSES = {'api_error', 'verification_error', 'retrieval_error'}
REVIEW_FIELDS = ('human_refused', 'human_correct', 'human_grounded',
                 'human_citation_accuracy', 'review_notes')


def row_key(row):
    return row['case_id'], row['mode']


def open_run(directory, config, cases):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'results.json'
    if path.exists():
        report = json.loads(path.read_text())
        if report['config'] != config or report['cases'] != cases:
            raise ValueError('Resume settings/cases differ. Start a new run to avoid mixing experiments.')
        return report
    return dict(config=config, cases=cases, rows=[], complete=False)


def save_run(directory, report):
    directory = Path(directory)
    review = directory / 'review.csv'
    edits = {}
    if review.exists():
        with review.open(newline='', encoding='utf-8') as handle:
            edits = {row_key(r): r for r in csv.DictReader(handle)}
    for row in report['rows']:
        previous = edits.get(row_key(row), {})
        for field in REVIEW_FIELDS:
            if previous.get(field, '') != '':
                row[field] = previous[field]
    temp = directory / 'results.json.tmp'
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(directory / 'results.json')
    if report['rows']:
        fields = list(dict.fromkeys(k for r in report['rows'] for k in r))
        temp = directory / 'review.csv.tmp'
        with temp.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report['rows']:
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})
        temp.replace(review)


def update_row(rows, row):
    for i, previous in enumerate(rows):
        if row_key(previous) == row_key(row):
            # Retain old failed attempts for cost/debug accounting.
            row['prior_attempts'] = previous.get('prior_attempts', []) + [
                {k: v for k, v in previous.items() if k != 'prior_attempts'}]
            rows[i] = row
            return
    rows.append(row)


def rate_limited(row):
    return any('RateLimitError' in str(d) or '429' in str(d)
               for d in row.get('diagnostics', []))
