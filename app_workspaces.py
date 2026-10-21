"""Library and evaluation workspaces for the local Streamlit app."""
import json
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from paper_ingestion import STARTER_PAPERS, download_arxiv
from rag_evaluation import evidence_recall, summarize
from app_storage import save_json

ROOT = Path(__file__).resolve().parent


def library_view(get_service):
    st.title('Your paper library')
    st.write('Build a collection you can question, compare, and trace back to the page.')
    upload_tab, arxiv_tab = st.tabs(['Upload PDFs', 'Add from arXiv'])
    with upload_tab:
        files = st.file_uploader('Research papers', type=['pdf'], accept_multiple_files=True,
                                 help='Up to 25 MB and 200 pages per paper. Text PDFs only.')
        ingest_uploads = st.button('Index uploaded papers', disabled=not files, type='primary')
    with arxiv_tab:
        selected = st.multiselect('Starter collection', list(STARTER_PAPERS))
        custom = st.text_input('Or paste an arXiv abstract / PDF URL', placeholder='https://arxiv.org/abs/1706.03762')
        ingest_arxiv = st.button('Index arXiv papers', disabled=not (selected or custom.strip()))
    if ingest_uploads or ingest_arxiv:
        with st.status('Indexing papers…', expanded=True) as progress:
            try:
                if ingest_uploads and len(files) > 5:
                    raise ValueError('Upload at most five papers per batch.')
                service = get_service()
                items = [(f.name, f, None) for f in files] if ingest_uploads else [
                    (url, None, url) for url in dict.fromkeys([STARTER_PAPERS[x] for x in selected] + ([custom.strip()] if custom.strip() else []))]
                failed = 0
                for name, uploaded, source in items:
                    try:
                        if source:
                            data, source = download_arxiv(source)
                            name = source.rsplit('/', 1)[-1] + '.pdf'
                        else:
                            data = uploaded.getvalue()
                        count = service.ingest(data, name, source)
                        st.write(f'{name}: {count} new passages indexed.')
                    except ValueError as error:
                        failed += 1
                        st.warning(f'{name}: {error}')
                    except Exception:
                        failed += 1
                        st.warning(f'{name}: could not index this paper. Check the PDF or connection and retry.')
                st.session_state['paper_count'] = service.paper_count
                st.session_state.pop('result', None)
                st.session_state.pop('paper_summary', None)
                progress.update(label=f'{len(items) - failed} of {len(items)} papers processed',
                                state='error' if failed else 'complete', expanded=True)
            except ValueError as error:
                st.error(str(error))
                progress.update(label='Paper could not be indexed', state='error')
            except Exception:
                st.error('Indexing could not finish. Check PDF readability, model availability, and the arXiv connection. Retry to resume missing chunks.')
                progress.update(label='Indexing interrupted', state='error')
    st.divider()
    if st.session_state.get('library_titles_version') != 2:
        st.session_state.pop('library', None)
        st.session_state['library_titles_version'] = 2
    if st.button('Load / refresh collection') or 'library' not in st.session_state or ingest_uploads or ingest_arxiv:
        try:
            with st.spinner('Reading collection…'):
                st.session_state['library'] = get_service().library()
                st.session_state['paper_count'] = len(st.session_state['library'])
        except Exception:
            st.error('The collection could not load. Check local model availability.')
    papers = st.session_state.get('library', [])
    if papers:
        st.metric('Indexed papers', len(papers))
        search = st.text_input('Find a paper')
        st.caption('Click a title to open and generate its summary using your selected LLM settings and API quota.')
        matching = [p for p in papers if search.casefold() in (p['title'] + p['source']).casefold()]
        for paper in matching:
            with st.container(border=True):
                st.button(paper['title'], key='library_title_' + paper['source'], width='stretch',
                          on_click=open_paper_summary, args=(paper['source'],))
                st.caption(f"{paper['chunks']} indexed passages · {paper['source']}")
        if not matching:
            st.info('No papers match your search.')
    elif 'library' in st.session_state:
        st.info('No papers indexed yet. Upload a text PDF or add an arXiv paper above to get started.')
    st.caption('PDF text is embedded locally and persisted in Chroma. Answer generation sends retrieved passages to the selected LLM provider. Page numbers are PDF pages; section headings are inferred.')


METRIC_GUIDE = {
    'Reference evidence recall@5': ('Evidence found', 'How much of the reference evidence appeared in the top 5 retrieved passages. Exact excerpt matching can miss equivalent wording. Higher is better.'),
    'False refusal rate (lower is better)': ('Missed answer opportunities', 'Questions labelled answerable that were refused. Excludes requests that failed. Lower is better.'),
    'Unsupported answer rate (lower is better)': ('Answers without available evidence', 'Questions labelled unanswerable that received an answer. Excludes failed requests. Lower is better.'),
    'Mean response time (seconds)': ('Average response time', 'Average elapsed time per attempted question, including failed requests.'),
    'Error rate (lower is better)': ('Request failures', 'Attempted questions that failed because of API or processing errors. Lower is better.'),
    'Answer correctness': ('Answer correctness', 'Your review of whether the answer matches the reference answer and the paper.'),
    'Groundedness': ('Supported claims', 'Your review of how much of the answer is supported by the retrieved passages.'),
    'Citation accuracy': ('Citation accuracy', 'Supported claim–citation pairs divided by all cited claim–citation pairs, averaged across reviewed answers.'),
}
REVIEW_FIELDS = [('correct', 'Is the answer correct?', 'Compare the answer with the reference and original paper.'),
                 ('grounded', 'Are the claims supported?', 'Check each factual claim against the passages below. A plausible claim is not enough.'),
                 ('citation_accuracy', 'Do the citations support their claims?', 'Check each claim–citation pair. Score = supported pairs ÷ all cited pairs. For example, 3 of 4 = 75%.')]


def render_metrics(metrics, human=False):
    names = {'Answer correctness', 'Groundedness', 'Citation accuracy'}
    selected = [m for m in metrics if (m['metric'] in names) == human]
    columns = st.columns(3)
    for index, metric in enumerate(selected):
        title, explanation = METRIC_GUIDE[metric['metric']]
        value = metric['value']
        display = 'Not reviewed' if human else 'Not measured'
        if value is not None:
            display = f'{value:.1f} s' if 'seconds' in metric['metric'] else f'{value:.0%}'
        with columns[index % 3]:
            with st.container(border=True):
                st.metric(title, display, help=explanation)
                st.caption(explanation)
                st.caption(f"Based on {metric['evaluated']} questions" if metric['evaluated'] else
                           'Review an answered question below.' if human else 'No eligible labelled questions in this run.')


def evaluation_view(get_service, api_key=None, llm_options=None):
    st.title('How well is your assistant working?')
    st.write('Run a short check, see where it succeeds, and review answers against the papers.')
    section = st.radio('Evaluation workspace', ['View results', 'Run a new check'], horizontal=True, key='evaluation_section')
    root = ROOT / 'evaluation_results'
    if section == 'Run a new check':
        run_evaluation(get_service, root, api_key, llm_options)
        return
    runs = sorted(root.glob('*/results.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        st.info('No evaluations yet. Open “Run a new check” to test four questions against your library.')
        return
    preferred = st.session_state.pop('new_evaluation_path', None)
    if preferred and Path(preferred) in runs:
        st.session_state['evaluation_run'] = Path(preferred)
    if st.session_state.get('evaluation_run') not in runs:
        st.session_state['evaluation_run'] = runs[0]
    selected = st.selectbox('Choose a saved check', runs, key='evaluation_run',
                            format_func=lambda p: p.parent.name,
                            help='Newest saved report first. Results describe the library and model at the time of that run.')
    try:
        report = json.loads(selected.read_text())
        rows = report.get('results', report.get('rows', []))
        if not isinstance(rows, list) or not all(isinstance(r, dict) and 'id' in r for r in rows):
            raise ValueError('Invalid report')
    except (ValueError, OSError, AttributeError):
        st.error('This saved check could not be opened. Choose another report or run a new check.')
        return
    if not rows:
        st.info('This check has no completed questions yet. Run a new check to collect results.')
        return
    review_path = selected.parent / 'reviews.json'
    reviews = {}
    try:
        reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
        if not isinstance(reviews, dict):
            raise ValueError('Invalid reviews')
        metrics = summarize(rows, reviews)
    except (ValueError, KeyError, TypeError, OSError):
        st.warning('This report or its reviews use an unsupported format. Choose a newer check; your files have not been changed.')
        st.download_button('Download original report', selected.read_text(), 'evaluation-report.json')
        return
    answered = [r for r in rows if r.get('status') == 'answered']
    reviewed = sum(all(reviews.get(r['id'], {}).get(field) is not None for field, _, _ in REVIEW_FIELDS) for r in answered)
    errors = sum(r.get('status') == 'error' for r in rows)
    expected = len(report.get('cases', [])) or len(rows)
    st.caption(f"{len(rows)} of {expected} questions attempted · {len(answered)} answered · {sum(r.get('status') == 'refused' for r in rows)} refused · {errors} failed")
    if len(rows) < expected:
        st.warning('This check stopped early. Its scores cover only the attempted questions; they do not represent the full test.')
    if errors:
        st.warning('Some requests failed. Inspect those questions below and check your API key, quota, model access, or connection before retrying.')
    st.subheader('1. Automatic results')
    st.caption('These measure retrieval and response behaviour—not whether every answer is factually correct. Reference labels may be assistant-reviewed.')
    render_metrics(metrics)
    st.subheader('2. Answer quality — your review')
    st.caption(f'{reviewed} of {len(answered)} answered questions fully reviewed. Missing reviews are not counted as zero.')
    if answered:
        st.progress(reviewed / len(answered))
    render_metrics(metrics, human=True)
    st.info('Next step: select an answered question, compare it with the reference and passages, then save your review.' if reviewed < len(answered)
            else 'Reviews are complete for the answered questions. Inspect refusals and failures before drawing conclusions.' if answered
            else 'There are no answers to review. Inspect refusals or errors below, check the required papers, and run a new check.')
    st.subheader('3. Inspect a question')
    status_labels = {'answered':'Answered', 'refused':'Refused', 'error':'Request failed'}
    row = st.selectbox('Question', rows, format_func=lambda r: f"{status_labels.get(r.get('status'), 'Unknown')} · {r.get('question', r['id'])}", key=f'question_{selected.parent.name}')
    expectation = 'An evidence-based answer' if row.get('should_answer') else 'A refusal — evidence is not available'
    st.caption('Expected behaviour: ' + expectation)
    left, right = st.columns(2)
    with left:
        st.markdown('**Assistant’s response**')
        st.write(row.get('answer') or 'No answer was returned.')
    with right:
        st.markdown('**Reference answer**')
        st.write(row.get('reference_answer') or 'No reference answer is available for this question.')
    if row.get('diagnostics'):
        with st.expander('Why was this refused or interrupted?'):
            st.write('These are processing diagnostics, not answer-quality scores.')
            for diagnostic in row['diagnostics']:
                st.text(str(diagnostic))
    st.markdown('**Retrieved passages**')
    if not row.get('evidence'):
        st.caption('No passages were saved for this question.')
    for index, evidence in enumerate(row.get('evidence', []), 1):
        with st.expander(f"{evidence.get('citation', '')} · Passage {index} · {evidence.get('title') or 'Source paper'}"):
            source = evidence.get('source', 'Source unavailable')
            st.text(source)
            parsed = urlparse(source)
            if parsed.scheme in ('https', 'http') and parsed.netloc:
                st.link_button('Open source paper', source)
            st.caption(f"PDF page: {evidence.get('page', 'not recorded')}")
            st.text(evidence.get('passage', 'No passage text'))
    if row.get('status') == 'answered':
        with st.form('review_' + selected.parent.name + '_' + row['id']):
            st.markdown('**Review this answer**')
            st.caption('Leave a score as “Not reviewed” if you have not checked it. These are your assessments, not automatic model scores.')
            values = {}
            for field, title, help_text in REVIEW_FIELDS:
                previous = reviews.get(row['id'], {}).get(field)
                choices = [None, 0., .25, .5, .75, 1.]
                if previous not in choices:
                    choices.append(previous)
                values[field] = st.selectbox(title, choices, index=choices.index(previous), help=help_text,
                    format_func=lambda v: 'Not reviewed' if v is None else f'{v:.0%} — ' +
                    ('None' if v == 0 else 'All' if v == 1 else 'Partly'))
                st.caption(help_text)
            values['notes'] = st.text_area('What should improve? (optional)', value=reviews.get(row['id'], {}).get('notes', ''))
            if st.form_submit_button('Save review', type='primary'):
                try:
                    reviews[row['id']] = {**values, 'reviewer':'human', 'reviewed_at':datetime.now(timezone.utc).isoformat()}
                    save_json(review_path, reviews)
                    save_json(selected.parent / 'metrics.json', summarize(rows, reviews))
                    st.session_state['review_saved'] = True
                    st.rerun()
                except (OSError, ValueError):
                    st.error('The review could not be fully saved. Check folder permissions and retry.')
    if st.session_state.pop('review_saved', False):
        st.success('Review saved. The quality scores above have been updated.')
    with st.expander('Download & technical details'):
        st.caption('A small benchmark is a diagnostic check, not proof of accuracy on future questions.')
        st.dataframe(metrics, hide_index=True, width='stretch')
        st.download_button('Download report with reviews', json.dumps({'report':report,'reviews':reviews,'metrics':metrics}, indent=2),
                           'evaluation-with-reviews.json', mime='application/json')


def run_evaluation(get_service, root, api_key, llm_options):
    st.subheader('Check four questions')
    st.write('This checks three research answers and one question the assistant should refuse.')
    st.info('Before you start: add “Attention Is All You Need” and the BERT paper to Library. Otherwise the research questions may correctly be refused.')
    questions = [('q01', 'What is the main contribution of the Transformer architecture?'),
                 ('q02', 'How does scaled dot-product attention work?'),
                 ('q05', 'What pretraining objectives does BERT use?')]
    for _, question in questions:
        st.write('• ' + question)
    st.write('• What is inside my refrigerator right now? — should be refused')
    st.caption('Uses your selected LLM settings and API quota. Each question can make several verification calls. Saved checks are kept separately; previous results are not overwritten.')
    if st.button('Start four-question check', type='primary', key='start_evaluation'):
        if api_key == '':
            st.warning('Enter your API key under “Connect your AI provider” in the sidebar.')
            return
        rows = []
        try:
            labels = json.loads((ROOT / 'evaluation_reference_labels.json').read_text())
            cases = [dict(id=i, question=q, should_answer=True, **labels[i]) for i, q in questions]
            cases.append(dict(id='h01', question='What is inside my refrigerator right now?', should_answer=False, reviewed=True))
            directory = root / datetime.now(timezone.utc).strftime('app_%Y%m%dT%H%M%S_%fZ')
            directory.mkdir(parents=True)
            with st.status('Checking your assistant…', expanded=True) as progress:
                bar = st.progress(0)
                service = get_service()
                for case in cases:
                    st.write(case['question'])
                    result = service.ask(case['question'], api_key=api_key, **(llm_options or {}))
                    rows.append({**case, **result, 'evidence_recall_at_5':evidence_recall(case, result['evidence'])})
                    save_json(directory / 'results.json', dict(cases=cases, results=rows))
                    save_json(directory / 'metrics.json', summarize(rows, {}))
                    bar.progress(len(rows) / len(cases))
                    if result['status'] == 'error':
                        break
                progress.update(label=f'{len(rows)} of 4 questions checked', state='error' if any(r['status']=='error' for r in rows) else 'complete')
            st.session_state['new_evaluation_path'] = str(directory / 'results.json')
            st.success('Results saved. Open “View results” to see this check and review the answers.')
        except Exception:
            st.error('The check stopped. Check your LLM settings and connection. Any completed questions remain saved; open View results to inspect them.')


def open_paper_summary(source):
    st.session_state['summary_filter'] = ''
    st.session_state['workspace'] = 'Paper Summary'
    select_summary_paper(source)


def select_summary_paper(source=None):
    if source is not None:
        st.session_state['summary_paper'] = source
    st.session_state['summary_requested'] = True


def paper_summary_view(get_service, api_key=None, llm_options=None):
    st.title('Explore and summarize papers')
    st.write('Choose an indexed paper to get a concise, evidence-based overview with citations.')
    try:
        with st.spinner('Loading available papers…'):
            service = get_service()
            papers = service.library()
    except Exception:
        st.error('The paper collection could not load. Check local model availability and retry.')
        return
    st.caption(f'{len(papers)} papers available in your library')
    if not papers:
        st.info('Your library is empty. Open Library to upload a PDF or add an arXiv paper.')
        return
    search = st.text_input('Find a paper by title or source', key='summary_filter')
    visible = [p for p in papers if search.casefold() in (p['title'] + ' ' + p['source']).casefold()]
    if not visible:
        st.info('No papers match your search. Try another title or clear the filter.')
        return
    sources = [p['source'] for p in visible]
    titles = {p['source']:p['title'] for p in visible}
    if st.session_state.get('summary_paper') not in sources:
        st.session_state['summary_paper'] = None
        st.session_state.pop('summary_requested', None)
    st.caption('Click a paper title to generate its summary. This uses your selected model and API quota.')
    with st.container(height=320 if len(visible) > 5 else 'content', border=True):
        for paper in visible:
            st.button(paper['title'], key='summarize_title_' + paper['source'], width='stretch',
                      on_click=select_summary_paper, args=(paper['source'],),
                      help=f"{paper['source']} · {paper['chunks']} indexed passages")
    source = st.selectbox('Paper to summarize', sources, index=None,
                          placeholder='Choose a paper, or click its title above',
                          format_func=lambda value: titles[value], key='summary_paper',
                          on_change=select_summary_paper)
    if source is None:
        st.info('Select a paper to begin. No model requests are made until you choose one.')
        return
    parsed = urlparse(source)
    if parsed.scheme in ('https', 'http') and parsed.netloc:
        st.link_button('Open original paper', source)
    else:
        st.caption(source)
    st.caption('Uses the LLM settings in the sidebar. Covers the problem, method, experiments, findings, and limitations where supported by selected passages; this is not an exhaustive full-text review.')
    requested = st.session_state.pop('summary_requested', False)
    retry = st.button('Generate paper summary again', type='primary', key='generate_summary')
    if requested or retry:
        st.session_state.pop('paper_summary', None)
        if api_key == '':
            st.warning('Enter your API key under “Connect your AI provider” in the sidebar.')
        else:
            try:
                with st.spinner('Finding key passages and verifying the summary…'):
                    result = service.summarize_paper(source, api_key=api_key, **(llm_options or {}))
                    st.session_state['paper_summary'] = {'source':source, 'result':result}
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error('The summary could not finish. Check model access and your connection, then retry.')
    saved = st.session_state.get('paper_summary')
    if not saved or saved['source'] != source:
        return
    result = saved['result']
    st.subheader('Paper summary')
    st.caption('Generated with ' + result.get('provider', 'provider not recorded') + ' · ' + result.get('model', 'model not recorded'))
    if result['status'] == 'answered':
        st.markdown(result['answer'])
        st.caption('Checked against retrieved passages. Inspect the evidence before relying on a claim.')
    elif result['status'] == 'refused':
        diagnostics = ' '.join(result.get('diagnostics', []))
        if 'Answerability check' in diagnostics:
            st.info('The model judged the selected passages insufficient to summarize this paper. This does not mean the full paper lacks the information.')
        elif 'audit.unsupported_claim' in diagnostics:
            st.info('The draft contained claims the verifier could not support, even after revision. No unverified summary is shown.')
        else:
            st.info('The summary did not pass the evidence checks. See the verification details below for the stage that failed.')
        st.caption('Try Generate paper summary again after updating the app. If it repeats, share the paper title and verification diagnostic.')
    else:
        st.error('The provider could not complete verification. Check your key, model access, quota, or connection.')
    with st.expander('Source passages and verification details'):
        for item in result['evidence']:
            st.markdown(f"**{item['citation']} · PDF page {item.get('page', 'not recorded')}**")
            st.text(item['passage'])
        for diagnostic in result.get('diagnostics', []):
            st.text(diagnostic)
    st.download_button('Download summary and evidence', json.dumps(result, indent=2),
                       file_name='paper-summary.json', mime='application/json')
