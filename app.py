"""Paper Desk: python -m streamlit run app.py"""
import json
import re
from urllib.parse import urlparse
import streamlit as st
from app_backend import ResearchService

st.set_page_config(page_title='Citation-Grounded AI Research Assistant', page_icon='📚', layout='wide')
st.markdown('''<style>
.stApp {background:#f6f8fb;color:#14283e;}
.block-container {max-width:1120px;padding-top:2.7rem;padding-bottom:3rem;}
section[data-testid="stSidebar"] {background:#112b3a;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1.25rem 1rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.65rem;}
section[data-testid="stSidebar"] hr {margin:.5rem 0;}
section[data-testid="stSidebar"] .st-key-clear button {
    background:#b9f0e2 !important;border-color:#b9f0e2 !important;
}
section[data-testid="stSidebar"] label {color:#edf4f6;}
/* Keep the settings expander readable in its resting, hover, and focus states. */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background:#1c3c4c !important;color:#f1f6f8 !important;border-radius:8px;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color:#f1f6f8 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    color:#f1f6f8 !important;fill:currentColor;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible {
    outline:2px solid #67c4b7;outline-offset:2px;
}
/* Select controls use a light surface with explicitly dark text and arrows. */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#f1f6f8 !important;color:#14283e !important;border-color:#c7d7df !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] p,
section[data-testid="stSidebar"] [data-baseweb="select"] input {
    color:#14283e !important;-webkit-text-fill-color:#14283e;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color:#14283e !important;fill:currentColor;
}
/* Dropdown menus render outside the sidebar in a portal. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"] {
    background:#f1f6f8;color:#14283e;
}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover {
    background:#dcefeb;color:#14283e;
}


section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] small {color:#edf4f6;}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background:#f1f6f8;border:1px solid #c7d7df;color:#14283e;
    justify-content:flex-start;text-align:left;padding:9px 12px;
    border-radius:10px;min-height:42px;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p {
    color:#14283e;font-size:14px;font-weight:500;line-height:1.5;
    white-space:normal;overflow-wrap:anywhere;text-align:left;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background:#dcefeb;border-color:#67c4b7;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:focus-visible {
    outline:3px solid #67c4b7;outline-offset:3px;
}
h1 {font-size:3.4rem !important;letter-spacing:-.055em;line-height:1.1;}
h2,h3 {letter-spacing:-.025em;}
div[data-testid="stForm"] {background:white;border:1px solid #dbe4ec;border-radius:18px;padding:1.4rem;box-shadow:0 10px 30px #16334a08;}
.hero-label {font:600 12px sans-serif;letter-spacing:.15em;color:#087f79;margin-bottom:16px;}
.intro {font-size:18px;color:#536578;max-width:650px;line-height:1.65;margin-bottom:26px;}
.feature {background:#fff;border:1px solid #e1e8ef;border-radius:16px;padding:24px;min-height:174px;}
.feature .number {color:#087f79;font-size:12px;font-weight:700;letter-spacing:.12em;}
.feature h3 {font-size:20px;margin-top:14px;}
.feature p {color:#607083;font-size:14px;line-height:1.6;}
@media(max-width:640px){h1{font-size:2.5rem !important;}.block-container{padding-top:1rem;}}

.st-key-research_header h1 {font-size:clamp(2rem, 3.5vw, 2.8rem) !important;line-height:1.15;letter-spacing:-.035em;}
.st-key-research_header .intro {max-width:none;font-size:17px;margin-bottom:8px;}
.st-key-research_header .hero-label {margin-bottom:8px;}
.st-key-research_provider_status {overflow-wrap:anywhere;}
.st-key-research_library_card, .st-key-research_summary_card {
    min-height:190px;background:#fff;border-radius:14px;
}
.st-key-research_library_card > div, .st-key-research_summary_card > div {gap:.65rem;}
.st-key-research_library_card [data-testid="stButton"],
.st-key-research_summary_card [data-testid="stButton"] {margin-top:auto;}
div[class*="st-key-research_prompt_"] button {
    min-height:96px;text-align:left;justify-content:flex-start;padding:12px 16px;
}
@media(max-width:640px){
    .st-key-research_library_card,.st-key-research_summary_card {min-height:0;}
    div[class*="st-key-research_prompt_"] button {min-height:64px;}
}
</style>''', unsafe_allow_html=True)

from app_access import require_access
require_access()

@st.cache_resource(show_spinner=False)
def get_service(retrieval_version='summary-coverage-v11'):
    return ResearchService()


def source_link(source):
    parsed = urlparse(source)
    if parsed.scheme in ('https', 'http') and parsed.netloc:
        st.link_button('Open paper ↗', source)
    else:
        st.caption(source)

def new_question():
    st.session_state['workspace'] = 'Research'
    st.session_state['question'] = ''
    st.session_state.pop('result', None)


def choose_question(question, result=None):
    st.session_state['workspace'] = 'Research'
    st.session_state['question'] = question
    st.session_state.pop('result', None)
    if result is not None:
        st.session_state['result'] = result


def clear_personal_key():
    st.session_state['personal_api_key'] = ''


def change_provider():
    clear_personal_key()
    st.session_state['llm_model'] = ''


if st.session_state.get('workspace') == 'Evaluation':
    st.session_state['workspace'] = 'Paper Summary'
st.session_state.setdefault('history', [])
with st.sidebar:
    st.markdown('### 📚 Paper Desk')
    st.caption('RESEARCH WITH EVIDENCE')
    st.button('＋ New question', key='clear', width='stretch', on_click=new_question,
              help='Start a blank question. Your recent answers remain available.')
    workspace = st.radio('Workspace', ['Research', 'Library', 'Paper Summary'], key='workspace')
    if 'paper_count' in st.session_state:
        st.caption(f"{st.session_state['paper_count']} papers in your collection")
    with st.expander('Connect your AI provider', expanded=True):
        from llm_provider import PROVIDERS
        st.caption('Choose your provider and enter your own API key to ask questions or summarize papers.')
        st.selectbox('Provider', PROVIDERS, key='llm_provider',
                     on_change=change_provider)
        st.text_input('Model ID', key='llm_model',
                      placeholder='Exact model ID from your provider',
                      help='For Groq, leave blank to use the default model.')
        if st.session_state['llm_provider'] == 'OpenAI-compatible':
            st.text_input('API base URL', key='llm_base_url', placeholder='https://your-provider.example/v1',
                          on_change=clear_personal_key)
            st.caption('The server owner must allow custom hosts using RAG_ALLOWED_LLM_HOSTS. Chat Completions API compatibility is required.')
        st.text_input('Your API key', type='password', key='personal_api_key',
                      help='Never written to project files. Changing provider or endpoint clears it.')
        st.button('Clear my key', on_click=clear_personal_key, key='clear_api_key')
        if st.button('Test connection', key='test_llm',
                     help='Sends a small test request to the selected model; provider charges may apply.'):
            from llm_provider import create_client
            import os
            try:
                selected_provider = st.session_state['llm_provider']
                selected_model = st.session_state.get('llm_model', '').strip()
                if not selected_model and selected_provider == 'Groq':
                    selected_model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
                client = create_client(selected_provider, selected_model,
                    st.session_state.get('personal_api_key', ''), st.session_state.get('llm_base_url'))
                with st.spinner('Checking model access…'):
                    response = client.invoke([('system', 'Return only the JSON object requested.'),
                                              ('human', 'Return {"ok": true}.')])
                    verified = json.loads(response.content)
                if isinstance(verified, dict) and verified.get('ok') is True:
                    st.success('Connected. The model returned valid JSON.')
                else:
                    st.warning('Connected, but the response did not follow the required JSON format.')
            except ValueError:
                st.error('Check the key, model ID, and custom endpoint configuration. The model must return valid JSON.')
            except Exception as error:
                status = getattr(error, 'status_code', None)
                message = {401: 'API key rejected.', 403: 'This key does not have permission.',
                           404: 'Model or endpoint not found.', 429: 'Rate limit or quota reached.'}.get(status,
                           'Connection failed. Check the provider, model, and network connection.')
                st.error(message)

        st.caption('Your key is sent through this server to the selected provider and held in session memory. Use HTTPS when hosted.')
    st.markdown('**Try a research question**')
    examples = [
        ('Transformer contributions', 'What is the main contribution of the Transformer architecture?'),
        ('BERT pretraining objectives', 'What pretraining objectives does BERT use?'),
        ('How attention works', 'How does scaled dot-product attention work?')]
    for i, (label, question_text) in enumerate(examples):
        st.button(label, key=f'example{i}', width='stretch', help=question_text,
                  on_click=choose_question, args=(question_text,))
    st.markdown('**Recent questions**')
    if st.session_state['history']:
        with st.container(height=260 if len(st.session_state['history']) > 4 else 'content', border=False):
            for i, previous in enumerate(st.session_state['history']):
                st.button(previous['question'], key=f'history{i}', width='stretch',
                          on_click=choose_question, args=(previous['question'], previous))
    else:
        st.caption('Your answers will appear here.')
    with st.expander('How to use this workspace'):
        st.caption('1. Add papers in Library.\n2. Ask a focused question.\n3. Inspect the cited evidence.\n4. Summarize a selected paper.')
        st.caption('Questions and supporting passages are sent to the selected LLM provider. History is limited to ten answers in this browser session.')

# Always pass an explicit session key; a missing key must never fall back to server credentials.
request_api_key = st.session_state.get('personal_api_key', '').strip()
provider = st.session_state.get('llm_provider', 'Groq')
model = st.session_state.get('llm_model', '').strip()
llm_options = dict(provider=provider, model=model or (None if provider == 'Groq' else ''),
                   base_url=st.session_state.get('llm_base_url', '') if provider == 'OpenAI-compatible' else None)

from app_workspaces import library_view, paper_summary_view
if workspace == 'Library':
    library_view(get_service)
    st.stop()
if workspace == 'Paper Summary':
    paper_summary_view(get_service, api_key=request_api_key, llm_options=llm_options)
    st.stop()

def open_workspace(name):
    st.session_state['workspace'] = name


with st.container(key='research_header'):
    st.markdown('<div class="hero-label">YOUR PAPERS. TRACEABLE ANSWERS.</div>', unsafe_allow_html=True)
    st.title('Research your papers')
    st.markdown('<div class="intro">Ask a question, explore the evidence, or read a focused paper summary.</div>', unsafe_allow_html=True)
with st.container(border=True, key='research_provider_status'):
    st.markdown('**Use your preferred AI model**')
    st.write('Connect a model from OpenAI, Anthropic, Google Gemini, Groq, or an OpenAI-compatible service.')
    if request_api_key:
        st.caption('Selected: ' + provider + ' · ' + (model or ('Default model' if provider == 'Groq' else 'Model not selected')))
        st.caption('Key entered · use Test connection to verify access.')
    else:
        st.caption('Add your own API key in the sidebar to generate answers. Choose your provider and model under Connect your AI provider.')
collection, summaries = st.columns(2, gap='medium')
with collection:
    with st.container(border=True, key='research_library_card'):
        st.markdown('**Paper library**')
        count = st.session_state.get('paper_count')
        st.write(f'{count} indexed papers' if count is not None else 'Browse available papers')
        st.caption('Add research papers from PDFs or arXiv.')
        st.button('Open library', key='research_library', on_click=open_workspace, args=('Library',), width='stretch')
with summaries:
    with st.container(border=True, key='research_summary_card'):
        st.markdown('**Paper summaries**')
        st.write('Understand one paper at a time')
        st.caption('Read key ideas and findings with citations.')
        st.button('Browse summaries', key='research_summaries', on_click=open_workspace, args=('Paper Summary',), width='stretch')
st.subheader('Ask your collection')
with st.form('ask_form'):
    question = st.text_area('What would you like to understand?', key='question', height=110,
                            max_chars=2000, placeholder='For example: How does LoRA reduce the cost of fine-tuning?')
    st.caption('Searches all indexed papers. Name a model or method for better results. Your question and relevant passages are sent to your selected provider.')
    submitted = st.form_submit_button('Find a grounded answer →', type='primary', key='submit')

if submitted:
    st.session_state.pop('result', None)
    if not question.strip():
        st.warning('Enter a question to search your papers.')
    elif request_api_key == '':
        st.warning('Enter your API key under “Connect your AI provider” in the sidebar.')
    else:
        try:
            with st.status('Preparing your research answer…', expanded=True) as progress:
                st.write('Loading the paper collection and local search models.')
                service = get_service()
                st.write('Finding relevant passages and checking the answer against the evidence.')
                result = service.ask(question, api_key=request_api_key, **llm_options)
                st.session_state['result'] = result
                st.session_state['paper_count'] = service.paper_count
                if result['status'] != 'error':
                    st.session_state['history'] = [result] + [r for r in st.session_state['history']
                        if r['question'] != result['question']][:9]
                progress.update(label='Research complete' if result['status'] != 'error' else 'Request could not finish',
                                state='complete' if result['status'] != 'error' else 'error', expanded=False)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error('The research service could not finish this request. Check the server connection and model availability, then try again.')

result = st.session_state.get('result')
if result:
    st.divider()
    st.subheader(result['question'])
    st.caption(f"{result['seconds']:.1f} seconds · {len(result['evidence'])} retrieved passages")
    with st.expander('How this answer was produced'):
        st.write('The app combines keyword and semantic search, ranks the passages, and checks the answer against its cited evidence.')
        st.caption('Provider: ' + result.get('provider', 'Not recorded') + ' · Model: ' + result.get('model', 'Not recorded'))
        if result.get('retrieval_query'):
            st.caption('Retrieval query: ' + result['retrieval_query'])
    answer_tab, evidence_tab = st.tabs(['Answer & sources', 'Inspect evidence'])
    with answer_tab:
        if result['status'] == 'error':
            st.error(result['answer'])
            diagnostic = ' '.join(result['diagnostics'])
            if 'RateLimitError' in diagnostic:
                st.warning('The provider is rate-limiting requests. Wait for capacity to reset before retrying.')
            elif 'authentication' in diagnostic:
                st.warning('The selected API key was rejected. Check Connect your AI provider in the sidebar and retry.')
            elif 'APIConnectionError' in diagnostic or 'connection_failed' in diagnostic:
                st.warning('The server could not connect to the LLM provider. Check the internet connection and retry.')
            elif 'provider_configuration' in diagnostic:
                st.warning('Check the model ID, API endpoint, and whether your account has access to that model.')
            if diagnostic:
                st.caption('Verification diagnostic: ' + diagnostic)
        elif result['status'] == 'refused':
            st.info(result['answer'])
            st.write('Inspect the retrieved passages below, or try a narrower question. If the relevant paper is missing, add it in Library.')
            if result.get('diagnostics'):
                st.caption('Evidence check: ' + ' · '.join(result['diagnostics']))
        else:
            body = result['answer'].split('\n\nSources:\n')[0]
            st.success('Evidence check passed · Model-reviewed, not a guarantee of correctness')
            st.markdown(body)
            labels = set(re.findall(r'\[\d+\]', body))
            sources = dict.fromkeys((e['citation'],e['source']) for e in result['evidence'] if e['citation'] in labels)
            st.markdown('#### Cited papers')
            for label, source in sources:
                with st.container(border=True):
                    entry = next(e for e in result['evidence'] if e['citation'] == label)
                    st.write(f"{label} {entry.get('title') or 'Research paper'}")
                    source_link(source)
                    st.caption('Retrieved excerpt — inspect the evidence tab for all passages from this paper.')
                    st.text(entry['passage'][:500] + ('…' if len(entry['passage']) > 500 else ''))
            st.caption('Checked against retrieved evidence. Review the original passages before relying on a claim.')
        st.download_button('Download answer & evidence', json.dumps(result, indent=2),
                           file_name='research-answer.json', mime='application/json')
    with evidence_tab:
        st.caption('These are the retrieved passages, including any that were not used in the answer.')
        for i, item in enumerate(result['evidence'],1):
            with st.expander(f"{item['citation']} · Passage {i}"):
                st.write(item.get('title') or 'Research paper')
                source_link(item['source'])
                st.caption(f"PDF page: {item.get('page', 'unavailable')} · Section (inferred): {item.get('section', 'unavailable')}")
                if 'rerank_score' in item:
                    st.caption(f"Reranker score: {item['rerank_score']:.3f} · Rank fusion: {item['fusion_score']:.5f} · Relevance signals, not confidence percentages")
                st.text(item['passage'])
else:
    st.markdown('**Try a question**')
    suggestions = [
        ('Understand a method', 'What is query key and value in transformers'),
        ('Explore a contribution', 'What is the main contribution of the Transformer architecture?'),
        ('Explain an architecture', 'What is ColBERT and how does it work?')]
    for column, (label, prompt) in zip(st.columns(3), suggestions):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.button(prompt, key='research_prompt_' + label, on_click=choose_question, args=(prompt,), width='stretch')
    st.caption('Each question is independent. To summarize a specific paper, open Paper Summary and click its title.')
    with st.expander('What to expect from an answer'):
        st.write('Answers include numbered citations and source passages you can inspect. If the retrieved evidence is insufficient, the assistant may decline to answer.')
        st.write('Evidence checks reduce unsupported answers but do not guarantee correctness. Review the original sources before relying on a claim.')
        st.caption('The first request may take longer while local search models load. Model requests use your provider’s API quota.')
