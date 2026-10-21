# Citation-Grounded AI Research Assistant · Paper Desk

A local Streamlit research workspace with PDF/arXiv ingestion, Chroma storage, semantic + BM25 retrieval, cross-encoder reranking, cited answers, and paper summaries. No notebook execution is needed.

## Run locally

```sh
conda activate myenv
python -m pip install -r requirements-app.txt
python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```

Or run `./start_app.sh` after activating the environment. Open http://localhost:8501. Keep the terminal running. Restart the server after Python module updates if hot reload retains old code; a restart clears session history and personal API keys.

## Connect your model

In **Connect your AI provider**, select Groq, OpenAI, Anthropic, Google Gemini, or OpenAI-compatible, then enter a model ID and your API key. Groq can use the configured default model when the model field is blank. **Test connection** makes one small, potentially billable request and checks that the model can return JSON. It does not certify answer quality.

The UI requires the user's own key and never falls back to a server key. The key is masked and held in session memory, passed through the app server to the selected provider, and excluded from reports and shared model caches. **Clear my key** removes the field value; it does not revoke the provider key or cancel a request already in flight. Changing provider clears both the key and model. Changing a custom endpoint clears the key. Notebook/script callers retain their separate server-key support.

Custom APIs must implement bearer-token authenticated Chat Completions at the supplied API base URL. The operator must set `RAG_ALLOWED_LLM_HOSTS` to allowed hostnames (comma-separated). HTTPS on port 443 is required; redirects are not followed. Proprietary APIs such as Bedrock and Azure deployment-specific endpoints need additional adapters. Every selected model must return the requested JSON for verification.

## Research

Ask a focused question about the indexed collection. Questions are independent; this is not conversational follow-up retrieval. Suggested questions fill the question box without sending a model request. Answers show inline citations, cited papers, source passages, and downloadable JSON. Technical details appear under **How this answer was produced**.

The app combines semantic and keyword retrieval, rank fusion, and cross-encoder reranking. It checks answerability, source references, and claim support; a rejected draft may be revised once and audited again. Unverified answers are withheld. Scores indicate relevance, not calibrated confidence. Checks reduce unsupported answers but do not guarantee correctness.

## Library

Library automatically lists indexed papers by title. Upload up to five text PDFs per batch (25 MB and 200 pages per file), or add HTTPS arXiv PDF/abstract URLs. Scanned PDFs need OCR first. Each paper is processed independently: one failed download or PDF does not block the other papers in the batch. Retry failed papers to resume missing chunks; deterministic IDs prevent inserting identical chunks again.

Extraction, embeddings, and retrieval run locally. Chroma persists under `chroma_db`, or the directory selected by `RAG_DATABASE`. An empty database can be populated through Library. The collection uses `sentence-transformers/all-MiniLM-L6-v2`; first use downloads embedding and reranking models unless cached. Do not write to the same Chroma directory from a second process while the app is using it.

`paper_titles.json` contains official arXiv titles for the original collection. Ingestion retrieves metadata for new arXiv papers when needed. Uploads use PDF title metadata or a recognized title-page heading, falling back to the filename. Pages are PDF page numbers; section names are inferred. Older notebook chunks may lack pages. Reingesting a notebook paper with the new page chunker does not remove its old chunks.

## Paper Summary

Click a title in Library or Paper Summary to generate a cited overview using your selected model and API quota. Search by title or source. The selector keeps substantive opening text and retrieves complementary method, results, and limitation passages from that paper only. This is a summary of selected evidence, not an exhaustive full-text review. Missing topics are omitted rather than invented.

Switching paper titles generates a new summary; ordinary page reruns do not repeat requests. **Generate paper summary again** retries explicitly. Summaries show the model used, source evidence, and diagnostics when verification fails. JSON downloads exclude API keys. Evaluation is no longer in the navigation; older evaluation files remain available on disk.

## Troubleshooting

- **Missing key:** enter your own key in the sidebar. No server account is silently charged.
- **Rejected key or model access:** check provider, model ID, endpoint, and permissions using Test connection.
- **Rate limit:** wait for your provider's quota to recover before retrying.
- **Timeout/connection failure:** check connectivity and provider availability.
- **Refused answer/summary:** inspect the retrieved passages and verification diagnostics. Relevant information may exist in the full paper but be absent from the selected passages.
- **Library failed to load:** check database permissions and model availability.
- **Redacted exception after an update:** inspect the server terminal; restart Streamlit if it retained an old imported module.

## Docker and hosting

```sh
docker compose up --build -d
# Stop:
docker compose down
```

Docker binds localhost:8501, runs as a non-root user, drops Linux capabilities, and rotates logs. Chroma, old evaluation reports, and the model cache are mounted separately. Keys and local data are excluded from the image. Make sure mounted directories are writable by container UID 1000. A server Groq key is optional and is not used by the UI.

Use one replica with local Chroma. For a private hosted deployment, configure HTTPS with WebSocket support and set `RAG_REQUIRE_AUTH=true` plus `RAG_ALLOWED_EMAILS`. Add OIDC settings in `.streamlit/secrets.toml` (excluded from Git/image):

```toml
[auth]
redirect_uri = "https://YOUR-HOST/oauth2callback"
cookie_secret = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
client_id = "YOUR_PROVIDER_CLIENT_ID"
client_secret = "YOUR_PROVIDER_CLIENT_SECRET"
server_metadata_url = "YOUR_PROVIDER_OIDC_DISCOVERY_URL"
```

Mount this file read-only at `/app/.streamlit/secrets.toml` in Docker. The identity provider must supply a verified email claim. All permitted users share the paper library; there is no tenant isolation. Follow the [Streamlit authentication guide](https://docs.streamlit.io/develop/concepts/connections/authentication).

## Validation and limits

```sh
python -m unittest discover -s tests -q
```

Tests cover UI actions, source isolation, guardrails, provider request formats, key handling, ingestion, and legacy evaluation helpers. Provider adapters use mocked HTTP responses unless a live test is explicitly run. Browser visual QA, Docker builds, real OIDC/HTTPS, and load testing require the target environment. Public untrusted uploads also need isolated parsing workers and per-user quotas. No public deployment is implied by these files.
