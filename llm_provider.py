"""Request-scoped text adapters. No credentials are persisted or cached."""
import os
from types import SimpleNamespace
from urllib.parse import urlsplit
import httpx

PROVIDERS = ('Groq', 'OpenAI', 'Anthropic', 'Google Gemini', 'OpenAI-compatible')
ENDPOINTS = {'OpenAI': 'https://api.openai.com/v1',
             'Anthropic': 'https://api.anthropic.com/v1',
             'Google Gemini': 'https://generativelanguage.googleapis.com/v1beta/openai'}


class ProviderError(Exception):
    def __init__(self, status_code, headers):
        super().__init__(f'LLM provider returned HTTP {status_code}')
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers)


def validate_endpoint(base_url):
    url = urlsplit(base_url.strip())
    allowed = {urlsplit(v).hostname for v in ENDPOINTS.values()}
    allowed.update(x.strip().lower() for x in os.getenv('RAG_ALLOWED_LLM_HOSTS', '').split(',') if x.strip())
    if (url.scheme != 'https' or url.hostname not in allowed or url.username or url.password
            or url.query or url.fragment or url.port not in (None, 443)):
        raise ValueError('Use an HTTPS API base URL on a host configured in RAG_ALLOWED_LLM_HOSTS.')
    return base_url.strip().rstrip('/')


class TextClient:
    def __init__(self, provider, model, api_key, base_url):
        self.provider, self.model = provider, model
        self._key, self._base = api_key, base_url

    def invoke(self, messages):
        converted = [{'role': 'user' if role == 'human' else role, 'content': text} for role, text in messages]
        payload = {'model': self.model, 'messages': converted}
        headers = {'Authorization': 'Bearer ' + self._key}
        endpoint = self._base + '/chat/completions'
        if self.provider == 'Anthropic':
            endpoint = self._base + '/messages'
            headers = {'x-api-key': self._key, 'anthropic-version': '2023-06-01'}
            payload['system'] = '\n'.join(m['content'] for m in converted if m['role'] == 'system')
            payload['messages'] = [m for m in converted if m['role'] != 'system']
            payload['max_tokens'] = 4096
        # Do not follow redirects with credentials; errors never include provider bodies.
        with httpx.Client(timeout=45, follow_redirects=False) as client:
            response = client.post(endpoint, headers=headers, json=payload)
        if response.status_code >= 300:
            raise ProviderError(response.status_code, response.headers)
        data = response.json()
        if self.provider == 'Anthropic':
            content = ''.join(block['text'] for block in data['content'] if block.get('type') == 'text')
        else:
            content = data['choices'][0]['message']['content']
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Provider returned no text response.')
        return SimpleNamespace(content=content)


def create_client(provider, model, api_key, base_url=None):
    if provider not in PROVIDERS:
        raise ValueError('Select a supported LLM provider.')
    if not api_key or not api_key.strip():
        raise ValueError('Enter the selected provider’s API key in LLM settings.')
    if not model or not model.strip():
        raise ValueError('Enter a model ID available in your provider account.')
    if provider == 'Groq':
        from langchain_groq import ChatGroq
        return ChatGroq(model=model.strip(), groq_api_key=api_key.strip(), temperature=0, max_retries=0, timeout=45)
    endpoint = validate_endpoint(base_url or '') if provider == 'OpenAI-compatible' else ENDPOINTS[provider]
    return TextClient(provider, model.strip(), api_key.strip(), endpoint)
