"""Bounded PDF ingestion and arXiv-only downloads."""
from functools import lru_cache
import json
import xml.etree.ElementTree as ET
import hashlib
import io
import re
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

MAX_BYTES = 25 * 1024 * 1024
STARTER_PAPERS = {
    'Attention Is All You Need': 'https://arxiv.org/pdf/1706.03762',
    'BERT: Pre-training of Deep Bidirectional Transformers': 'https://arxiv.org/pdf/1810.04805',
    'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks': 'https://arxiv.org/pdf/2005.11401',
}



@lru_cache(maxsize=1)
def title_catalog():
    catalog_path = Path(__file__).with_name('paper_titles.json')
    try:
        catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    except (OSError, ValueError):
        catalog = {}
    if not isinstance(catalog, dict):
        catalog = {}
    catalog = {source:title for source,title in catalog.items() if isinstance(title,str) and title.strip()}
    for title, url in STARTER_PAPERS.items():
        catalog.setdefault(url, title)
    return catalog


def paper_title(source, stored_title=None, text=''):
    """Prefer catalogued arXiv titles; retain supplied titles for uploaded papers."""
    catalog = title_catalog()
    canonical = re.sub(r'v\d+(?:\.pdf)?$', '', source)
    canonical = canonical.removesuffix('.pdf').replace('/abs/', '/pdf/')
    if canonical in catalog:
        return catalog[canonical]
    if stored_title and stored_title != source and not stored_title.lower().endswith('.pdf'):
        return stored_title
    if source.startswith('upload:'):
        for title in catalog.values():
            pattern = r'^\s*' + r'\s+'.join(re.escape(word) for word in title.split()) + r'\s*$'
            if re.search(pattern, text[:1500], re.I | re.M):
                return title
    return stored_title or 'Title unavailable'


def fetch_arxiv_title(source):
    """Bounded metadata request; ingestion can continue if arXiv is unavailable."""
    canonical = arxiv_url(source)
    ident = canonical.rsplit('/pdf/', 1)[-1]
    request = urllib.request.Request('https://export.arxiv.org/api/query?id_list=' + ident,
                                    headers={'User-Agent':'CitationResearchAssistant/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read(1024 * 1024)
        entry = ET.fromstring(data).find('{http://www.w3.org/2005/Atom}entry')
        if entry is not None:
            entry_id = entry.findtext('{http://www.w3.org/2005/Atom}id', '')
            if re.sub(r'v\d+$', '', entry_id.rsplit('/abs/', 1)[-1]) != re.sub(r'v\d+$', '', ident):
                return None
            return ' '.join(entry.findtext('{http://www.w3.org/2005/Atom}title', '').split()) or None
    except (OSError, ValueError, ET.ParseError):
        return None
    return None


def arxiv_url(value):
    parsed = urlparse(value.strip())
    if parsed.scheme != 'https' or parsed.netloc not in ('arxiv.org', 'www.arxiv.org', 'export.arxiv.org'):
        raise ValueError('Use an HTTPS arxiv.org abstract or PDF URL.')
    match = re.fullmatch(r'/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)(?:\.pdf)?', parsed.path)
    if not match or parsed.query or parsed.fragment:
        raise ValueError('Enter a direct arXiv paper URL, without query parameters.')
    return 'https://arxiv.org/pdf/' + match[1]


class ArxivRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, arxiv_url(newurl))


def download_arxiv(url):
    url = arxiv_url(url)
    request = urllib.request.Request(url, headers={'User-Agent': 'CitationResearchAssistant/1.0'})
    with urllib.request.build_opener(ArxivRedirect()).open(request, timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('Paper exceeds the 25 MB limit.')
    return data, url


def parse_pdf(data, name, source=None):
    from pypdf import PdfReader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if len(data) > MAX_BYTES or not data.startswith(b'%PDF-'):
        raise ValueError('Upload a PDF smaller than 25 MB.')
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError('Password-protected PDFs are not supported.')
    if len(reader.pages) > 200:
        raise ValueError('Use papers with at most 200 pages.')
    digest = hashlib.sha256(data).hexdigest()
    name = Path(name).name
    title = str((reader.metadata or {}).get('/Title') or name)[:300]
    source = source or f'upload:{digest[:16]}/{name}'
    title = paper_title(source, title, reader.pages[0].extract_text() or '' if reader.pages else '')
    if source.startswith('https://arxiv.org/') and paper_title(source) == 'Title unavailable':
        title = fetch_arxiv_title(source) or title
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    documents, ids = [], []
    section = 'Section unavailable'
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        # Heading is a heuristic, explicitly labelled in the interface.
        for line in text.splitlines():
            if re.fullmatch(r'(?:\d+(?:\.\d+)*\.?\s+)[A-Z][^.!?]{2,90}|Abstract|References', line.strip()):
                section = line.strip()
                break
        for index, chunk in enumerate(splitter.split_text(text)):
            if len(chunk.strip()) < 40:
                continue
            documents.append(Document(page_content=chunk, metadata=dict(source=source, title=title,
                page=page_number, section=section, section_inferred=True, content_hash=digest)))
            ids.append(hashlib.sha256(f'{source}:{digest}:{page_number}:{index}'.encode()).hexdigest())
    if not documents:
        raise ValueError('No readable text found. Scanned PDFs need OCR before upload.')
    return documents, ids
