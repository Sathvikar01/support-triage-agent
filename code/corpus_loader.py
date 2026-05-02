import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None
    warnings.warn(
        "PyYAML is not installed. Markdown frontmatter (title, source_url, etc.) "
        "will be silently ignored. Install it with: pip install pyyaml",
        stacklevel=2,
    )

from config import DATA_DIR, MAX_CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Document:
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


def _strip_frontmatter(text: str) -> tuple:
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) if yaml else {}
            meta = meta or {}
        except Exception:
            meta = {}
        return text[m.end():], meta
    return text, {}


def _infer_company(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "hackerrank" in parts:
        return "HackerRank"
    if "claude" in parts:
        return "Claude"
    if "visa" in parts:
        return "Visa"
    return "Unknown"


def _split_markdown_sections(text: str) -> List[tuple]:
    sections = []
    pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]
    if matches[0].start() > 0:
        sections.append(("", text[:matches[0].start()]))
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    return sections


def _chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_size:
        return [text.strip()] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        if end < len(text):
            for sep in ['\n\n', '\n', '. ', ' ']:
                idx = text.rfind(sep, start + max_size // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def load_corpus() -> List[Document]:
    documents = []
    for md_file in sorted(DATA_DIR.rglob("*.md")):
        if md_file.name.startswith("index"):
            continue
        raw = md_file.read_text(encoding="utf-8", errors="replace")
        body, frontmatter = _strip_frontmatter(raw)
        company = _infer_company(md_file)
        title = frontmatter.get("title", md_file.stem)
        source_url = frontmatter.get("source_url", frontmatter.get("final_url", ""))
        relative = md_file.relative_to(DATA_DIR)
        breadcrumb = str(relative.parent).replace("\\", "/")

        sections = _split_markdown_sections(body)
        chunk_id = 0
        for heading, section_body in sections:
            parent_prefix = f"# {title}\n" if heading else ""
            if heading:
                parent_prefix = f"# {title}\n## {heading}\n"
            full_text = parent_prefix + section_body
            chunks = _chunk_text(full_text)
            for chunk in chunks:
                if len(chunk) < 20:
                    continue
                documents.append(Document(
                    content=chunk,
                    metadata={
                        "source": str(relative),
                        "company": company,
                        "title": title,
                        "section": heading or title,
                        "url": source_url,
                        "breadcrumb": breadcrumb,
                        "chunk_id": chunk_id,
                    },
                ))
                chunk_id += 1
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} chunks from {DATA_DIR}")
    for d in docs[:3]:
        print(f"  [{d.metadata['company']}] {d.metadata['title']} / {d.metadata['section']}: {d.content[:80]}...")
