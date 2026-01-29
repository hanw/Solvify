#!/usr/bin/env python3
"""
EPUB to Jupyter Notebook Converter

Converts EPUB files into Jupyter notebooks, one notebook per chapter,
suitable for use with solve.it.com.

Inspired by https://github.com/karpathy/reader3 for EPUB parsing.
"""

import argparse
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

# Try to import optional dependencies
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import ebooklib
    from ebooklib import epub
    HAS_EBOOKLIB = True
except ImportError:
    HAS_EBOOKLIB = False


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for better readability.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text.strip():
        return []

    # Pattern to split on sentence-ending punctuation followed by space
    # Handles: . ! ? and also ." !) ?" etc.
    # Avoids splitting on: Mr. Mrs. Dr. etc., numbers like 1.5, abbreviations

    # Common abbreviations that shouldn't end sentences
    abbreviations = r'(?<![Mm]r)(?<![Mm]rs)(?<![Dd]r)(?<![Pp]rof)(?<![Ss]t)(?<![Jj]r)(?<![Ss]r)(?<![Ii]nc)(?<![Ll]td)(?<![Ee]tc)(?<![Vv]s)(?<![Ee]\.g)(?<![Ii]\.e)'

    # Split on sentence endings, but keep the punctuation with the sentence
    # This regex looks for: period/exclamation/question + optional quote/paren + space + capital letter or end
    sentences = []

    # Simple approach: split on common sentence boundaries
    # Match: .!? followed by optional closing punctuation, then whitespace
    pattern = r'([.!?]["\'\)\]]*)\s+'

    parts = re.split(pattern, text)

    current_sentence = ""
    for i, part in enumerate(parts):
        if not part:
            continue
        current_sentence += part
        # If this part is punctuation (ends a sentence), finalize the sentence
        if re.match(r'^[.!?]["\'\)\]]*$', part):
            sentence = current_sentence.strip()
            if sentence:
                sentences.append(sentence)
            current_sentence = ""

    # Don't forget the last part if it doesn't end with sentence punctuation
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    return sentences


@dataclass
class ContentBlock:
    """Represents a single structural block extracted from HTML."""
    block_type: str  # "heading", "paragraph", "list", "blockquote", "table", "code", "hr", "figure"
    html: str
    text: str
    heading_level: int = 0  # 1-6 for headings, 0 otherwise


@dataclass
class Chapter:
    """Represents a chapter extracted from an EPUB file."""
    index: int
    title: str
    content_html: str
    content_text: str
    blocks: list[ContentBlock] = field(default_factory=list)
    href: str = ""


@dataclass
class Book:
    """Represents a parsed EPUB book."""
    title: str
    author: str
    chapters: list[Chapter] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def clean_html(html_content: str) -> str:
    """
    Clean HTML content by removing scripts, styles, and other unwanted elements.

    Args:
        html_content: Raw HTML content

    Returns:
        Cleaned HTML content
    """
    if not HAS_BS4:
        # Fallback: simple regex-based cleaning
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        return html_content

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove dangerous/unwanted elements
    for tag in soup.find_all(['script', 'style', 'iframe', 'video', 'nav', 'form', 'button', 'input']):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
        comment.extract()

    return str(soup)


def html_to_text(html_content: str) -> str:
    """
    Convert HTML content to plain text.

    Args:
        html_content: HTML content

    Returns:
        Plain text content
    """
    if not HAS_BS4:
        # Fallback: simple regex-based conversion
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to Markdown format.

    Args:
        html_content: HTML content

    Returns:
        Markdown formatted content
    """
    if not HAS_BS4:
        return html_to_text(html_content)

    soup = BeautifulSoup(html_content, 'html.parser')

    # Process the HTML and convert to markdown
    markdown_lines = []

    def process_element(element, depth=0):
        """Recursively process HTML elements and convert to markdown."""
        if element.name is None:
            # Text node - preserve whitespace for proper spacing
            text = str(element)
            # Normalize internal whitespace but preserve leading/trailing space indicator
            has_leading = text and text[0].isspace()
            has_trailing = text and text[-1].isspace()
            text = ' '.join(text.split())
            if text:
                if has_leading:
                    text = ' ' + text
                if has_trailing:
                    text = text + ' '
                return text
            elif has_leading or has_trailing:
                return ' '
            return ""

        tag = element.name.lower()

        # Handle different tags
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(tag[1])
            text = element.get_text(strip=True)
            if text:
                return f"\n{'#' * level} {text}\n"
            return ""

        elif tag == 'p':
            text = ''.join(process_element(child) for child in element.children)
            text = ' '.join(text.split())  # Normalize whitespace
            if text:
                # Split paragraph into sentences for better readability
                sentences = split_into_sentences(text)
                if sentences:
                    return '\n' + '\n'.join(sentences) + '\n'
                return f"\n{text}\n"
            return ""

        elif tag == 'br':
            return "\n"

        elif tag in ['strong', 'b']:
            text = element.get_text(strip=True)
            if text:
                return f"**{text}**"
            return ""

        elif tag in ['em', 'i']:
            text = element.get_text(strip=True)
            if text:
                return f"*{text}*"
            return ""

        elif tag == 'a':
            text = element.get_text(strip=True)
            href = element.get('href', '')
            if text and href:
                return f"[{text}]({href})"
            return text or ""

        elif tag == 'blockquote':
            text = element.get_text(strip=True)
            if text:
                lines = text.split('\n')
                return '\n' + '\n'.join(f"> {line}" for line in lines) + '\n'
            return ""

        elif tag in ['ul', 'ol']:
            items = []
            for i, li in enumerate(element.find_all('li', recursive=False)):
                text = li.get_text(strip=True)
                if tag == 'ol':
                    items.append(f"{i+1}. {text}")
                else:
                    items.append(f"- {text}")
            if items:
                return '\n' + '\n'.join(items) + '\n'
            return ""

        elif tag == 'li':
            # Handled by ul/ol
            return ""

        elif tag == 'code':
            text = element.get_text()
            if '\n' in text:
                return f"\n```\n{text}\n```\n"
            return f"`{text}`"

        elif tag == 'pre':
            text = element.get_text()
            return f"\n```\n{text}\n```\n"

        elif tag == 'img':
            alt = element.get('alt', 'image')
            src = element.get('src', '')
            return f"![{alt}]({src})"

        elif tag == 'hr':
            return "\n---\n"

        elif tag in ['div', 'span', 'section', 'article', 'main', 'body', 'html']:
            # Container elements - process children
            result = []
            for child in element.children:
                processed = process_element(child, depth)
                if processed:
                    result.append(processed)
            return ''.join(result)

        elif tag == 'table':
            # Simple table handling
            rows = element.find_all('tr')
            if not rows:
                return ""

            table_md = []
            for i, row in enumerate(rows):
                cells = row.find_all(['th', 'td'])
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                table_md.append('| ' + ' | '.join(cell_texts) + ' |')

                # Add header separator after first row
                if i == 0:
                    table_md.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')

            return '\n' + '\n'.join(table_md) + '\n'

        else:
            # Default: just get text content
            return element.get_text(strip=True)

    # Find body or process whole document
    body = soup.find('body')
    if body:
        result = process_element(body)
    else:
        result = process_element(soup)

    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def extract_blocks(html_content: str) -> list[ContentBlock]:
    """
    Parse HTML into a flat list of typed ContentBlocks.

    Walks the top-level children of <body> (or the root) and emits one
    ContentBlock per structural element (heading, paragraph, list, etc.).

    Args:
        html_content: Cleaned HTML content

    Returns:
        List of ContentBlock objects
    """
    if HAS_BS4:
        soup = BeautifulSoup(html_content, 'html.parser')
        root = soup.find('body') or soup
        return _extract_blocks_from_element(root)
    else:
        return _extract_blocks_regex(html_content)


def _extract_blocks_from_element(root) -> list[ContentBlock]:
    """
    Recursively extract ContentBlocks from a BeautifulSoup element.

    Walks direct children of the given element. Container tags (div, section, etc.)
    are recursed into by passing the element itself — no re-parsing needed.

    After extraction, consecutive bare text nodes (not wrapped in any HTML tag)
    are merged into a single block. This handles cases like concrete poetry or
    indented verse where each line is a raw text node separated by <br/> tags.

    Args:
        root: A BeautifulSoup Tag or BeautifulSoup object

    Returns:
        List of ContentBlock objects
    """
    raw_blocks: list[ContentBlock] = []

    for elem in root.children:
        # Skip whitespace-only text nodes
        if elem.name is None:
            text = str(elem).strip()
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="paragraph",
                    html=str(elem),
                    text=text,
                    heading_level=-1,  # sentinel: bare text node
                ))
            continue

        tag = elem.name.lower()

        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            text = elem.get_text(strip=True)
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="heading",
                    html=str(elem),
                    text=text,
                    heading_level=int(tag[1]),
                ))

        elif tag == 'p':
            text = elem.get_text(strip=True)
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="paragraph",
                    html=str(elem),
                    text=text,
                ))

        elif tag in ('ul', 'ol'):
            text = elem.get_text(strip=True)
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="list",
                    html=str(elem),
                    text=text,
                ))

        elif tag == 'blockquote':
            text = elem.get_text(strip=True)
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="blockquote",
                    html=str(elem),
                    text=text,
                ))

        elif tag == 'table':
            text = elem.get_text(strip=True)
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="table",
                    html=str(elem),
                    text=text,
                ))

        elif tag in ('pre', 'code'):
            text = elem.get_text()
            if text.strip():
                raw_blocks.append(ContentBlock(
                    block_type="code",
                    html=str(elem),
                    text=text,
                ))

        elif tag == 'hr':
            raw_blocks.append(ContentBlock(
                block_type="hr",
                html=str(elem),
                text="",
            ))

        elif tag == 'figure':
            text = elem.get_text(strip=True)
            raw_blocks.append(ContentBlock(
                block_type="figure",
                html=str(elem),
                text=text,
            ))

        elif tag == 'br':
            # <br/> between text nodes — skip, the merge step handles spacing
            continue

        elif tag in ('div', 'section', 'article', 'main', 'span'):
            # Recurse into container elements directly (no re-parsing)
            inner_blocks = _extract_blocks_from_element(elem)
            raw_blocks.extend(inner_blocks)

        else:
            # Fallback: treat as paragraph if it has text
            text = elem.get_text(strip=True)
            if text:
                raw_blocks.append(ContentBlock(
                    block_type="paragraph",
                    html=str(elem),
                    text=text,
                ))

    # --- Merge consecutive inline/bare text nodes ---
    # Bare text nodes use heading_level=-1 as a sentinel.
    # Inline elements (<i>, <b>, <em>, <strong>, <a>) that appear between
    # bare text nodes are also part of the same text flow (e.g. concrete
    # poetry with italicized words). We mark these as -2 and merge them
    # together with -1 nodes into single blocks.
    INLINE_TAGS = {'i', 'b', 'em', 'strong', 'a', 'sup', 'sub', 'small', 'big'}

    # Re-tag inline-element blocks that sit between bare text nodes
    for i, block in enumerate(raw_blocks):
        if block.heading_level == 0 and block.block_type == "paragraph":
            # Check if this block's HTML is an inline tag (not wrapped in <p>)
            html_stripped = block.html.strip()
            if html_stripped.startswith('<'):
                tag_match = re.match(r'^<(\w+)', html_stripped)
                if tag_match and tag_match.group(1).lower() in INLINE_TAGS:
                    block.heading_level = -2  # sentinel: inline element

    blocks: list[ContentBlock] = []
    for block in raw_blocks:
        is_mergeable = block.heading_level in (-1, -2)
        prev_is_mergeable = blocks and blocks[-1].heading_level in (-1, -2)
        if is_mergeable and prev_is_mergeable:
            # Merge into previous block
            prev = blocks[-1]
            blocks[-1] = ContentBlock(
                block_type="paragraph",
                html=prev.html + " " + block.html,
                text=prev.text + " " + block.text,
                heading_level=-1,
            )
        else:
            blocks.append(block)

    # Reset sentinel heading_level to 0
    for block in blocks:
        if block.heading_level < 0:
            block.heading_level = 0

    return blocks


def _extract_blocks_regex(html_content: str) -> list[ContentBlock]:
    """
    Fallback block extraction using regex when BeautifulSoup is unavailable.

    Args:
        html_content: Raw HTML content

    Returns:
        List of ContentBlock objects (headings and paragraphs only)
    """
    blocks: list[ContentBlock] = []

    # Extract headings
    for m in re.finditer(r'<(h[1-6])[^>]*>(.*?)</\1>', html_content, re.DOTALL | re.IGNORECASE):
        tag, inner = m.group(1), m.group(2)
        text = re.sub(r'<[^>]+>', '', inner).strip()
        if text:
            blocks.append(ContentBlock(
                block_type="heading",
                html=m.group(0),
                text=text,
                heading_level=int(tag[1]),
            ))

    # Extract paragraphs
    for m in re.finditer(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL | re.IGNORECASE):
        text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        if text:
            blocks.append(ContentBlock(
                block_type="paragraph",
                html=m.group(0),
                text=text,
            ))

    return blocks


def compute_cell_boundaries(blocks: list[ContentBlock],
                            target_cell_words: int = 800,
                            max_cell_words: int = 1500) -> list[int]:
    """
    Compute a bitmask indicating where to split blocks into notebook cells.

    Given N blocks, returns an N-length list of 0s and 1s.
    boundary[i] == 1 means "start a new cell at block i".
    boundary[0] is always 1.

    Scoring approach (per position i, for i >= 1):
      1. Structural signals — headings and <hr> always start a new cell.
      2. Adhesive rules — some pairs must NOT be split (heading+content,
         paragraph+list, paragraph+blockquote).
      3. Lexical shift — Jaccard distance between adjacent block word sets.
         Low overlap suggests a topic change.
      4. Size pressure — accumulated word count since last boundary pushes
         toward splitting at paragraph boundaries.

    The score is thresholded at 0.5 to produce the binary mask.

    Args:
        blocks: List of ContentBlock objects
        target_cell_words: Ideal word count per cell before applying pressure
        max_cell_words: Hard maximum; always split when exceeded

    Returns:
        List of ints (0 or 1), same length as blocks
    """
    n = len(blocks)
    if n == 0:
        return []
    if n == 1:
        return [1]

    boundary = [0] * n
    boundary[0] = 1

    words_since_boundary = len(blocks[0].text.split())

    for i in range(1, n):
        curr = blocks[i]
        prev = blocks[i - 1]
        score = 0.0

        # --- Structural signals (definitive) ---
        if curr.block_type == "heading":
            boundary[i] = 1
            words_since_boundary = len(curr.text.split())
            continue

        if prev.block_type == "hr":
            boundary[i] = 1
            words_since_boundary = len(curr.text.split())
            continue

        # --- Adhesive rules (force no-split) ---
        # Keep heading with its following content
        if prev.block_type == "heading":
            words_since_boundary += len(curr.text.split())
            continue  # boundary[i] stays 0

        # Keep paragraph + immediately following list/blockquote together
        if prev.block_type == "paragraph" and curr.block_type in ("list", "blockquote"):
            words_since_boundary += len(curr.text.split())
            continue

        # --- Lexical shift (Jaccard) ---
        # Only apply lexical scoring when both blocks have enough words
        # to produce a meaningful signal. Short dialogue lines and
        # one-liners produce noisy Jaccard values.
        MIN_WORDS_FOR_JACCARD = 15
        words_prev = set(prev.text.lower().split())
        words_curr = set(curr.text.lower().split())
        if (len(words_prev) >= MIN_WORDS_FOR_JACCARD and
                len(words_curr) >= MIN_WORDS_FOR_JACCARD and
                words_prev and words_curr):
            jaccard = len(words_prev & words_curr) / len(words_prev | words_curr)
            if jaccard < 0.1:
                score += 0.6
            elif jaccard < 0.2:
                score += 0.3

        # --- Size pressure ---
        curr_words = len(curr.text.split())
        if words_since_boundary >= max_cell_words:
            # Hard limit: must split
            score = 1.0
        elif words_since_boundary >= target_cell_words:
            # Soft pressure: linearly increase score from 0 to 0.5
            pressure = (words_since_boundary - target_cell_words) / (max_cell_words - target_cell_words)
            score += min(pressure * 0.5, 0.5)

        # --- Threshold ---
        if score >= 0.5:
            boundary[i] = 1
            words_since_boundary = curr_words
        else:
            words_since_boundary += curr_words

    return boundary


def group_blocks_by_boundary(blocks: list[ContentBlock],
                             boundary: list[int]) -> list[list[ContentBlock]]:
    """
    Group blocks into cell groups based on the boundary bitmask.

    Args:
        blocks: List of ContentBlock objects
        boundary: Bitmask from compute_cell_boundaries()

    Returns:
        List of groups, where each group is a list of ContentBlocks
        that belong in the same notebook cell.
    """
    if not blocks:
        return []

    groups: list[list[ContentBlock]] = []
    current_group: list[ContentBlock] = []

    for i, block in enumerate(blocks):
        if boundary[i] == 1 and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append(block)

    if current_group:
        groups.append(current_group)

    return groups


def block_to_markdown(block: ContentBlock) -> str:
    """
    Convert a single ContentBlock to Markdown.

    Args:
        block: ContentBlock to convert

    Returns:
        Markdown string for this block
    """
    if block.block_type == "heading":
        return f"{'#' * block.heading_level} {block.text}"

    if block.block_type == "hr":
        return "---"

    if block.block_type == "code":
        return f"```\n{block.text}\n```"

    # Wrap fragment in a <body> so html_to_markdown processes it correctly
    wrapped = f"<body>{block.html}</body>"
    md = html_to_markdown(wrapped)
    return md.strip()


def blocks_to_cell_source(blocks: list[ContentBlock],
                          output_format: str = "markdown") -> list[str]:
    """
    Convert a group of ContentBlocks into notebook cell source lines.

    Args:
        blocks: List of blocks belonging to one cell
        output_format: "markdown" or "text"

    Returns:
        List of source lines for a notebook cell
    """
    if output_format == "text":
        parts = [b.text for b in blocks if b.text]
        content = "\n\n".join(parts)
    else:
        parts = [block_to_markdown(b) for b in blocks]
        content = "\n\n".join(parts)

    # Clean up excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    lines = []
    for line in content.split('\n'):
        lines.append(line + '\n')

    # Trim trailing empty newline
    if lines and lines[-1] == '\n':
        lines = lines[:-1]
    elif lines:
        lines[-1] = lines[-1].rstrip('\n')

    return lines


def parse_epub_with_ebooklib(epub_path: str) -> Book:
    """
    Parse an EPUB file using ebooklib.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        Book object with parsed content
    """
    book = epub.read_epub(epub_path)

    # Extract metadata
    title = book.get_metadata('DC', 'title')
    title = title[0][0] if title else Path(epub_path).stem

    author = book.get_metadata('DC', 'creator')
    author = author[0][0] if author else "Unknown Author"

    parsed_book = Book(
        title=title,
        author=author,
        metadata={
            'language': book.get_metadata('DC', 'language'),
            'identifier': book.get_metadata('DC', 'identifier'),
            'publisher': book.get_metadata('DC', 'publisher'),
        }
    )

    # Extract chapters from spine
    chapter_index = 0
    for spine_item in book.spine:
        item_id = spine_item[0]
        item = book.get_item_with_id(item_id)

        if item is None:
            continue

        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            raw_content = item.get_content().decode('utf-8', errors='ignore')
            cleaned_html = clean_html(raw_content)
            plain_text = html_to_text(cleaned_html)

            # Skip empty chapters
            if not plain_text.strip():
                continue

            # Try to extract chapter title from content
            chapter_title = extract_chapter_title(cleaned_html, chapter_index)

            # Extract structural blocks
            blocks = extract_blocks(cleaned_html)

            chapter = Chapter(
                index=chapter_index,
                title=chapter_title,
                content_html=cleaned_html,
                content_text=plain_text,
                blocks=blocks,
                href=item.get_name()
            )

            parsed_book.chapters.append(chapter)
            chapter_index += 1

    return parsed_book


def parse_epub_manual(epub_path: str) -> Book:
    """
    Parse an EPUB file manually without ebooklib.

    EPUB files are ZIP archives with a specific structure:
    - META-INF/container.xml points to the OPF file
    - The OPF file contains metadata and the spine (reading order)
    - Content files are typically XHTML

    Args:
        epub_path: Path to the EPUB file

    Returns:
        Book object with parsed content
    """
    with zipfile.ZipFile(epub_path, 'r') as zf:
        # Read container.xml to find the OPF file
        container_xml = zf.read('META-INF/container.xml').decode('utf-8')
        container_root = ET.fromstring(container_xml)

        # Find the rootfile element
        ns = {'container': 'urn:oasis:names:tc:opendocument:xmlns:container'}
        rootfile = container_root.find('.//container:rootfile', ns)

        if rootfile is None:
            # Try without namespace
            rootfile = container_root.find('.//{*}rootfile')

        if rootfile is None:
            raise ValueError("Could not find rootfile in container.xml")

        opf_path = rootfile.get('full-path')
        opf_dir = os.path.dirname(opf_path)

        # Read and parse the OPF file
        opf_content = zf.read(opf_path).decode('utf-8')
        opf_root = ET.fromstring(opf_content)

        # Define OPF namespaces
        opf_ns = {
            'opf': 'http://www.idpf.org/2007/opf',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }

        # Extract metadata
        metadata = opf_root.find('.//{http://www.idpf.org/2007/opf}metadata')
        if metadata is None:
            metadata = opf_root.find('.//{*}metadata')

        title = "Unknown Title"
        author = "Unknown Author"

        if metadata is not None:
            title_elem = metadata.find('.//{http://purl.org/dc/elements/1.1/}title')
            if title_elem is None:
                title_elem = metadata.find('.//{*}title')
            if title_elem is not None and title_elem.text:
                title = title_elem.text

            creator_elem = metadata.find('.//{http://purl.org/dc/elements/1.1/}creator')
            if creator_elem is None:
                creator_elem = metadata.find('.//{*}creator')
            if creator_elem is not None and creator_elem.text:
                author = creator_elem.text

        parsed_book = Book(title=title, author=author)

        # Build manifest (id -> href mapping)
        manifest = {}
        manifest_elem = opf_root.find('.//{http://www.idpf.org/2007/opf}manifest')
        if manifest_elem is None:
            manifest_elem = opf_root.find('.//{*}manifest')

        if manifest_elem is not None:
            for item in manifest_elem.findall('.//{http://www.idpf.org/2007/opf}item'):
                item_id = item.get('id')
                href = item.get('href')
                media_type = item.get('media-type', '')
                if item_id and href:
                    manifest[item_id] = {
                        'href': href,
                        'media_type': media_type
                    }

            # Also try without namespace
            for item in manifest_elem.findall('.//{*}item'):
                item_id = item.get('id')
                href = item.get('href')
                media_type = item.get('media-type', '')
                if item_id and href and item_id not in manifest:
                    manifest[item_id] = {
                        'href': href,
                        'media_type': media_type
                    }

        # Read spine (reading order)
        spine = opf_root.find('.//{http://www.idpf.org/2007/opf}spine')
        if spine is None:
            spine = opf_root.find('.//{*}spine')

        chapter_index = 0
        if spine is not None:
            for itemref in spine.findall('.//{http://www.idpf.org/2007/opf}itemref'):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    item_info = manifest[idref]
                    if 'html' in item_info.get('media_type', '').lower() or \
                       'xml' in item_info.get('media_type', '').lower():

                        # Construct the full path to the content file
                        content_path = item_info['href']
                        if opf_dir:
                            content_path = f"{opf_dir}/{content_path}"

                        try:
                            raw_content = zf.read(content_path).decode('utf-8', errors='ignore')
                            cleaned_html = clean_html(raw_content)
                            plain_text = html_to_text(cleaned_html)

                            # Skip empty chapters
                            if not plain_text.strip():
                                continue

                            chapter_title = extract_chapter_title(cleaned_html, chapter_index)

                            blocks = extract_blocks(cleaned_html)

                            chapter = Chapter(
                                index=chapter_index,
                                title=chapter_title,
                                content_html=cleaned_html,
                                content_text=plain_text,
                                blocks=blocks,
                                href=content_path
                            )

                            parsed_book.chapters.append(chapter)
                            chapter_index += 1

                        except KeyError:
                            # Content file not found, skip
                            continue

            # Also try without namespace
            for itemref in spine.findall('.//{*}itemref'):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    # Check if we already processed this
                    already_processed = any(c.href == manifest[idref]['href'] or
                                          c.href == f"{opf_dir}/{manifest[idref]['href']}"
                                          for c in parsed_book.chapters)
                    if already_processed:
                        continue

                    item_info = manifest[idref]
                    if 'html' in item_info.get('media_type', '').lower() or \
                       'xml' in item_info.get('media_type', '').lower():

                        content_path = item_info['href']
                        if opf_dir:
                            content_path = f"{opf_dir}/{content_path}"

                        try:
                            raw_content = zf.read(content_path).decode('utf-8', errors='ignore')
                            cleaned_html = clean_html(raw_content)
                            plain_text = html_to_text(cleaned_html)

                            if not plain_text.strip():
                                continue

                            chapter_title = extract_chapter_title(cleaned_html, chapter_index)

                            blocks = extract_blocks(cleaned_html)

                            chapter = Chapter(
                                index=chapter_index,
                                title=chapter_title,
                                content_html=cleaned_html,
                                content_text=plain_text,
                                blocks=blocks,
                                href=content_path
                            )

                            parsed_book.chapters.append(chapter)
                            chapter_index += 1

                        except KeyError:
                            continue

    return parsed_book


def extract_chapter_title(html_content: str, default_index: int) -> str:
    """
    Extract chapter title from HTML content.

    Args:
        html_content: HTML content of the chapter
        default_index: Default index to use if no title found

    Returns:
        Chapter title
    """
    if HAS_BS4:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Try to find title in heading tags
        for tag in ['h1', 'h2', 'h3', 'title']:
            elem = soup.find(tag)
            if elem:
                title = elem.get_text(strip=True)
                if title and len(title) < 200:  # Reasonable title length
                    return title
    else:
        # Fallback: use regex
        for pattern in [r'<h1[^>]*>([^<]+)</h1>', r'<h2[^>]*>([^<]+)</h2>',
                       r'<title[^>]*>([^<]+)</title>']:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if title and len(title) < 200:
                    return title

    return f"Chapter {default_index + 1}"


def parse_epub(epub_path: str) -> Book:
    """
    Parse an EPUB file.

    Uses ebooklib if available, otherwise falls back to manual parsing.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        Book object with parsed content
    """
    if HAS_EBOOKLIB:
        return parse_epub_with_ebooklib(epub_path)
    else:
        return parse_epub_manual(epub_path)


def create_notebook(chapter: Chapter, book_title: str, book_author: str,
                   output_format: str = "markdown") -> dict:
    """
    Create a Jupyter notebook for a chapter.

    If the chapter has structural blocks, uses boundary detection to split
    content into multiple semantically-grouped cells. Otherwise falls back
    to a single content cell.

    Args:
        chapter: Chapter object
        book_title: Title of the book
        book_author: Author of the book
        output_format: Format for content cells ("markdown" or "text")

    Returns:
        Notebook dictionary in Jupyter format
    """
    cells = []

    # Title cell with book and chapter info
    title_source = [
        f"# {chapter.title}\n",
        "\n",
        f"*From: {book_title}*\n",
        f"*By: {book_author}*\n"
    ]

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": title_source
    })

    if chapter.blocks:
        # --- Multi-cell path: use boundary detection ---
        boundary = compute_cell_boundaries(chapter.blocks)
        groups = group_blocks_by_boundary(chapter.blocks, boundary)

        for group in groups:
            source_lines = blocks_to_cell_source(group, output_format)
            if source_lines:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": source_lines
                })
    else:
        # --- Fallback: single content cell (no blocks extracted) ---
        if output_format == "markdown":
            content = html_to_markdown(chapter.content_html)
        else:
            content = chapter.content_text

        content_lines = []
        for line in content.split('\n'):
            content_lines.append(line + '\n')

        if content_lines and content_lines[-1] == '\n':
            content_lines = content_lines[:-1]
        elif content_lines:
            content_lines[-1] = content_lines[-1].rstrip('\n')

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": content_lines
        })

    # Create the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "book_metadata": {
                "title": book_title,
                "author": book_author,
                "chapter_index": chapter.index,
                "chapter_title": chapter.title
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def save_notebook(notebook: dict, output_path: str) -> None:
    """
    Save a notebook to a file.

    Args:
        notebook: Notebook dictionary
        output_path: Path to save the notebook
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)


def convert_epub_to_notebooks(epub_path: str, output_dir: Optional[str] = None,
                             output_format: str = "markdown") -> list[str]:
    """
    Convert an EPUB file to Jupyter notebooks, one per chapter.

    Args:
        epub_path: Path to the EPUB file
        output_dir: Directory to save notebooks (default: <epub_name>_notebooks)
        output_format: Format for content ("markdown" or "text")

    Returns:
        List of paths to created notebooks
    """
    epub_path = Path(epub_path)

    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")

    # Parse the EPUB
    print(f"Parsing EPUB: {epub_path}")
    book = parse_epub(str(epub_path))
    print(f"Found {len(book.chapters)} chapters in '{book.title}' by {book.author}")

    # Create output directory
    if output_dir is None:
        output_dir = epub_path.stem + "_notebooks"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a notebook for each chapter
    created_files = []

    for chapter in book.chapters:
        notebook = create_notebook(chapter, book.title, book.author, output_format)

        # Create a safe filename
        safe_title = re.sub(r'[^\w\s-]', '', chapter.title)
        safe_title = re.sub(r'\s+', '_', safe_title)[:50]
        filename = f"chapter_{chapter.index:02d}_{safe_title}.ipynb"

        notebook_path = output_path / filename
        save_notebook(notebook, str(notebook_path))
        created_files.append(str(notebook_path))

        print(f"  Created: {filename}")

    # Create an index notebook
    index_notebook = create_index_notebook(book, created_files)
    index_path = output_path / "00_index.ipynb"
    save_notebook(index_notebook, str(index_path))
    created_files.insert(0, str(index_path))
    print(f"  Created: 00_index.ipynb")

    return created_files


def create_index_notebook(book: Book, chapter_files: list[str]) -> dict:
    """
    Create an index notebook with table of contents.

    Args:
        book: Book object
        chapter_files: List of chapter notebook paths

    Returns:
        Index notebook dictionary
    """
    cells = []

    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {book.title}\n",
            "\n",
            f"**Author:** {book.author}\n",
            "\n",
            "---\n",
            "\n",
            "## Table of Contents\n"
        ]
    })

    # Table of contents
    toc_lines = []
    for i, chapter in enumerate(book.chapters):
        filename = Path(chapter_files[i]).name if i < len(chapter_files) else f"chapter_{i:02d}.ipynb"
        toc_lines.append(f"{i + 1}. [{chapter.title}]({filename})\n")

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": toc_lines
    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "book_metadata": {
                "title": book.title,
                "author": book.author,
                "total_chapters": len(book.chapters)
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.epub
  %(prog)s book.epub -o my_notebooks
  %(prog)s book.epub --format text
        """
    )

    parser.add_argument(
        "epub_file",
        help="Path to the EPUB file to convert"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        help="Output directory for notebooks (default: <epub_name>_notebooks)"
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default="markdown",
        help="Output format for chapter content (default: markdown)"
    )

    parser.add_argument(
        "--list-chapters",
        action="store_true",
        help="List chapters without converting"
    )

    args = parser.parse_args()

    # Check if EPUB file exists
    if not os.path.exists(args.epub_file):
        print(f"Error: File not found: {args.epub_file}", file=sys.stderr)
        sys.exit(1)

    # Check dependencies
    if not HAS_BS4:
        print("Warning: BeautifulSoup not installed. Using basic HTML processing.")
        print("Install with: pip install beautifulsoup4")

    if not HAS_EBOOKLIB:
        print("Warning: ebooklib not installed. Using basic EPUB parsing.")
        print("Install with: pip install ebooklib")

    try:
        if args.list_chapters:
            # Just list chapters
            book = parse_epub(args.epub_file)
            print(f"\nBook: {book.title}")
            print(f"Author: {book.author}")
            print(f"\nChapters ({len(book.chapters)}):")
            for chapter in book.chapters:
                print(f"  {chapter.index + 1}. {chapter.title}")
        else:
            # Convert to notebooks
            created_files = convert_epub_to_notebooks(
                args.epub_file,
                args.output_dir,
                args.format
            )
            print(f"\nSuccessfully created {len(created_files)} notebooks")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
