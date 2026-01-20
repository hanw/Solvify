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


@dataclass
class Chapter:
    """Represents a chapter extracted from an EPUB file."""
    index: int
    title: str
    content_html: str
    content_text: str
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

            chapter = Chapter(
                index=chapter_index,
                title=chapter_title,
                content_html=cleaned_html,
                content_text=plain_text,
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

                            chapter = Chapter(
                                index=chapter_index,
                                title=chapter_title,
                                content_html=cleaned_html,
                                content_text=plain_text,
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

                            chapter = Chapter(
                                index=chapter_index,
                                title=chapter_title,
                                content_html=cleaned_html,
                                content_text=plain_text,
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

    # Content cell
    if output_format == "markdown":
        content = html_to_markdown(chapter.content_html)
    else:
        content = chapter.content_text

    # Split content into lines for notebook format
    content_lines = []
    for line in content.split('\n'):
        content_lines.append(line + '\n')

    # Remove trailing newline from last line if it exists
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
