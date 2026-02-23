#!/usr/bin/env python3
"""
epub_to_notebook.py – Convert an EPUB book into a Jupyter notebook (.ipynb)
with fine-grained cell splitting (one cell per section, not per chapter).

Usage:
    python epub_to_notebook.py <input.epub> [output.ipynb]

Dependencies: beautifulsoup4, lxml  (stdlib: zipfile, json, re, xml)
"""

import zipfile
import json
import re
import sys
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup, NavigableString, Tag


# ---------------------------------------------------------------------------
# EPUB reading helpers (no ebooklib dependency)
# ---------------------------------------------------------------------------

def read_epub(epub_path: str) -> dict:
    """Return {spine_order: [(file_path, html_bytes), ...], metadata: {...}}."""
    z = zipfile.ZipFile(epub_path)

    # 1. Find rootfile from META-INF/container.xml
    container = ET.fromstring(z.read("META-INF/container.xml"))
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile_path = container.find(".//c:rootfile", ns).attrib["full-path"]
    rootfile_dir = os.path.dirname(rootfile_path)

    # 2. Parse the OPF (content.opf) for manifest + spine
    opf = ET.fromstring(z.read(rootfile_path))
    opf_ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
    manifest = {}
    for item in opf.findall(".//opf:manifest/opf:item", opf_ns):
        manifest[item.attrib["id"]] = item.attrib["href"]

    spine_ids = [
        ref.attrib["idref"]
        for ref in opf.findall(".//opf:spine/opf:itemref", opf_ns)
    ]

    # 3. Read metadata
    meta = {}
    title_el = opf.find(".//dc:title", opf_ns)
    if title_el is not None:
        meta["title"] = title_el.text
    author_el = opf.find(".//dc:creator", opf_ns)
    if author_el is not None:
        meta["author"] = author_el.text

    # 4. Build spine items
    spine = []
    for sid in spine_ids:
        href = manifest.get(sid, "")
        full = os.path.join(rootfile_dir, href) if rootfile_dir else href
        try:
            spine.append((href, z.read(full)))
        except KeyError:
            pass

    return {"spine": spine, "metadata": meta, "zip": z, "rootdir": rootfile_dir}


# ---------------------------------------------------------------------------
# HTML → Markdown converter (lightweight, tailored for technical books)
# ---------------------------------------------------------------------------

def html_to_markdown(element, depth=0) -> str:
    """Convert a BeautifulSoup element tree to Markdown.
    
    This is the public entry point that handles the element itself 
    (if it's a known tag) or iterates over its children.
    """
    if isinstance(element, Tag) and element.name in _HANDLED_TAGS:
        result = _tag_to_md(element, depth)
    else:
        result = _children_to_md(element, depth)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# Tags that _tag_to_md knows how to handle
_HANDLED_TAGS = frozenset({
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "pre", "code", "blockquote",
    "ul", "ol", "li", "table",
    "em", "i", "strong", "b", "a", "img",
    "figure", "figcaption", "dl", "sup", "sub",
})


def _children_to_md(element, depth=0) -> str:
    """Convert only the children of an element to Markdown (no wrapping)."""
    parts = []
    for child in element.children:
        if isinstance(child, NavigableString):
            text = str(child)
            text = re.sub(r"[ \t]+", " ", text)
            parts.append(text)
        elif isinstance(child, Tag):
            parts.append(_tag_to_md(child, depth))
    return "".join(parts)


def _tag_to_md(tag: Tag, depth: int) -> str:
    name = tag.name

    # --- Headings ---
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = tag.get_text().strip()
        return f"\n\n{'#' * level} {text}\n\n"

    # --- Paragraphs ---
    if name == "p":
        inner = _children_to_md(tag, depth)
        return f"\n\n{inner}\n\n"

    # --- Emphasis / Strong ---
    if name in ("em", "i"):
        inner = _children_to_md(tag, depth)
        return f"*{inner.strip()}*"
    if name in ("strong", "b"):
        inner = _children_to_md(tag, depth)
        return f"**{inner.strip()}**"

    # --- Code ---
    if name == "code":
        inner = tag.get_text()
        if "\n" in inner:
            return f"\n```\n{inner.strip()}\n```\n"
        return f"`{inner}`"
    if name == "pre":
        code = tag.find("code")
        text = code.get_text() if code else tag.get_text()
        data_type = tag.get("data-type", "")
        return f"\n```\n{text.strip()}\n```\n"

    # --- Lists ---
    if name in ("ul", "ol"):
        items = []
        for i, li in enumerate(tag.find_all("li", recursive=False)):
            prefix = f"{i+1}." if name == "ol" else "-"
            text = _children_to_md(li, depth + 1).strip()
            # Indent continuation lines
            lines = text.split("\n")
            indented = lines[0]
            for line in lines[1:]:
                indented += "\n  " + line
            items.append(f"{prefix} {indented}")
        return "\n\n" + "\n".join(items) + "\n\n"

    if name == "li":
        return _children_to_md(tag, depth)

    # --- Blockquote ---
    if name == "blockquote":
        inner = _children_to_md(tag, depth).strip()
        quoted = "\n".join("> " + line for line in inner.split("\n"))
        return f"\n\n{quoted}\n\n"

    # --- Links ---
    if name == "a":
        text = tag.get_text().strip()
        href = tag.get("href", "")
        if href and not href.startswith("#"):
            return f"[{text}]({href})"
        return text

    # --- Images ---
    if name == "img":
        alt = tag.get("alt", "")
        src = tag.get("src", "")
        return f"![{alt}]({src})"

    # --- Figures ---
    if name == "figure":
        inner = _children_to_md(tag, depth)
        return f"\n\n{inner.strip()}\n\n"
    if name == "figcaption":
        text = tag.get_text().strip()
        return f"\n*{text}*\n"

    # --- Tables ---
    if name == "table":
        return _table_to_md(tag)

    # --- Definition lists ---
    if name == "dl":
        parts = []
        for child in tag.children:
            if isinstance(child, Tag):
                if child.name == "dt":
                    parts.append(f"\n**{child.get_text().strip()}**")
                elif child.name == "dd":
                    parts.append(f"\n: {_children_to_md(child, depth).strip()}")
        return "\n".join(parts) + "\n"

    # --- Superscript (footnotes) ---
    if name == "sup":
        text = tag.get_text().strip()
        return f"^{text}"
    if name == "sub":
        return tag.get_text().strip()

    # --- Divs/Sections/Spans: recurse ---
    if name in ("div", "section", "span", "aside", "header", "footer", "nav",
                 "article", "main", "details", "summary", "mark"):
        # Check for special data-types
        data_type = tag.get("data-type", "")
        inner = _children_to_md(tag, depth)
        if data_type == "note" or data_type == "warning" or data_type == "tip":
            label = data_type.upper()
            return f"\n\n> **{label}:** {inner.strip()}\n\n"
        if data_type == "example":
            return f"\n\n---\n\n{inner.strip()}\n\n---\n\n"
        return inner

    # --- Skip script/style ---
    if name in ("script", "style", "meta", "link"):
        return ""

    # --- Fallback: just recurse ---
    return _children_to_md(tag, depth)


def _table_to_md(table: Tag) -> str:
    """Convert an HTML table to markdown."""
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text().strip().replace("|", "\\|"))
        rows.append(cells)

    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Cell splitting logic
# ---------------------------------------------------------------------------

def split_chapter_into_cells(html_bytes: bytes) -> list[dict]:
    """
    Split a chapter HTML into multiple notebook cells.
    Returns list of {"source": str, "cell_type": "markdown"} dicts.

    Strategy:
    - Chapter intro (title + paragraphs before first sect1) → 1 cell
    - Each sect1 with no sect2 children → 1 cell
    - Each sect1 with sect2 children:
        - sect1 intro (title + paragraphs before first sect2) → 1 cell
        - Each sect2 → 1 cell
    - Footnotes/references → 1 cell
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    cells = []

    # Find the main content container
    chapter_div = (
        soup.find("div", class_="chapter")
        or soup.find("div", class_="part")
        or soup.find("div", class_="preface")
        or soup.find("div", class_="appendix")
        or soup.find("div", class_="glossary")
        or soup.find("section")
        or soup.find("body")
        or soup
    )

    # Collect top-level children of the chapter div
    sect1_sections = chapter_div.find_all("section", attrs={"data-type": "sect1"}, recursive=False)

    if not sect1_sections:
        # No sect1 structure – look for div.sect1 inside sections
        wrapper_sections = chapter_div.find_all("section", recursive=False)
        if wrapper_sections:
            for ws in wrapper_sections:
                inner_sect1s = ws.find_all("section", attrs={"data-type": "sect1"}, recursive=False)
                sect1_sections.extend(inner_sect1s)

    # --- Build intro cell (everything before first sect1) ---
    intro_parts = []
    for child in chapter_div.children:
        if isinstance(child, Tag):
            if child.name == "section" and child.get("data-type") == "sect1":
                break
            # Skip map images for cleaner output
            if child.get("class") and "map-ebook" in str(child.get("class", "")):
                continue
            md = html_to_markdown(child)
            if md.strip():
                intro_parts.append(md)
        elif isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                intro_parts.append(text)

    if intro_parts:
        cells.append(_make_cell("\n\n".join(intro_parts)))

    # --- Process each sect1 ---
    for sect1 in sect1_sections:
        # The sect1 may contain a wrapper div
        inner = sect1.find("div", class_="sect1") or sect1
        sect2_sections = inner.find_all("section", attrs={"data-type": "sect2"}, recursive=False)

        if not sect2_sections:
            # No subsections: entire sect1 is one cell
            md = html_to_markdown(inner)
            if md.strip():
                cells.append(_make_cell(md))
        else:
            # sect1 intro before first sect2
            sect1_intro_parts = []
            for child in inner.children:
                if isinstance(child, Tag):
                    if child.name == "section" and child.get("data-type") == "sect2":
                        break
                    md = html_to_markdown(child)
                    if md.strip():
                        sect1_intro_parts.append(md)
            if sect1_intro_parts:
                cells.append(_make_cell("\n\n".join(sect1_intro_parts)))

            # Each sect2 as its own cell
            for sect2 in sect2_sections:
                inner2 = sect2.find("div", class_="sect2") or sect2
                # Check for sect3 inside sect2
                sect3_sections = inner2.find_all("section", attrs={"data-type": "sect3"}, recursive=False)
                if not sect3_sections:
                    md = html_to_markdown(inner2)
                    if md.strip():
                        cells.append(_make_cell(md))
                else:
                    # sect2 intro
                    s2_intro = []
                    for child in inner2.children:
                        if isinstance(child, Tag):
                            if child.name == "section" and child.get("data-type") == "sect3":
                                break
                            md = html_to_markdown(child)
                            if md.strip():
                                s2_intro.append(md)
                    if s2_intro:
                        cells.append(_make_cell("\n\n".join(s2_intro)))
                    for sect3 in sect3_sections:
                        inner3 = sect3.find("div", class_="sect3") or sect3
                        md = html_to_markdown(inner3)
                        if md.strip():
                            cells.append(_make_cell(md))

    # --- Footnotes / References ---
    for fn_div in chapter_div.find_all("div", attrs={"data-type": "footnotes"}):
        md = html_to_markdown(fn_div)
        if md.strip():
            cells.append(_make_cell(md))

    return cells


def _make_cell(source: str) -> dict:
    """Create a notebook markdown cell dict."""
    # Clean up the source
    source = re.sub(r"\n{3,}", "\n\n", source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n"),  # will be joined later
    }


# ---------------------------------------------------------------------------
# Classify spine items
# ---------------------------------------------------------------------------

def classify_spine_item(href: str) -> str:
    """Classify a spine item by its filename."""
    basename = os.path.basename(href).lower()
    if re.match(r"ch\d+", basename):
        return "chapter"
    if "part" in basename:
        return "part"
    if "preface" in basename:
        return "preface"
    if "toc" in basename:
        return "toc"
    if "cover" in basename:
        return "cover"
    if "title" in basename:
        return "titlepage"
    if "copyright" in basename:
        return "copyright"
    if "dedication" in basename:
        return "dedication"
    if "glossary" in basename:
        return "glossary"
    if "colophon" in basename:
        return "colophon"
    if "ix0" in basename or "index" in basename:
        return "index"
    return "other"


# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------

def epub_to_notebook(epub_path: str, output_path: str = None,
                     include_frontmatter: bool = True,
                     include_backmatter: bool = False):
    """Convert an EPUB to a Jupyter notebook with multi-cell chapters."""
    epub_data = read_epub(epub_path)
    spine = epub_data["spine"]
    meta = epub_data["metadata"]

    all_cells = []

    # Title cell
    title = meta.get("title", "Untitled")
    author = meta.get("author", "")
    title_md = f"# {title}"
    if author:
        title_md += f"\n\n*{author}*"
    all_cells.append(_make_cell(title_md))

    frontmatter_types = {"preface", "dedication"}
    backmatter_types = {"glossary", "colophon", "index"}
    skip_types = {"cover", "titlepage", "copyright", "toc"}

    for href, html_bytes in spine:
        item_type = classify_spine_item(href)

        if item_type in skip_types:
            continue
        if item_type in frontmatter_types and not include_frontmatter:
            continue
        if item_type in backmatter_types and not include_backmatter:
            continue

        if item_type in ("chapter", "preface", "glossary"):
            # Fine-grained splitting
            chapter_cells = split_chapter_into_cells(html_bytes)
            if chapter_cells:
                # Add a separator cell before each chapter
                all_cells.append(_make_cell("---"))
                all_cells.extend(chapter_cells)
        elif item_type == "part":
            # Part title pages → single cell
            soup = BeautifulSoup(html_bytes, "lxml")
            md = html_to_markdown(soup)
            if md.strip():
                all_cells.append(_make_cell(f"---\n\n{md}"))
        else:
            soup = BeautifulSoup(html_bytes, "lxml")
            md = html_to_markdown(soup)
            if md.strip():
                all_cells.append(_make_cell(md))

    # Build the notebook JSON
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "title": title,
            "author": author,
        },
        "cells": [],
    }

    for cell in all_cells:
        # Convert source lines to proper format (list of strings with \n)
        source_lines = cell["source"]
        if isinstance(source_lines, list):
            # Add \n to each line except the last
            formatted = [line + "\n" for line in source_lines[:-1]]
            if source_lines:
                formatted.append(source_lines[-1])
            source_lines = formatted

        notebook["cells"].append({
            "cell_type": cell["cell_type"],
            "metadata": cell.get("metadata", {}),
            "source": source_lines,
        })

    if output_path is None:
        base = os.path.splitext(os.path.basename(epub_path))[0]
        output_path = f"{base}.ipynb"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Created: {output_path}")
    print(f"  Cells: {len(notebook['cells'])}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python epub_to_notebook.py <input.epub> [output.ipynb]")
        sys.exit(1)

    epub_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    epub_to_notebook(epub_path, output_path)
