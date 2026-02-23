#!/usr/bin/env python3
"""
Markdown to Jupyter Notebook Converter

Converts a markdown file (with optional figure directory) into a single
Jupyter notebook with text split into readable cells and images embedded
as base64 cell attachments.

Usage:
    python md_to_notebook.py ChanLun.md
    python md_to_notebook.py ChanLun.md -o output.ipynb
    python md_to_notebook.py ChanLun.md --figures ChanLun_figures
"""

import argparse
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path


def read_image_as_attachment(image_path: Path) -> tuple[str, str] | None:
    """Read an image file and return (mime_type, base64_data) or None if not found."""
    if not image_path.exists():
        return None
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        suffix = image_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return mime_type, data


def make_markdown_cell(source: list[str], attachments: dict | None = None) -> dict:
    """Create a Jupyter markdown cell."""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }
    if attachments:
        cell["attachments"] = attachments
    return cell


def is_page_marker(line: str) -> bool:
    """Check if a line is a page comment like <!-- Page N -->."""
    return bool(re.match(r"^\s*<!--\s*Page\s+\d+\s*-->\s*$", line))


def is_separator(line: str) -> bool:
    """Check if a line is a horizontal rule separator."""
    return bool(re.match(r"^\s*---\s*$", line))


def is_standalone_page_number(line: str) -> bool:
    """Check if a line is just a page number (standalone digit(s))."""
    return bool(re.match(r"^\s*\d{1,4}\s*$", line))


def is_article_title(line: str) -> bool:
    """Check if a line is an article title (教你炒股票N：... or 股市闲谈：...)."""
    # Match article titles that appear in the body (not in the TOC).
    # TOC lines have …… page-number suffixes; body titles don't.
    if "…" in line:
        return False
    return bool(
        re.match(
            r"^(教你炒股票\s*\d+|股市闲谈)\s*[：:].+\(\d{4}-\d{2}-\d{2}",
            line.strip(),
        )
    )


def is_image_line(line: str) -> re.Match | None:
    """Check if a line is a markdown image reference. Returns the match or None."""
    return re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line.strip())


def parse_md_to_blocks(lines: list[str], md_dir: Path) -> list[dict]:
    """
    Parse markdown lines into blocks suitable for notebook cells.

    Returns a list of block dicts:
        {"type": "title", "text": "..."}
        {"type": "paragraph", "lines": ["..."]}
        {"type": "image", "alt": "...", "path": Path(...)}
    """
    blocks: list[dict] = []
    current_paragraph: list[str] = []

    def flush_paragraph():
        if current_paragraph:
            text = "\n".join(current_paragraph).strip()
            if text:
                blocks.append({"type": "paragraph", "lines": list(current_paragraph)})
            current_paragraph.clear()

    for line in lines:
        stripped = line.rstrip("\n")

        # Skip noise
        if is_page_marker(stripped) or is_separator(stripped) or is_standalone_page_number(stripped):
            flush_paragraph()
            continue

        # Skip blank lines — they delimit paragraphs
        if not stripped.strip():
            flush_paragraph()
            continue

        # Article title
        if is_article_title(stripped):
            flush_paragraph()
            blocks.append({"type": "title", "text": stripped.strip()})
            continue

        # Image
        m = is_image_line(stripped)
        if m:
            flush_paragraph()
            alt_text = m.group(1)
            img_rel = m.group(2)
            img_path = md_dir / img_rel
            blocks.append({"type": "image", "alt": alt_text, "path": img_path})
            continue

        # Regular text line — accumulate into paragraph
        current_paragraph.append(stripped.strip())

    flush_paragraph()
    return blocks


def blocks_to_cells(blocks: list[dict]) -> list[dict]:
    """Convert parsed blocks into Jupyter notebook cells."""
    cells: list[dict] = []

    for block in blocks:
        if block["type"] == "title":
            title_text = block["text"]
            cells.append(make_markdown_cell([f"## {title_text}\n"]))

        elif block["type"] == "paragraph":
            text = "\n".join(block["lines"])
            cells.append(make_markdown_cell([text + "\n"]))

        elif block["type"] == "image":
            img_path: Path = block["path"]
            filename = img_path.name
            result = read_image_as_attachment(img_path)
            if result is None:
                # Image not found — keep as plain markdown reference
                cells.append(
                    make_markdown_cell([f"![{block['alt']}]({img_path})\n"])
                )
            else:
                mime_type, b64_data = result
                attachments = {filename: {mime_type: b64_data}}
                cells.append(
                    make_markdown_cell(
                        [f"![{block['alt']}](attachment:{filename})\n"],
                        attachments=attachments,
                    )
                )

    return cells


def create_notebook(cells: list[dict]) -> dict:
    """Wrap cells into a full notebook JSON structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert(md_path: str, output_path: str | None = None, figures_dir: str | None = None) -> str:
    """
    Convert a markdown file to a Jupyter notebook.

    Args:
        md_path: Path to the markdown file.
        output_path: Path for the output .ipynb (default: same stem + .ipynb).
        figures_dir: Override for the figures directory (default: inferred from image paths).

    Returns:
        Path to the created notebook.
    """
    md_file = Path(md_path)
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")

    md_dir = md_file.parent
    if figures_dir:
        # If a custom figures dir is given, we don't need to change md_dir
        # since image paths in the markdown are relative to the md file's dir.
        pass

    lines = md_file.read_text(encoding="utf-8").splitlines()

    blocks = parse_md_to_blocks(lines, md_dir)
    cells = blocks_to_cells(blocks)

    if not cells:
        print("Warning: no cells were generated.", file=sys.stderr)

    notebook = create_notebook(cells)

    if output_path is None:
        output_path = str(md_file.with_suffix(".ipynb"))

    out = Path(output_path)
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Markdown file to a Jupyter notebook with embedded images.",
    )
    parser.add_argument("md_file", help="Path to the markdown file")
    parser.add_argument("-o", "--output", help="Output notebook path (default: <input>.ipynb)")
    parser.add_argument("--figures", help="Figures directory (default: inferred from image paths)")

    args = parser.parse_args()

    try:
        out = convert(args.md_file, args.output, args.figures)
        print(f"Created: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
