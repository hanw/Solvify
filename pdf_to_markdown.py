#!/usr/bin/env python3
"""
PDF to Markdown Converter

Extracts text directly from PDF using PyMuPDF for accuracy and speed.
Extracts embedded figures as image files and places them inline with
the surrounding text based on their position on the page.
Optionally uses Claude vision API to describe figures.

Requirements:
    pip install pymupdf
    pip install anthropic  (optional, for figure descriptions)
"""

import argparse
import base64
import hashlib
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF


def extract_page_content(
    doc: fitz.Document, page_num: int, figures_dir: Path, page_label: str
) -> list[dict]:
    """Extract interleaved text and image blocks from a page, sorted by Y position.

    Returns a list of dicts, each either:
        {"type": "text", "y": float, "content": str}
        {"type": "image", "y": float, "filename": str, "width": int, "height": int}
    """
    page = doc.load_page(page_num)
    blocks = page.get_text("dict")["blocks"]
    result = []
    img_counter = 0
    seen_image_hashes = set()

    for block in blocks:
        y_pos = block["bbox"][1]
        block_type = block.get("type", 0)

        if block_type == 0:  # text block
            lines_text = []
            for line in block.get("lines", []):
                span_texts = []
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    if t:
                        span_texts.append(t)
                line_str = "".join(span_texts).strip()
                if line_str:
                    lines_text.append(line_str)
            content = "\n".join(lines_text)
            if content.strip():
                result.append({"type": "text", "y": y_pos, "content": content})

        elif block_type == 1:  # image block
            # Extract the image via xref from the block
            img_w = block["bbox"][2] - block["bbox"][0]
            img_h = block["bbox"][3] - block["bbox"][1]

            # Skip tiny images
            if img_w < 30 or img_h < 30:
                continue

            # Try to get the image data from the block
            # PyMuPDF dict blocks include image data for type=1
            img_bytes = block.get("image")
            ext = block.get("ext", "png")
            if not ext:
                ext = "png"

            if img_bytes:
                # Deduplicate by content hash
                img_hash = hashlib.md5(img_bytes).hexdigest()
                if img_hash in seen_image_hashes:
                    continue
                seen_image_hashes.add(img_hash)

                img_counter += 1
                filename = f"{page_label}_fig{img_counter}.{ext}"
                filepath = figures_dir / filename
                filepath.write_bytes(img_bytes)
                result.append({
                    "type": "image",
                    "y": y_pos,
                    "filename": filename,
                    "width": int(img_w),
                    "height": int(img_h),
                })

    # Also extract via get_images for higher quality versions
    # and any images not captured by dict blocks
    page_images = page.get_images(full=True)
    seen_xrefs = set()
    for img_info in page_images:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue
        w = base_image.get("width", 0)
        h = base_image.get("height", 0)
        if w < 50 or h < 50:
            continue

        img_data = base_image["image"]

        # Check if we already have this image (by matching file content size)
        already_saved = False
        for item in result:
            if item["type"] != "image":
                continue
            filepath = figures_dir / item["filename"]
            if not filepath.exists():
                continue
            existing_size = filepath.stat().st_size
            # Same data or higher quality replacement
            if existing_size == len(img_data):
                already_saved = True
                break
            # Similar display dimensions = same image, keep higher quality
            if abs(item.get("width", 0) - w) < 50 and abs(item.get("height", 0) - h) < 50:
                if len(img_data) > existing_size:
                    filepath.write_bytes(img_data)
                already_saved = True
                break
        if not already_saved:
            img_counter += 1
            ext = base_image.get("ext", "png")
            filename = f"{page_label}_fig{img_counter}.{ext}"
            filepath = figures_dir / filename
            filepath.write_bytes(img_data)
            result.append({
                "type": "image",
                "y": 9999,
                "filename": filename,
                "width": w,
                "height": h,
            })

    # Sort by Y position
    result.sort(key=lambda b: b["y"])
    return result


def describe_figure_with_llm(client, image_path: Path, model: str) -> str:
    """Use Claude vision to describe a figure."""
    image_bytes = image_path.read_bytes()
    ext = image_path.suffix.lower().lstrip(".")
    media_type = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif", "webp": "image/webp",
    }.get(ext, "image/png")

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this chart/figure in 1-2 sentences. "
                                "Focus on what it shows. Respond in the same language as "
                                "any text in the image. Output ONLY the description."
                            ),
                        },
                    ],
                }],
            )
            return response.content[0].text.strip()
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt * 3)
            else:
                return "图表"
    return "图表"


def convert_pdf(
    pdf_path: str,
    output_path: str | None = None,
    figures_dir: str | None = None,
    describe_figures: bool = False,
    model: str = "claude-sonnet-4-20250514",
    start_page: int = 1,
    end_page: int | None = None,
) -> str:
    """Convert a PDF file to Markdown with figures inline.

    Args:
        pdf_path: Path to the PDF file.
        output_path: Output .md file path.
        figures_dir: Directory to save extracted figures.
        describe_figures: Use Claude API to describe figures.
        model: Claude model for figure descriptions.
        start_page: First page (1-based).
        end_page: Last page (1-based, inclusive).

    Returns:
        Path to the output Markdown file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    else:
        output_path = Path(output_path)

    if figures_dir is None:
        figures_dir = pdf_path.stem + "_figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set up LLM client if needed
    client = None
    if describe_figures:
        try:
            import anthropic
            api_key = os.environ.get("SOLVIFY_ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                client = anthropic.Anthropic(api_key=api_key)
            else:
                print("Warning: No API key found. Skipping figure descriptions.", file=sys.stderr)
                describe_figures = False
        except ImportError:
            print("Warning: anthropic not installed. Skipping figure descriptions.", file=sys.stderr)
            describe_figures = False

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count

    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages, end_page) if end_page else total_pages

    print(f"PDF: {pdf_path} ({total_pages} pages)")
    print(f"Processing pages {start_idx + 1}-{end_idx}")
    print(f"Output: {output_path}")
    print(f"Figures: {figures_dir}")
    if describe_figures:
        print(f"Figure descriptions: ON (model: {model})")
    print()

    total_figures = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_path.stem}\n\n")

        for page_num in range(start_idx, end_idx):
            page_display = page_num + 1
            page_label = f"p{page_display:03d}"

            # Extract interleaved text and images
            blocks = extract_page_content(doc, page_num, figures_dir, page_label)

            fig_count = sum(1 for b in blocks if b["type"] == "image")
            total_figures += fig_count
            fig_indicator = f" [{fig_count} fig]" if fig_count else ""
            print(f"  Page {page_display}/{end_idx}{fig_indicator}", flush=True)

            # Write page marker
            f.write(f"<!-- Page {page_display} -->\n\n")

            # Write blocks in order (text and images interleaved)
            for block in blocks:
                if block["type"] == "text":
                    f.write(block["content"])
                    f.write("\n\n")
                elif block["type"] == "image":
                    fig_path = figures_dir / block["filename"]
                    alt_text = f"Figure from page {page_display}"
                    if describe_figures and client:
                        alt_text = describe_figure_with_llm(client, fig_path, model)
                    f.write(f"![{alt_text}]({fig_path})\n\n")

            f.write("---\n\n")
            f.flush()

    doc.close()

    print(f"\nDone! {end_idx - start_idx} pages processed, {total_figures} figures extracted.")
    print(f"Output: {output_path}")
    print(f"Figures directory: {figures_dir}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown with inline figure extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                         # basic extraction
  %(prog)s document.pdf -o output.md             # custom output path
  %(prog)s document.pdf --start 10 --end 20      # page range
  %(prog)s document.pdf --describe-figures        # use LLM for figure captions
        """,
    )
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output Markdown file path")
    parser.add_argument("--figures-dir", help="Directory to save extracted figures")
    parser.add_argument(
        "--describe-figures", action="store_true",
        help="Use Claude API to generate figure descriptions (requires API key)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514",
        help="Claude model for figure descriptions (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument("--start", type=int, default=1, help="Start page (1-based, default: 1)")
    parser.add_argument("--end", type=int, default=None, help="End page (1-based, inclusive)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: File not found: {args.pdf_file}", file=sys.stderr)
        sys.exit(1)

    convert_pdf(
        pdf_path=args.pdf_file,
        output_path=args.output,
        figures_dir=args.figures_dir,
        describe_figures=args.describe_figures,
        model=args.model,
        start_page=args.start,
        end_page=args.end,
    )


if __name__ == "__main__":
    main()
