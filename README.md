# EPUB to Jupyter Notebook Converter

Convert EPUB files into Jupyter notebooks for use with solve.it.com.

Inspired by [karpathy/reader3](https://github.com/karpathy/reader3) for EPUB parsing.

## Features

- Extracts chapters from EPUB files
- Creates one Jupyter notebook per chapter
- Converts HTML content to Markdown
- Generates index notebook with table of contents
- Works with or without optional dependencies (graceful fallback)

## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install ebooklib beautifulsoup4 lxml
```

## Usage

### Basic usage

```bash
python epub_to_notebook.py book.epub
```

This creates a `book_notebooks/` directory with:
- `00_index.ipynb` - Table of contents
- `chapter_00_*.ipynb` - Individual chapter notebooks

### Specify output directory

```bash
python epub_to_notebook.py book.epub -o my_notebooks
```

### Output as plain text instead of Markdown

```bash
python epub_to_notebook.py book.epub --format text
```

### List chapters without converting

```bash
python epub_to_notebook.py book.epub --list-chapters
```

## Output Format

Each chapter notebook contains:
1. A title cell with chapter name, book title, and author
2. A content cell with the chapter text in Markdown format

The notebooks are compatible with Jupyter and can be used with solve.it.com for interactive reading and analysis.

## Dependencies

- `ebooklib` - EPUB parsing (optional, has fallback)
- `beautifulsoup4` - HTML processing (optional, has fallback)
- `lxml` - XML/HTML parser for BeautifulSoup

The converter works without these dependencies using built-in Python libraries, but the output quality is better with them installed.
