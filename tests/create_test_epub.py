#!/usr/bin/env python3
"""Create a simple test EPUB file for testing the converter."""

import os
import zipfile
from pathlib import Path


def create_test_epub(output_path: str = "test_book.epub"):
    """Create a minimal valid EPUB file for testing."""

    # EPUB is a ZIP file with specific structure
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # mimetype must be first and uncompressed
        zf.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)

        # container.xml
        container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''
        zf.writestr('META-INF/container.xml', container_xml)

        # content.opf (package file)
        content_opf = '''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="uid">test-book-001</dc:identifier>
        <dc:title>Test Book: A Sample EPUB</dc:title>
        <dc:creator>Test Author</dc:creator>
        <dc:language>en</dc:language>
    </metadata>
    <manifest>
        <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter3" href="chapter3.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine>
        <itemref idref="chapter1"/>
        <itemref idref="chapter2"/>
        <itemref idref="chapter3"/>
    </spine>
</package>'''
        zf.writestr('OEBPS/content.opf', content_opf)

        # Chapter 1
        chapter1 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 1: Introduction</title>
</head>
<body>
    <h1>Chapter 1: Introduction</h1>

    <p>This is the first chapter of our test book. It contains some <strong>bold text</strong>
    and some <em>italic text</em> to test the HTML to Markdown conversion.</p>

    <h2>A Subsection</h2>

    <p>Here is a paragraph with a <a href="https://example.com">link</a> to test anchor conversion.</p>

    <blockquote>
        This is a blockquote that should be converted to Markdown quote format.
    </blockquote>

    <p>And here is a list:</p>
    <ul>
        <li>First item</li>
        <li>Second item</li>
        <li>Third item</li>
    </ul>

    <p>The end of chapter one.</p>
</body>
</html>'''
        zf.writestr('OEBPS/chapter1.xhtml', chapter1)

        # Chapter 2
        chapter2 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 2: The Middle</title>
</head>
<body>
    <h1>Chapter 2: The Middle</h1>

    <p>Welcome to the second chapter. This chapter tests more complex HTML structures.</p>

    <h2>Code Example</h2>

    <p>Here is some inline <code>code</code> in a paragraph.</p>

    <pre>def hello():
    print("Hello, World!")
    return True</pre>

    <h2>A Table</h2>

    <table>
        <tr>
            <th>Name</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Alpha</td>
            <td>100</td>
        </tr>
        <tr>
            <td>Beta</td>
            <td>200</td>
        </tr>
    </table>

    <p>That concludes chapter two.</p>
</body>
</html>'''
        zf.writestr('OEBPS/chapter2.xhtml', chapter2)

        # Chapter 3
        chapter3 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 3: Conclusion</title>
</head>
<body>
    <h1>Chapter 3: Conclusion</h1>

    <p>This is the final chapter of our test book.</p>

    <h2>Summary</h2>

    <p>In this book, we covered:</p>

    <ol>
        <li>Basic text formatting</li>
        <li>Lists and blockquotes</li>
        <li>Code blocks and tables</li>
    </ol>

    <hr/>

    <p><strong>Thank you for reading!</strong></p>

    <p>This test EPUB was created to verify the epub_to_notebook converter works correctly.</p>
</body>
</html>'''
        zf.writestr('OEBPS/chapter3.xhtml', chapter3)

    print(f"Created test EPUB: {output_path}")
    return output_path


if __name__ == "__main__":
    # Create the test EPUB in the tests directory
    script_dir = Path(__file__).parent
    output_path = script_dir / "test_book.epub"
    create_test_epub(str(output_path))
