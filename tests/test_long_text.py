#!/usr/bin/env python3
"""Test the sentence splitting with long text."""

import sys
sys.path.insert(0, '/home/user/Solvify')

from epub_to_notebook import split_into_sentences, html_to_markdown

# Test the sentence splitting function
long_text = """Determinism means that given a complete physical state at time T1, you can exactly determine the physical state at time T2 via physical laws. To do this you must know everything about the physical state at T1. In the case of trying to predict the future state of a human, to do this would mean knowing the exact configuration of every subatomic particle that makes up every atom that makes up every molecule that makes up the human at time T1. It also means knowing the configuration of every subatomic particle that may impinge upon that specific human during the time interval from T1 to T2. Since photons are the fastest things that can exist, and they move at the speed of light, it means that for every second that ticks by, the configuration of every subatomic particle within one light-second of that person must be known."""

print("Testing split_into_sentences:")
sentences = split_into_sentences(long_text)
for i, s in enumerate(sentences):
    print(f"{i+1}. {s}")

print("\n" + "="*80 + "\n")

# Test with HTML
html_content = f"""
<html>
<body>
<div>{long_text}</div>
</body>
</html>
"""

print("Testing html_to_markdown:")
markdown = html_to_markdown(html_content)
print(markdown)
