#!/usr/bin/env python3
"""Create a richer test EPUB that simulates a real book with longer prose,
multiple sections, topic shifts, and varied HTML structures."""

import zipfile
from pathlib import Path


def create_rich_test_epub(output_path: str = "rich_test_book.epub"):
    """Create a realistic multi-chapter EPUB for testing cell clustering."""

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)

        container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''
        zf.writestr('META-INF/container.xml', container_xml)

        content_opf = '''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="uid">rich-test-001</dc:identifier>
        <dc:title>The Art of Computing</dc:title>
        <dc:creator>Ada Lovelace</dc:creator>
        <dc:language>en</dc:language>
    </metadata>
    <manifest>
        <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
        <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
        <item id="ch3" href="ch3.xhtml" media-type="application/xhtml+xml"/>
        <item id="ch4" href="ch4.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine>
        <itemref idref="ch1"/>
        <itemref idref="ch2"/>
        <itemref idref="ch3"/>
        <itemref idref="ch4"/>
    </spine>
</package>'''
        zf.writestr('OEBPS/content.opf', content_opf)

        # Chapter 1: Long chapter with multiple sections, topic shifts, lists
        ch1 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1: The Dawn of Computation</title></head>
<body>
    <h1>Chapter 1: The Dawn of Computation</h1>

    <p>The history of computation stretches back thousands of years, from the earliest counting
    devices to the sophisticated machines we use today. Understanding this journey helps us
    appreciate the remarkable achievements of human ingenuity and the foundations upon which
    modern computer science rests.</p>

    <p>Ancient civilizations developed remarkably sophisticated methods for calculation. The
    Babylonians used a base-60 number system that still influences how we measure time and
    angles. The Egyptians employed methods of multiplication based on successive doubling.
    These early innovations laid the groundwork for all subsequent mathematical development.</p>

    <h2>The Abacus and Early Mechanical Devices</h2>

    <p>The abacus, invented independently in several cultures, represents one of humanity's
    first computational tools. Chinese merchants used the suanpan for rapid arithmetic, while
    the Roman abacus facilitated commerce across the empire. These devices demonstrate that
    the desire to mechanize calculation is deeply rooted in human culture.</p>

    <p>In the seventeenth century, Blaise Pascal constructed the Pascaline, a mechanical
    calculator capable of addition and subtraction. This device used a series of interlocking
    gears, each representing a decimal digit. When one gear completed a full rotation, it
    would advance the next gear by one position, implementing the carry operation automatically.</p>

    <p>Gottfried Wilhelm Leibniz improved upon Pascal's design with the Stepped Reckoner,
    which could perform all four arithmetic operations. Leibniz also developed the binary
    number system, recognizing that any number could be represented using only zeros and ones.
    This insight would prove prophetic centuries later.</p>

    <h2>Babbage and the Analytical Engine</h2>

    <p>Charles Babbage conceived the Difference Engine in 1822, designed to compute polynomial
    functions automatically. Though never completed in his lifetime, the design demonstrated
    that complex mathematical operations could be mechanized. The British government invested
    heavily in the project before eventually withdrawing support.</p>

    <p>Babbage's more ambitious Analytical Engine, designed in the 1830s, incorporated many
    features we associate with modern computers:</p>

    <ul>
        <li>A "mill" for performing arithmetic operations (analogous to the ALU)</li>
        <li>A "store" for holding numbers (analogous to memory)</li>
        <li>Punched cards for programming (inspired by the Jacquard loom)</li>
        <li>Conditional branching capabilities</li>
        <li>The ability to loop through instructions</li>
    </ul>

    <p>Ada Lovelace, working with Babbage, wrote what is widely considered the first computer
    program — an algorithm for computing Bernoulli numbers on the Analytical Engine. Her notes
    went beyond mere calculation, envisioning that such machines could manipulate symbols
    according to rules and potentially compose music or produce graphics.</p>

    <hr/>

    <p>The transition from mechanical to electronic computation would take another century,
    but the conceptual foundations were firmly established. Babbage and Lovelace had articulated
    the key abstractions — stored programs, conditional execution, and loops — that define
    computation itself.</p>

    <p>Their work influenced generations of engineers and mathematicians who would eventually
    build the electronic computers of the twentieth century. The intellectual lineage from
    Babbage to Turing to von Neumann represents one of the most consequential threads in
    the history of technology.</p>
</body>
</html>'''
        zf.writestr('OEBPS/ch1.xhtml', ch1)

        # Chapter 2: A chapter with code blocks, tables, and blockquotes
        ch2 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2: Algorithms and Data Structures</title></head>
<body>
    <h1>Chapter 2: Algorithms and Data Structures</h1>

    <p>An algorithm is a finite sequence of well-defined instructions for solving a class of
    problems or performing a computation. The study of algorithms is central to computer
    science, providing the theoretical foundation for understanding what can be computed
    efficiently.</p>

    <p>The word "algorithm" derives from the name of the Persian mathematician Muhammad ibn
    Musa al-Khwarizmi, whose ninth-century works introduced Hindu-Arabic numerals and
    algebraic methods to the Western world. His systematic approach to solving equations
    established the paradigm of step-by-step procedural problem solving.</p>

    <h2>Sorting: A Fundamental Problem</h2>

    <p>Sorting is perhaps the most studied problem in computer science. Given a collection
    of items with a defined ordering, arrange them in non-decreasing sequence. Despite its
    apparent simplicity, sorting reveals deep truths about computational complexity.</p>

    <blockquote>
        "If you want to become a good programmer, practice sorting algorithms. If you want
        to become a great programmer, study why sorting algorithms work." — Anonymous
    </blockquote>

    <p>The simplest sorting algorithms, like bubble sort and insertion sort, compare adjacent
    elements and swap them if they are out of order. These algorithms are easy to understand
    and implement but have quadratic time complexity.</p>

    <p>Here is a simple implementation of insertion sort:</p>

    <pre>def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr</pre>

    <h2>Comparison of Sorting Algorithms</h2>

    <p>Different sorting algorithms offer different trade-offs between time complexity,
    space complexity, and practical performance. The following table summarizes the key
    characteristics:</p>

    <table>
        <tr>
            <th>Algorithm</th>
            <th>Best Case</th>
            <th>Average Case</th>
            <th>Worst Case</th>
            <th>Space</th>
        </tr>
        <tr>
            <td>Insertion Sort</td>
            <td>O(n)</td>
            <td>O(n²)</td>
            <td>O(n²)</td>
            <td>O(1)</td>
        </tr>
        <tr>
            <td>Merge Sort</td>
            <td>O(n log n)</td>
            <td>O(n log n)</td>
            <td>O(n log n)</td>
            <td>O(n)</td>
        </tr>
        <tr>
            <td>Quick Sort</td>
            <td>O(n log n)</td>
            <td>O(n log n)</td>
            <td>O(n²)</td>
            <td>O(log n)</td>
        </tr>
        <tr>
            <td>Heap Sort</td>
            <td>O(n log n)</td>
            <td>O(n log n)</td>
            <td>O(n log n)</td>
            <td>O(1)</td>
        </tr>
    </table>

    <h2>Trees and Graphs</h2>

    <p>Beyond arrays and linked lists, trees and graphs provide powerful abstractions for
    representing hierarchical and networked relationships. A tree is a connected acyclic
    graph, while a general graph may contain cycles.</p>

    <p>Binary search trees maintain a sorted structure that enables efficient lookup,
    insertion, and deletion operations. Each node stores a key, and the left subtree
    contains only keys less than the node's key, while the right subtree contains only
    keys greater than the node's key.</p>

    <p>Self-balancing trees like AVL trees and red-black trees guarantee O(log n) operations
    by automatically restructuring after insertions and deletions. These data structures
    form the backbone of databases, file systems, and language runtime libraries.</p>

    <hr/>

    <p>The interplay between algorithms and data structures is fundamental. The choice of
    data structure often determines which algorithms are efficient, and the requirements
    of an algorithm often dictate which data structures are appropriate. This symbiotic
    relationship is at the heart of software engineering.</p>
</body>
</html>'''
        zf.writestr('OEBPS/ch2.xhtml', ch2)

        # Chapter 3: Short chapter with nested divs and sections
        ch3 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 3: The Turing Machine</title></head>
<body>
    <h1>Chapter 3: The Turing Machine</h1>

    <div class="introduction">
        <p>In 1936, Alan Turing published a paper that would fundamentally change our
        understanding of computation. "On Computable Numbers, with an Application to the
        Entscheidungsproblem" introduced the concept of a theoretical machine that could
        simulate any algorithmic process.</p>
    </div>

    <section>
        <h2>Definition and Components</h2>

        <p>A Turing machine consists of several components working together:</p>

        <ol>
            <li>An infinite tape divided into cells, each capable of holding a symbol</li>
            <li>A head that can read and write symbols on the tape and move left or right</li>
            <li>A state register that stores the current state of the machine</li>
            <li>A finite table of instructions (the transition function)</li>
        </ol>

        <p>Despite its simplicity, this abstract device can compute anything that any
        modern computer can compute. This remarkable equivalence is known as the
        Church-Turing thesis.</p>
    </section>

    <section>
        <h2>The Halting Problem</h2>

        <p>Turing's most celebrated result was proving that no general algorithm exists
        to determine whether an arbitrary program will eventually halt or run forever.
        This was the first concrete example of an undecidable problem.</p>

        <p>The proof proceeds by contradiction. Suppose such an algorithm H exists that
        takes a program P and input I and returns true if P halts on I. We can then
        construct a program D that takes a program P as input, runs H(P, P), and does
        the opposite: if H says P halts on itself, D loops forever; if H says P loops,
        D halts.</p>

        <p>Now consider what happens when we run D on itself. If D(D) halts, then H(D, D)
        must have returned false, meaning D should loop. But if D(D) loops, then H(D, D)
        must have returned true, meaning D should halt. Either way we reach a contradiction,
        proving that H cannot exist.</p>

        <blockquote>
            "We can only see a short distance ahead, but we can see plenty there that
            needs to be done." — Alan Turing
        </blockquote>
    </section>

    <section>
        <h2>Legacy and Impact</h2>

        <p>The Turing machine formalism provided the theoretical foundation for the entire
        field of computer science. It gave us a precise definition of what it means for a
        function to be computable, established fundamental limits on computation, and
        inspired the design of practical stored-program computers.</p>

        <p>Today, the ACM Turing Award — often called the "Nobel Prize of Computing" —
        honors individuals who have made lasting contributions to the field. Recipients
        include luminaries such as Dijkstra, Knuth, Rivest, and Berners-Lee.</p>
    </section>
</body>
</html>'''
        zf.writestr('OEBPS/ch3.xhtml', ch3)

        # Chapter 4: Long flat prose (no headings mid-chapter) to test lexical + size splitting
        ch4 = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 4: Reflections on Artificial Intelligence</title></head>
<body>
    <h1>Chapter 4: Reflections on Artificial Intelligence</h1>

    <p>The quest to create intelligent machines has captivated humanity for centuries.
    From the mythological automata of ancient Greece to the chess-playing Turk of the
    eighteenth century, people have long imagined mechanical minds capable of thought
    and reason. Modern artificial intelligence research, born in the mid-twentieth
    century, has transformed these fantasies into engineering challenges.</p>

    <p>The Dartmouth Conference of 1956 is generally considered the founding event of
    artificial intelligence as a field. John McCarthy, Marvin Minsky, Nathaniel Rochester,
    and Claude Shannon proposed a summer research project based on the conjecture that
    "every aspect of learning or any other feature of intelligence can in principle be so
    precisely described that a machine can be made to simulate it."</p>

    <p>Early AI research was characterized by unbridled optimism. Herbert Simon predicted
    in 1957 that within ten years a computer would be chess champion and would discover
    and prove an important mathematical theorem. While these predictions proved premature,
    the ambition they reflected drove decades of productive research.</p>

    <p>The first major approach to AI was symbolic reasoning, also known as Good Old-Fashioned
    AI. This paradigm represented knowledge as symbols and rules, manipulating them through
    logical inference. Expert systems, which encoded domain knowledge as if-then rules,
    achieved commercial success in the 1980s for tasks like medical diagnosis and mineral
    exploration.</p>

    <p>However, symbolic AI struggled with tasks that humans find effortless: recognizing
    faces, understanding natural language, and navigating physical environments. These
    capabilities seemed to require a different approach entirely.</p>

    <p>The connectionist revolution brought neural networks to the forefront. Inspired by
    the structure of biological brains, these systems learn patterns from data rather than
    following explicit rules. The perceptron, invented by Frank Rosenblatt in 1958, was
    the first neural network capable of learning. Though limited in its original form,
    the core idea of learning through connection weights would eventually triumph.</p>

    <p>The development of backpropagation in the 1980s enabled training of multi-layer
    neural networks. Geoffrey Hinton, David Rumelhart, and Ronald Williams demonstrated
    that gradient descent could effectively adjust weights throughout a deep network,
    solving the credit assignment problem that had stymied earlier researchers.</p>

    <p>Despite this breakthrough, neural networks fell out of favor in the 1990s as
    support vector machines and other kernel methods achieved superior performance on
    many benchmarks. The machine learning community largely moved away from neural
    approaches, viewing them as theoretically unprincipled and computationally expensive.</p>

    <p>The deep learning revolution began around 2012, when convolutional neural networks
    dramatically outperformed traditional methods on the ImageNet image classification
    challenge. This success was enabled by three converging factors: vastly larger datasets,
    more powerful GPU hardware, and algorithmic innovations like dropout regularization
    and batch normalization.</p>

    <p>Since then, deep learning has transformed field after field. Computer vision systems
    now exceed human performance on many visual recognition tasks. Machine translation
    has improved dramatically through attention mechanisms and transformer architectures.
    Game-playing systems have defeated world champions in Go, poker, and StarCraft.</p>

    <p>Large language models represent the latest frontier. Trained on vast corpora of
    text from the internet, these models demonstrate remarkable capabilities in natural
    language understanding, generation, and reasoning. They can write code, summarize
    documents, answer questions, and engage in open-ended conversation.</p>

    <p>Yet fundamental questions remain unanswered. Do these systems truly understand
    language, or are they sophisticated pattern matchers? Can scaling current approaches
    lead to artificial general intelligence, or are qualitatively different architectures
    needed? How do we ensure that increasingly capable AI systems remain aligned with
    human values and intentions?</p>

    <p>The philosophical implications are equally profound. If a machine can pass the
    Turing test, does it possess consciousness? The Chinese Room argument, proposed by
    John Searle, challenges the notion that symbol manipulation alone constitutes
    understanding. Searle imagines a person in a room following rules to produce
    Chinese responses without understanding Chinese, arguing that computers similarly
    lack genuine comprehension.</p>

    <p>As we stand at this inflection point, the future of artificial intelligence
    remains both thrilling and uncertain. The technology holds immense promise for
    scientific discovery, medical breakthroughs, and human flourishing. But it also
    poses risks that demand careful consideration — from labor displacement to
    autonomous weapons to the alignment problem itself.</p>

    <p>Perhaps the most important lesson from the history of AI is humility. The field
    has repeatedly oscillated between exuberant optimism and deep disappointment. The
    problems that seemed easy turned out to be hard, and the problems that seemed
    impossibly hard yielded to unexpected approaches. Whatever the future holds, it
    will likely surprise us.</p>
</body>
</html>'''
        zf.writestr('OEBPS/ch4.xhtml', ch4)

    print(f"Created rich test EPUB: {output_path}")
    return output_path


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_path = script_dir / "rich_test_book.epub"
    create_rich_test_epub(str(output_path))
