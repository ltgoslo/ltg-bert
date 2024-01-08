import sys
from smart_open import open
from normalize import clean
import html


def preprocess(f):
    prev_line = None
    for line in f:
        line = ' '.join(line.strip().split())

        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:]

        line = clean(line, minimal=True)

        if len(line) == 0:
            if prev_line is None or len(prev_line) > 0:
                yield ""
            prev_line = line
            continue

        if "&lt" in line and "&gt" in line:
            continue

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        if "&lt" in line and "&gt" in line:
            continue

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        if "&lt" in line and "&gt" in line:
            continue

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        if "&lt" in line and "&gt" in line:
            continue

        if "->" in line:
            continue

        if line.endswith(":") and not any(c.isalpha() for c in line):
            continue

        if not line.endswith(":") and not line.startswith('"') and not line.endswith('"') and not (line.startswith("(") and line.endswith(")")) and not (line.startswith("[") and line.endswith("]")) and not (line.startswith("{") and line.endswith("}")):
            line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/qed.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/qed.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
