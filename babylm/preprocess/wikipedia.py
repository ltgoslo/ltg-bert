import sys
from smart_open import open
from normalize import clean
import re


regex_1 = re.compile(r"\[\d+\]")
regex_2 = re.compile(r"\[\[([^\|\]]*)\|*[^\]]*\]\]")
regex_3 = re.compile(r"= = = ([^\=]*) = = =")


def preprocess(f):
    prev_line = None
    for i, line in enumerate(f):
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if i > 0 and line.startswith("= = = "):
            yield ""

        if len(line) == 0:
            continue

        if line.startswith("[[Category:") or line.startswith("[[File:"):
            continue
        
        line = regex_1.sub("", line)
        line = regex_2.sub(r"\1", line)
        line = regex_3.sub(r"\1", line)

        yield line


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/wikipedia.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/wikipedia.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
