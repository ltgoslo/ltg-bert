import sys
from smart_open import open
from normalize import clean
import re


regex = re.compile(r"\[\d+\]")


def preprocess(f):
    prev_line = None
    for line in f:
        line = ' '.join(line.strip().split())
        line = regex.sub("", line)
        line = clean(line, minimal=True)

        if len(line) == 0:
            if prev_line is not None and prev_line != "":
                yield prev_line
                yield ""
            prev_line = None
            continue

        if "is a commune. It is" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a commune found" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a city in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a village in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a municipality in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a town in" in line and len(line) < 128:
            prev_line = None
            continue

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        if prev_line is not None:
            yield prev_line

        prev_line = line


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/simple_wikipedia.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/simple_wikipedia.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
