import sys
from smart_open import open
from normalize import clean


def preprocess(f):
    num_non_blank_lines = 0
    for line in f:
        if len(line.strip()) == 0:
            if num_non_blank_lines > 1:
                yield ""

            num_non_blank_lines = 0
            continue

        if line.startswith("    "):
            line = f"[TAB] {' '.join(line.strip().split())}"
        else:
            line = ' '.join(line.strip().split())

        num_non_blank_lines += 1
        line = clean(line, minimal=True)

        yield line


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/children_stories.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/children_stories.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
