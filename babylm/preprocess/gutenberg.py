import sys
from smart_open import open
from normalize import clean


def preprocess(f):
    last_num_non_blank_lines = 0
    num_blank_lines = 0
    accumulated_line = []
    for line in f:
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if len(line) == 0:
            if len(accumulated_line) > 0:
                yield ' '.join(accumulated_line)
                last_num_non_blank_lines = len(accumulated_line)

            if num_blank_lines == 1 and last_num_non_blank_lines > 1:
                yield ""

            accumulated_line = []
            num_blank_lines += 1
            continue

        num_blank_lines = 0
        accumulated_line.append(line)


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/gutenberg.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/gutenberg.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
