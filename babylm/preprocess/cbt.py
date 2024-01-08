import sys
from smart_open import open
from normalize import clean


def preprocess(f):
    for line in f:
        line = line.strip()

        line = line.replace("-LRB-", "(")
        line = line.replace("-LCB-", "{")
        line = line.replace("-LSB-", "[")
        line = line.replace("-RRB-", ")")
        line = line.replace("-RCB-", "}")
        line = line.replace("-RSB-", "]")

        line = line.replace("`` ", '"')
        line = line.replace("``", '"')
        line = line.replace(" ''", '"')
        line = line.replace("''", '"')

        if len(line) == 0:
            yield ""
            continue

        line = clean(line)

        yield line


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/cbt.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/cbt.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
