import nltk
import sys


input_path = f"../data/processed_{sys.argv[1]}/all.txt"
output_path = f"../data/processed_{sys.argv[1]}/segmented.txt"

with open(output_path, "w") as f:
    for line in open(input_path):
        line = line.strip()

        if len(line) == 0:
            f.write('\n')
            continue

        sentences = nltk.sent_tokenize(line)
        sentences = '\n'.join(sentences) 
        f.write(f"{sentences}[PAR]\n")
