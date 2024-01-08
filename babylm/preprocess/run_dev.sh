mkdir -p ../data/processed_dev

python3 aochildes.py dev
python3 bnc_spoken.py dev
python3 cbt.py dev
python3 children_stories.py dev
python3 gutenberg.py dev
python3 open_subtitles.py dev
python3 qed.py dev
python3 simple_wikipedia.py dev
python3 switchboard.py dev
python3 wikipedia.py dev

cat ../data/processed_dev/aochildes.txt ../data/processed_dev/bnc_spoken.txt ../data/processed_dev/cbt.txt ../data/processed_dev/children_stories.txt ../data/processed_dev/gutenberg.txt ../data/processed_dev/open_subtitles.txt ../data/processed_dev/qed.txt ../data/processed_dev/simple_wikipedia.txt ../data/processed_dev/switchboard.txt ../data/processed_dev/wikipedia.txt > ../data/processed_dev/all.txt

python3 segment.py dev
