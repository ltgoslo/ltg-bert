mkdir -p ../data/processed_10M

python3 aochildes.py 10M
python3 bnc_spoken.py 10M
python3 cbt.py 10M
python3 children_stories.py 10M
python3 gutenberg.py 10M
python3 open_subtitles.py 10M
python3 qed.py 10M
python3 simple_wikipedia.py 10M
python3 switchboard.py 10M
python3 wikipedia.py 10M

cat ../data/processed_10M/aochildes.txt ../data/processed_10M/bnc_spoken.txt ../data/processed_10M/cbt.txt ../data/processed_10M/children_stories.txt ../data/processed_10M/gutenberg.txt ../data/processed_10M/open_subtitles.txt ../data/processed_10M/qed.txt ../data/processed_10M/simple_wikipedia.txt ../data/processed_10M/switchboard.txt ../data/processed_10M/wikipedia.txt > ../data/processed_10M/all.txt

python3 segment.py 10M
