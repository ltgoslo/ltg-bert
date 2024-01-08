mkdir -p ../data/processed

python3 aochildes.pyc 100M
python3 bnc_spoken.py 100M
python3 cbt.py 100M
python3 children_stories.py 100M
python3 gutenberg.py 100M
python3 open_subtitles.py 100M
python3 qed.py 100M
python3 simple_wikipedia.py 100M
python3 switchboard.py 100M
python3 wikipedia.py 100M

cat ../data/processed/aochildes.txt ../data/processed/bnc_spoken.txt ../data/processed/cbt.txt ../data/processed/children_stories.txt ../data/processed/gutenberg.txt ../data/processed/open_subtitles.txt ../data/processed/qed.txt ../data/processed/simple_wikipedia.txt ../data/processed/switchboard.txt ../data/processed/wikipedia.txt > ../data/processed/all.txt

python3 segment.py
