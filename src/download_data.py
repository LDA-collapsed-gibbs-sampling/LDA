from os import system
from pathlib import Path

with open("../data/peak_file_list.txt", "r") as f:
    for line in f:
        line = line.strip()
        uri = f"http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/{line}.gz"
        path = Path(f"../data/raw_peaks/{line.split('/')[1]}.gz").resolve()
        system(f"wget {uri} -O {path}")
        system(f"gzip -d {path}")
