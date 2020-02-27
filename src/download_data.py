# This script is used to download the Chip-Seq dataset.

from os import system
from pathlib import Path

raw_peaks = Path("../data/raw_peaks").resolve()

if not raw_peaks.exists():
    raw_peaks.mkdir()

with open("../data/peak_file_list.txt", "r") as f:
    for line in f:
        line = line.strip()
        uri = f"http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/{line}.gz"
        path = raw_peaks / f"{line.split('/')[1]}.gz"
        system(f"wget {uri} -O {path}")
        system(f"gzip -d {path}")
