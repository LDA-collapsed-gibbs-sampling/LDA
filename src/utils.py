from pandas import read_csv
from pathlib import Path
from os import system


def read_peak(path: Path):
    """Read a narrowPeak file as defined at http://genome.ucsc.edu/FAQ/FAQformat.html#format12"""
    return read_csv(Path(path).resolve(), sep="\t", header=None).rename(
        columns={
            0: "chrom",
            1: "start",
            2: "end",
            3: "name",
            4: "score",
            5: "strand",
            6: "sig_val",
            7: "pval",
            8: "qval",
            9: "peak",
        }
    )


def download_data(peaks_file: str):
    raw_peaks = Path("data").resolve() / "raw_peaks"

    if not raw_peaks.exists():
        raw_peaks.mkdir()

    with open(Path(peaks_file).resolve(), "r") as f:
        for line in f:
            line = line.strip()
            uri = f"http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/{line}.gz"
            path = raw_peaks / f"{line.split('/')[1]}.gz"
            system(f"wget {uri} -O {path}")
            system(f"gzip -d {path}")
