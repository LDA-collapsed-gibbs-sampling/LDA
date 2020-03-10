#!/usr/bin/env python
"""
Remove lines from the output of <bedtools nuc> that have ambiguous DNA chars
"""

import pandas as pd
from Bio import SeqIO
from sys import argv
from pathlib import Path

infile = Path(argv[1])
outfile = Path(argv[2])

counts = pd.read_csv(infile, sep="\t")

counts = counts[(counts["16_num_N"] == 0) & (counts["17_num_oth"] == 0)]
counts = counts.drop(columns=counts.columns[9:].values)

counts.to_csv(
    outfile, sep="\t", index=False, header=False,
)
