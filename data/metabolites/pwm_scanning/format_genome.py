#!/usr/bin/env python
"""
Fix the genome fasta file to work with bedtools
"""

import pandas as pd
from Bio import SeqIO
from sys import argv
from pathlib import Path

input_genome = Path(argv[1])
fixed_genome = Path(argv[2])

with open(fixed_genome, "w") as f:
    for item in SeqIO.parse(input_genome, "fasta"):
        f.write(f">{item.description}\n")
        f.write(f"{item.seq}\n")
