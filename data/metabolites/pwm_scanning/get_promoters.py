#!/usr/bin/env python
"""
Extract putative promotor regions from all tair genes.
"""

from argparse import ArgumentParser, FileType
from pathlib import Path
import pandas as pd
import numpy as np


def read_gff(file_path: str, parse_attrs: bool) -> pd.DataFrame:
    """Load a GFF file and return a pandas DataFrame."""

    def format_attributes(df):
        for item in df["attributes"]:
            split_on_semi = item.split(";")
            if "=" in split_on_semi[0]:
                yield {
                    x.split("=")[0]: x.split("=")[1] for x in split_on_semi if "=" in x
                }
            else:
                yield "."

    def get_name(row):
        if type(row["attributes"]) == dict:
            if "ID" in row["attributes"].keys():
                return row["attributes"]["ID"]
            elif "Name" in row["attributes"].keys():
                return row["attributes"]["Name"]
            else:
                return "."
        else:
            return "."

    cols = [
        "chrom",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    ]
    df = pd.read_csv(file_path, sep="\t", header=None)
    if df.shape[1] > 9:
        df = df.drop(df.columns[[i for i in range(9, df.shape[1])]], axis=1)

    df = df.rename(columns={i: cols[i] for i in range(df.shape[1])})

    if parse_attrs:
        if df["attributes"].all() != ".":
            df["attributes"] = pd.Series(format_attributes(df))
            df["name"] = df.apply(get_name, axis=1)

    return df


parser = ArgumentParser(usage="extract putative promotor regions from all tair genes")
parser.add_argument(
    "infile", type=str, help="input file containing gene regions, must be GFF",
)
parser.add_argument("outfile", type=str, help="output file to store the edited regions")
parser.add_argument("nucs_up", type=int, help="number of nucleotides upstream")
parser.add_argument("nucs_down", type=int, help="number of nucleotides downstream")
args = parser.parse_args()

chroms = ["Chr1", "Chr2", "Chr3", "Chr4", "Chr5", "ChrM", "ChrC"]
lengths = [30427671, 19698289, 23459830, 18585056, 26975502, 366924, 154478]
chrom_length_df = pd.DataFrame({"chrom": chroms, "length": lengths})

genes_path = Path(args.infile).resolve()
genes = read_gff(genes_path, parse_attrs=False)

genes = genes[genes["type"] == "gene"]

genes["tss"] = np.where(genes["strand"] == "-", genes["end"], genes["start"])

genes["end"] = np.where(
    genes["strand"] == "-", genes["tss"] + args.nucs_up, genes["tss"] + args.nucs_down
)
genes["start"] = np.where(
    genes["strand"] == "-", genes["tss"] - args.nucs_down, genes["tss"] - args.nucs_up
)

genes = pd.merge(genes, chrom_length_df)

genes = genes[(genes["start"] > 0) & (genes["end"] < genes["length"])]

genes = genes.drop(columns=["tss", "length"])

genes = genes.sample(1000)

outpath = Path(args.outfile).resolve()
genes.to_csv(outpath, index=False, header=False, sep="\t")
