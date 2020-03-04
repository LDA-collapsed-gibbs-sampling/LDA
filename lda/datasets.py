# This script contains functions to load the datasets

from __future__ import absolute_import, unicode_literals  # noqa
import os
import lda.utils


def load_datasets(dataset_path):
    '''
        Function to load the dataset.
    '''
    return lda.utils.ldac2dtm(open(dataset_path), offset=0)


def load_dataset_vocab(dataset_path):
    '''
        Function to load the vocabulary.
    '''
    with open(dataset_path) as f:
        vocabulary = tuple(f.read().split())
    vocab = []
    for v in vocabulary:
        vocab.append(int(v))
    return vocab


def load_dataset_titles(dataset_path):
    '''
        Function to load the dataset titles.
    '''
    with open(dataset_path) as f:
        titles = tuple(line.strip() for line in f.readlines())
    return titles
