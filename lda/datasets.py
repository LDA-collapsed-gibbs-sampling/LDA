from __future__ import absolute_import, unicode_literals  # noqa
import os
import lda.utils


_test_dir = os.path.join(os.path.dirname(__file__), 'tests')


def load_datasets(dataset_name):
        ldac_fn = os.path.join(_test_dir, dataset_name)
        return lda.utils.ldac2dtm(open(ldac_fn), offset=0)


def load_dataset_vocab(dataset_name):
        vocab_fn = os.path.join(_test_dir, dataset_name)
        print(vocab_fn)
        with open(vocab_fn) as f:
            vocab = tuple(f.read().split())
        return vocab


def load_reuters_titles():
        reuters_titles_fn = os.path.join(_test_dir, 'reuters.titles')
        with open(reuters_titles_fn) as f:
            titles = tuple(line.strip() for line in f.readlines())
        return titles
