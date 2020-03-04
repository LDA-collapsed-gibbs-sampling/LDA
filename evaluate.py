'''
	Evaluation Script
	This script is used to evaluate the probability of a document for some held-out document.
'''

from lda.importance_sampling import importance_sampling 
import argparse
from config import config as cfg
import random
import numpy as np
import lda.datasets as dataset
import os

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Evaluation parameters')
	parser.add_argument("--n-topics", type=int, default=cfg.N_TOPICS, help="No of topics")
	parser.add_argument("--num-samples", default=cfg.NUM_SAMPLES, help="No of samples")
	parser.add_argument("--alpha", default=cfg.ALPHA, help="Dirichlet prior for topics")
	parser.add_argument("--tokens-file", type=str, default=cfg.TOKENS_FILE, help="Tokens file")
	parser.add_argument("--topic-distr", type=str, default=cfg.TOPIC_DISTR, help="Pickled file for topic-word distribution")
	parser.add_argument("--data-file", type=str, default=cfg.DATA_FILE, help="Data filename")
	parser.add_argument("--test-doc", type=str, default=cfg.TEST_DOC, help="Testing document")
	args = parser.parse_args()

	topic_word = np.load(args.topic_distr, allow_pickle=True)
	topic_prior = np.repeat(args.alpha, args.n_topics).reshape(1, args.n_topics)
	
	# load the vocabulary.
	# vocab = dataset.load_dataset_vocab(os.path.join(cfg.DATA_DIR, args.tokens_file))

	# evaluate for each document.
	topic_list = list(range(args.n_topics))
	# documents = [{2:39, 3:30, 1:28, 4:48, 0:26, 5:29}]
	# documents = [{0:8, 7:3, 4:7, 1:3, 8:10, 6:7, 3:2, 2:1, 9:4, 5:5}, {0:1, 7:40, 4:2, 1:2, 8:2, 6:2, 3:1}]

	# load the documents
	# documents = open(args.test_doc).readlines()
	
	testing_file = os.path.join(cfg.DATA_DIR, args.test_doc)
	log_prob, count = 0, 0

	documents = open(testing_file).readlines()

	for d in documents:
		count += 1
		d = d.strip()
		d = d.split(" ")[1:]
		document = []
		for val in d:
			val = val.split(":")
			document.extend(np.repeat(int(val[0]), int(val[-1])))
		# for k, v in d.items():	document.extend(np.repeat(k, v))
		random.shuffle(document)
		log_prob += importance_sampling(document, topic_word, topic_prior, args.num_samples, topic_list)

	print("Log probability on the testing dataset: {:.3f}".format(log_prob/count))