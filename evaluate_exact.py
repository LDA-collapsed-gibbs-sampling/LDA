'''
	Evaluation Script
	This script is used to evaluate the probability of a document for some held-out document.
'''

from lda.exact import exact_inference 
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
	parser.add_argument("--topic-distr", type=str, default=cfg.TOPIC_DISTR, help="Pickled file for topic-word distribution")
	parser.add_argument("--data-file", type=str, default=cfg.DATA_FILE, help="Data filename")
	parser.add_argument("--test-doc", type=str, default=cfg.TEST_DOC, help="Testing document")
	args = parser.parse_args()

	topic_word = np.load(args.topic_distr, allow_pickle=True) 						# load the topic word distribution
	topic_prior = np.repeat(args.alpha, args.n_topics).reshape(1, args.n_topics)	# create a topic prior

	# evaluate for each document.
	topic_list = list(range(args.n_topics))

	testing_file = os.path.join(cfg.DATA_DIR, args.test_doc)
	log_prob, count = 0, 0
	documents = open(testing_file).readlines()

	exact, importance = [], []

	for d in documents:
		count += 1
		d = d.strip()
		d = d.split(" ")[1:]
		document = []
		for val in d:
			val = val.split(":")
			document.extend(np.repeat(int(val[0]), int(val[-1])))
		random.shuffle(document)
		log_prob += exact_inference(document, topic_word, topic_prior)
		print("Evaluated document: ", count)

	print("Log probability on the testing dataset: {:.3f}".format(log_prob/count))
