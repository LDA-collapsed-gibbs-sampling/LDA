'''
	Evaluation Script

	This script is used to evaluate the probability of a document for some held-out document.
	Importance vs LTR vs Exact Inference.
'''

from lda.importance_sampling import importance_sampling 
from lda.exact_inference import exact_inference
from lda.lefttoright import leftToRight
import argparse
from config import config as cfg
import random
import numpy as np
import lda.datasets as dataset
import os
import matplotlib.pyplot as plt
from math import log, exp

def compare_perplexity(nsamples, importance, ltr, exact):
	'''
		Perplexity graph for comparing different inference techniques.
	'''
	fig, ax = plt.subplots()
	plt.plot(nsamples, exact, label='exact')
	plt.plot(nsamples, importance, label='importance')
	plt.plot(nsamples, ltr, label='left to right')
	plt.xticks(nsamples, rotation=45)
	plt.title('Perplexity vs No of samples')
	plt.xlabel('No of samples')
	plt.ylabel('Perplexity')
	plt.legend()
	plt.show()
	plt.savefig('importance_exact.png')


def load_arguments():
	'''
		Load the arguments.
	'''
	parser = argparse.ArgumentParser(description='Evaluation parameters')
	parser.add_argument("--n-topics", type=int, default=cfg.N_TOPICS, help="No of topics")
	parser.add_argument("--num-samples", default=cfg.NUM_SAMPLES_LIST, help="No of samples")
	parser.add_argument("--alpha", default=cfg.ALPHA, help="Dirichlet prior for topics")
	parser.add_argument("--tokens-file", type=str, default=cfg.TOKENS_FILE, help="Tokens file")
	parser.add_argument("--topic-distr", type=str, default=cfg.TOPIC_DISTR, help="Pickled file for topic-word distribution")
	parser.add_argument("--data-file", type=str, default=cfg.DATA_FILE, help="Data filename")
	parser.add_argument("--test-doc", type=str, default=cfg.TEST_DOC, help="Testing document")
	args = parser.parse_args()
	return args

if __name__=='__main__':
	args = load_arguments()

	# load the topic word distribution
	topic_word = np.load(args.topic_distr+'.pkl', allow_pickle=True)
	topic_prior = np.repeat(args.alpha, args.n_topics)
	assert(topic_word.shape[0]==args.n_topics)

	# load the vocabulary.
	vocabfile = open(os.path.join(cfg.TEST_DIR, args.tokens_file)).readlines() 
	vocab = []
	for v in vocabfile:
		v = v.strip()
		vocab.append(int(v))
	print("vocab: ", vocab)
    	
	# evaluate for each document.
	topic_list = list(range(args.n_topics))
	
	# load the testing documents
	testing_file = os.path.join(cfg.TEST_DIR, args.test_doc)
	documents = open(testing_file).readlines()									# load the documents

	# Exact inference
	log_prob, count = 0, 0
	count, doc_length = 0, 0
	document_list = []
	for d in documents[:5]:
		count += 1
		d = d.strip()
		d = d.split(" ")[1:]
		document = []
		for val in d:
			val = val.split(":")
			document.extend(np.repeat(int(val[0]), int(val[-1])))
		doc_length += len(document)
		random.shuffle(document)
		document_list.append(document)
		log_prob += exact_inference(document, topic_prior, topic_word, vocab)
	
	log_exact = exp(log_prob /(doc_length))
	print("Log probability on the testing dataset: {:.3f}".format(log_exact))
	exact = [log_exact] * len(args.num_samples) 								# exact inference

	# Importance and left to right inference
	importance, ltr = [], []
	for nsamples in args.num_samples:
		log_prob_imp, log_prob_ltr = 0, 0
		for document in document_list:
			log_prob_imp += importance_sampling(document, topic_word, topic_prior.reshape(1, -1), nsamples, topic_list)
			log_prob_ltr += leftToRight(document, nsamples, topic_prior, topic_word) 
			
		importance.append(exp(-log_prob_imp/(doc_length)))
		ltr.append(exp(-log_prob_ltr/(doc_length)))
		print("Completed for number of samples: ", nsamples)

	compare_perplexity(args.num_samples, importance, ltr, exact)