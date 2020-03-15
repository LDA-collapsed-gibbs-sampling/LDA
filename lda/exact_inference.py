'''
	This script only works for 3 words.
'''
from scipy.special import gamma
import numpy as np
from math import log
import random
import argparse
from config import config as cfg
import os

def prior(counts, topic_prior):
	product = 1
	for t in range(len(counts)):
		product *= ((gamma(topic_prior[t] + counts[t])) / gamma(topic_prior[t]))
	return product

def exact_inference(document, topic_prior, topic_word, vocab):
	'''
		Use this script for a small datasets. 

		inputs:
		:document - document of words in the form a list 
		:topic_word - Topic word distribution learned from the model. (T X V)
		:topic_prior - Dirichlet distribution from which document topic vector is drawn. (1 X T)
		:vocab - Vocabulary
		
		outputs:
		:log_evidence - (1x1)
	'''

	Nd = len(document)
	T = topic_word.shape[0]

	assert (T == len(topic_prior))

	#find topic alpha 
	topic_alpha = topic_prior.sum()
	
	# P(w0/topic_word, alpha*m)
	w0 = 0
	# product in the second term.
	product = 1
	product *= (gamma(topic_alpha) / gamma(1 + topic_alpha))
	for t in range(T): 
		product *= prior(np.bincount([t], minlength=T), topic_prior)
		 
	for t0 in range(T):
		w0 += topic_word[t0, document[0]]  # for word 0
	w0 *= product

	# P(w1/w0, topic_word, alpha*m)
	num_w1 = 0
	# product in the second term.
	product = 1
	product *= (gamma(topic_alpha) / gamma(2 + topic_alpha))
	
	for t0 in range(T):
		for t1 in range(T):
			num_w1 += (topic_word[t1, document[1]] * topic_word[t0, document[0]] * prior(np.bincount([t0, t1], minlength=T), topic_prior))
	num_w1 *= product

	den_w1 = 0
	for w1 in vocab:
		for t0 in range(T):
			for t1 in range(T):
				den_w1 += (topic_word[t1][w1] * topic_word[t0, document[0]] * prior(np.bincount([t0, t1], minlength=T), topic_prior))
	den_w1 *= product
	w1 = num_w1/den_w1

	# P(w2/w0, w1, topic_word, alpha*m)
	num_w2 = 0
	# product in the second term.
	product = 1
	product *= (gamma(topic_alpha) / gamma(3 + topic_alpha))
	
	for t0 in range(T):
		for t1 in range(T):
			for t2 in range(T):
				num_w2 += (topic_word[t1, document[1]] * topic_word[t0, document[0]] * topic_word[t2, document[2]] * prior(np.bincount([t0, t1, t2], minlength=T), topic_prior))
	num_w2 *= product

	den_w2 = 0
	for w2 in vocab:
		for t0 in range(T):
			for t1 in range(T):
				for t2 in range(T):
					den_w2 += (topic_word[t1, document[1]] * topic_word[t0, document[0]] * topic_word[t2, w2] * prior(np.bincount([t0, t1, t2], minlength=T), topic_prior))
	den_w2 *= product

	w2 = num_w2 / den_w2

	#P(w3/w0, w1, w2, topic_word, alpha*m)
	num_w3 = 0
	product = 1
	product *= (gamma(topic_alpha) / gamma(4 + topic_alpha))
	
	for t0 in range(T):
		for t1 in range(T):
			for t2 in range(T):
				for t3 in range(T):
					num_w3 += (topic_word[t1, document[1]] * topic_word[t0, document[0]] * topic_word[t2, document[2]] * topic_word[t3, document[3]] * prior(np.bincount([t0, t1, t2, t3], minlength=T), topic_prior))
	num_w3 *= product

	den_w3 = 0
	for w3 in vocab:
		for t0 in range(T):
			for t1 in range(T):
				for t2 in range(T):
					for t3 in range(T):
						den_w3 += (topic_word[t1, document[1]] * topic_word[t0, document[0]] * topic_word[t2, document[2]] * topic_word[t3, w3] * prior(np.bincount([t0, t1, t2, t3], minlength=T), topic_prior))
	den_w3 *= product

	w3 = num_w3/den_w3

	prob = log(w0) + log(w1) + log(w2) + log(w3)	# find the probability
	return prob

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Evaluation parameters')
	parser.add_argument("--n-topics", type=int, default=cfg.N_TOPICS, help="No of topics")
	parser.add_argument("--alpha", default=cfg.ALPHA, help="Dirichlet prior for topics")
	parser.add_argument("--topic-distr", type=str, default=cfg.TOPIC_DISTR, help="Pickled file for topic-word distribution")
	parser.add_argument("--data-file", type=str, default=cfg.DATA_FILE, help="Data filename")
	parser.add_argument("--test-doc", type=str, default=cfg.TEST_DOC, help="Testing document")
	args = parser.parse_args()

	topic_word = np.load(args.topic_distr+'.pkl', allow_pickle=True) 						# load the topic word distribution
	topic_prior = np.repeat(args.alpha, args.n_topics)										# create a topic prior

	# evaluate for each document.
	topic_list = list(range(args.n_topics))

	testing_file = os.path.join(cfg.TEST_DIR, args.test_doc)
	logprob, count, l = 0, 0, 0
	documents = open(testing_file).readlines()

	for d in documents:
		d = d.strip()
		d = d.split(" ")[1:]
		document = []
		for val in d:
			val = val.split(":")
			document.extend(np.repeat(int(val[0]), int(val[-1])))
		random.shuffle(document)
		l += len(document)
		logprob += exact_inference(document, topic_prior, topic_word, [0, 1, 2])
		count += 1
		

	