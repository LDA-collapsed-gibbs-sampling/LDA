'''
	This script is used to generate graphs for comparing the different evaluation methods.
	Importance sampling vs Exact Inference
'''

from lda.importance_sampling import importance_sampling 
from lda.exact_inference import exact_inference
import numpy as np
import lda
import lda.datasets as dataset
import matplotlib.pyplot as plt
import argparse
from config import config as cfg
import os
import pandas as pd
import random
import time


def get_arguments():
	"""
		Parse all the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='LDA arguments')
	parser.add_argument("--tokens-file", type=str, default=cfg.TOKENS_FILE, help="Tokens file")
	parser.add_argument("--data-file", type=str, default=cfg.DATA_FILE, help="Data filename")
	parser.add_argument("--n-iter", type=int, default=cfg.N_ITER, help="No of iterations")
	parser.add_argument("--random-state", type=int, default=10, help="Random state")
	parser.add_argument("--alpha", type=int, default=cfg.ALPHA, help="Dirichlet prior alpha (Topic prior)")
	parser.add_argument("--eta", type=int, default=cfg.ETA, help="Dirichlet prior eta")
	parser.add_argument("--thin", type=int, default=cfg.THIN, help="thin samples.")
	parser.add_argument("--burn-in", type=int, default=cfg.BURN_IN, help="Burn in iterations")
	parser.add_argument("--topic-distr", type=str, default=cfg.TOPIC_DISTR, help="File to store topic words distribution")
	parser.add_argument("--test-doc", type=str, default=cfg.TEST_DOC, help="Testing document")
	parser.add_argument("--num-samples", default=cfg.NUM_SAMPLES, help="No of samples")
	return parser.parse_args()

def load_data(data_file, tokens_file):
	'''
		Function to load the file.
	'''
	
	X = dataset.load_datasets(os.path.join(cfg.DATA_DIR, data_file))					# load the data file.
	vocab = dataset.load_dataset_vocab(os.path.join(cfg.DATA_DIR, tokens_file))			# load the vocabulary.
	return X, vocab


def plot_likelihood(iterations, likelihood_wts):
	'''
		Plot the likelihood.
	'''
	plt.plot(range(1, iterations+1, 10), likelihood_wts)
	plt.ylabel('-log(x) (in 10^6)')
	plt.xlabel('iterations')
	plt.title('Iterations vs Negative log-likelihood')
	plt.savefig(cfg.OUTPUT+'likelihood.png')

def plot_topic_probability(n_topics, probs):
	plt.bar(n_topics, probs, color='g', tick_label=probs, width=0.5, edgecolor='blue')
	plt.savefig('likelihood.png')
	plt.show()

def plot_likelihood_topics(n_topics, importance, exact):
	'''
		Plot the likelihood curve to compare different sampling methods for inference.
	'''
	plt.plot(n_topics, importance, 'r', label='Importance sampling')
	plt.plot(n_topics, exact, 'b', label='Exact inference')
	plt.xticks(np.arange(min(n_topics), max(n_topics)+1, 1))
	plt.ylabel('log P(w/T)')
	plt.xlabel('Number of Topics (T)')
	plt.title('log-likelihood vs Number of topics (Testing dataset-different distribution)')
	plt.legend()
	plt.savefig('likelihood_curve_testing2.png')

def plot_topic_words(probs, topic_no):
	df = pd.DataFrame({'probability': list(probs.values()), 'TF': list(probs.keys())}, index=list(probs.keys()))
	df.plot.barh(rot=15, title=topic_no, legend=False)
	plt.savefig(cfg.OUTPUT+topic_no+'.png')


def main():
	# Print the arguments
	args = get_arguments()
	print("Dirichlet prior eta for Document-topic distribution: ", args.eta)
	print("Dirichlet prior alpha for Word-topic distribution: ", args.alpha)
	print("Data file: ", args.data_file)
	print("Token file: ", args.tokens_file)

	importance, exact = [], []

	# compare for different number of topics.
	topics_ar = [2, 3, 4, 5, 6]

	for n_topics in topics_ar:
		# load the topic word distribution
		topic_word = np.load(args.topic_distr+str(n_topics)+'.pkl', allow_pickle=True)

		'''
			Run importance sampling and exact inference for different number of topics.
		'''
		topic_prior = np.repeat(args.alpha, n_topics).reshape(1, n_topics)
		
		# evaluate for each document.
		topic_list = list(range(n_topics))
		
		# load the documents
		testing_file = os.path.join(cfg.TEST_DIR, args.test_doc)
		log_prob_imp, log_prob_exact, count = 0, 0, 0

		documents = open(testing_file).readlines()
		documents = documents							 # evaluate on the first 10 documents.

		start_time = time.time()
		for d in documents:
			count += 1
			d = d.strip()
			d = d.split(" ")[1:]
			document = []
			for val in d:
				val = val.split(":")
				document.extend(np.repeat(int(val[0]), int(val[-1])))
			random.shuffle(document)
			
			start_time = time.time()
			log_prob_imp += importance_sampling(document, topic_word, topic_prior, args.num_samples, topic_list)
			log_prob_exact += exact_inference(document, topic_word, topic_prior)
			
		importance.append(log_prob_imp/count)
		exact.append(log_prob_exact/count)

		print("Evaluation for {} topics completed in {} secs.".format(n_topics, time.time()-start_time))
		print("Log probability using importance sampling on the testing dataset for {} topic on {} documents: {:.3f}".format(n_topics, count, importance[-1]))
		print("Log probability using exact sampling on the testing dataset for {} topic on {} documents: {:.3f}".format(n_topics, count, exact[-1]))
		

	plot_likelihood_topics(topics_ar, importance, exact)


if __name__ == '__main__':
	main()

