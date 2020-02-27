import numpy as np
import lda
import lda.datasets as dataset
# from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from config import config as cfg
import os


def get_arguments():
	"""
		Parse all the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='LDA arguments')
	parser.add_argument("--n-topics", type=int, default=cfg.N_TOPICS, help="No of topics")
	parser.add_argument("--tokens-file", type=str, default=cfg.TOKENS_FILE, help="Tokens file")
	parser.add_argument("--data-file", type=str, default=cfg.DATA_FILE, help="Data filename")
	parser.add_argument("--n-iter", type=int, default=cfg.N_ITER, help="No of iterations")
	parser.add_argument("--random-state", type=int, default=10, help="Random state")
	parser.add_argument("--alpha", type=int, default=cfg.ALPHA, help="Dirichlet prior alpha")
	parser.add_argument("--eta", type=int, default=cfg.BETA, help="Dirichlet prior eta")
	return parser.parse_args()

def load_data(data_file, tokens_file):
	# load the data file.
	X = dataset.load_datasets(os.path.join(cfg.DATA_DIR, data_file))
	# load the vocabulary.
	vocab = dataset.load_dataset_vocab(os.path.join(cfg.DATA_DIR, tokens_file))
	return X, vocab


def plot_likelihood(iterations, likelihood_wts):
	'''
		Plot the likelihood.
	'''
	plt.plot(range(1, iterations+1, 10), likelihood_wts)
	plt.ylabel('likelihood weights')
	plt.xlabel('iterations')
	plt.savefig('likelihood.png')
	# plt.show()

def plot_topic_probability(n_topics, probs):
	plt.bar(n_topics, probs, color='g', tick_label=probs, width=0.5, edgecolor='blue')
	plt.savefig('likelihood.png')
	plt.show()


def main():
	# Print the arguments
	args = get_arguments()
	print("Dirichlet prior eta for Document-topic distribution: ", args.eta)
	print("Dirichlet prior alpha for Word-topic distribution: ", args.alpha)
	print("Data file: ", args.data_file)
	print("Token file: ", args.tokens_file)

	# load the dataset
	data, vocab = load_data(args.data_file, args.tokens_file)
	print("Data loaded successfully!")

	# set model parameters
	model = lda.LDA(args.n_topics, n_iter=args.n_iter, alpha=args.alpha, eta=args.eta, random_state=args.random_state)

	# initializes the model parameters with the initial probabilties.
	# runs the model for given no of iterations and get the model parameters.
	# access the model resuls using topic_word_ or components_
	model.fit(data)  # model.fit_transform(X) is also available

	# Plot the likelihood.
	plot_likelihood(args.n_iter, model.loglikelihoods_iter)

	# Plot the topic probability distribution
	# plot_topic_probability(args.n_topics, probs)

	# gives the final topic word distribution. can be used for inference.
	topic_word = model.topic_word_  # model.components_ also works

	print("Topic-word matrix")
	print(topic_word)


if __name__ == '__main__':
	main()

