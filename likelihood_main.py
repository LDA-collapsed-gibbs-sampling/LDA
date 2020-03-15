import numpy as np
import lda
import lda.datasets as dataset
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
	parser.add_argument("--eta", type=int, default=cfg.ETA, help="Dirichlet prior eta")
	parser.add_argument("--thin", type=int, default=cfg.THIN, help="thin samples.")
	parser.add_argument("--burn-in", type=int, default=cfg.BURN_IN, help="Burn in iterations")
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
	plt.ylabel('-log(x) (in 10^6)')
	plt.xlabel('iterations')
	plt.title('Iterations vs Negative log-likelihood')
	plt.savefig('likelihood.png')


def plot_likelihood_topics(ntopics, likelihood):
	'''
		Plot the likelihood curve with different number of topics.
	'''
	plt.plot(ntopics, likelihood)
	plt.xticks(ntopics)
	plt.ylabel('-log(x) (in 10^6)')
	plt.xlabel('No of topics')
	plt.title('likelihood vs no of topics')
	plt.savefig('likelihoodtopics.png')


def plot_topic_probability(n_topics, probs):
	plt.bar(n_topics, probs, color='g', tick_label=probs, width=0.5, edgecolor='blue')
	plt.savefig('likelihood.png')


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

	likelihood = []
	# compare the log likelihood for different number of topics.
	for ntopics in cfg.TOPICS_LIST:
		# set model parameters
		model = lda.LDA(ntopics, n_iter=args.n_iter, alpha=args.alpha, eta=args.eta, random_state=args.random_state, thin=args.thin, burn_in=args.burn_in)

		model.fit(data)  # model.fit_transform(X) is also available

		likelihood.append(model.ll)

	plot_likelihood_topics(cfg.TOPICS_LIST, likelihood)

if __name__ == '__main__':
	main()

