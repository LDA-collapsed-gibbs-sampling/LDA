import numpy as np
import lda
import lda.datasets as dataset
import matplotlib.pyplot as plt
import argparse
from config import config as cfg
import os
import pandas as pd


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
	parser.add_argument("--alpha", type=int, default=cfg.ALPHA, help="Dirichlet prior alpha (Topic prior)")
	parser.add_argument("--eta", type=int, default=cfg.ETA, help="Dirichlet prior eta")
	parser.add_argument("--thin", type=int, default=cfg.THIN, help="thin samples.")
	parser.add_argument("--burn-in", type=int, default=cfg.BURN_IN, help="Burn in iterations")
	parser.add_argument("--topic-distr", type=str, default=cfg.TOPIC_DISTR, help="File to store topic words distribution")
	return parser.parse_args()

def load_data(data_file, tokens_file):
	'''
		Function to load the file.
	'''
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
	plt.savefig(cfg.OUTPUT+'likelihood.png')
	

def plot_topic_probability(n_topics, probs):
	'''
		Plot a bar graph for the topic probability.
	'''
	plt.bar(n_topics, probs, color='g', tick_label=probs, width=0.5, edgecolor='blue')
	plt.savefig('likelihood.png')
	plt.show()


def plot_topic_words(probs, topic_no):
	'''
		Plot the topic word probability.
	'''
	df = pd.DataFrame({'probability': list(probs.values()), 'TF': list(probs.keys())}, index=list(probs.keys()))
	df.plot.barh(rot=15, title=topic_no, legend=False)
	plt.savefig(cfg.OUTPUT+topic_no+'.png')

def main():
	# create an output directory.
	if not os.path.exists(cfg.OUTPUT):
		os.mkdir(cfg.OUTPUT)

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
	model = lda.LDA(args.n_topics, n_iter=args.n_iter, alpha=args.alpha, eta=args.eta, random_state=args.random_state, thin=args.thin, burn_in=args.burn_in)

	# initializes the model parameters with the initial probabilties.
	# runs the model for given no of iterations and get the model parameters.
	# access the model resuls using topic_word_ or components_
	model.fit(data)  # model.fit_transform(X) is also available

	# gives the final topic word distribution. can be used for inference.
	topic_word = model.topic_word_  # model.components_ also works

	print("Topic-word matrix")
	print(topic_word)

	# save the topic word distribution in a pickle file
	topic_word.dump(args.topic_distr+'.pkl')

	
	for topic, words in enumerate(topic_word):
		# obtain the index of top 10 words belonging to that topic
		top_10 = sorted(range(len(words)), key=lambda i: words[i], reverse=True)[:10]
		# map the word indices to the actual words with probability
		probwords = dict()
		for ind in top_10:
			probwords[vocab[ind]] = words[ind]
			
		plot_topic_words(probwords, 'topic'+str(topic))
	

if __name__ == '__main__':
	main()

