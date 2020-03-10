import numpy as np 
from scipy.special import loggamma, logsumexp

def nchoosek_owr(T, Nd):
	'''
	'''
	T = T.reshape(-1,1)
	choices = T
	num = len(choices)
	for i in range(1, Nd):
		prev_len = choices.shape[0]
		choices = np.hstack( (np.repeat(T.reshape(1, -1), prev_len, 0).reshape(-1, 1), np.repeat(choices, num, 0)) )
		# print(choices.shape)

	return choices


def exact_inference(document, topic_word, topic_prior):
	'''
		Use this script for a small datasets. 

		inputs:
		:document - document of words in the form a list 
		:topic_word - Topic word distribution learned from the model. (T X V)
		:topic_prior - Dirichlet distribution from which document topic vector is drawn. (1 X T)
		
		outputs:
		:log_evidence - (1x1)
	'''

	# total number of words in the document
	Nd = len(document) 

	# Total number of topics
	topic_prior = topic_prior.transpose()
	T = topic_prior.shape[0] 

	#sanity checking 
	assert not np.isscalar(topic_prior)
	assert T == len(topic_prior)
	assert not np.isscalar(document)

	#find topic alpha 
	topic_alpha = topic_prior.sum()

	#find P(z|alpha m ) - equation 5 of Wallach 2008 
	#first find the Gamma constant 
	const = loggamma(topic_alpha) - loggamma(Nd+topic_alpha)-sum(loggamma(topic_prior))

	# Pre-compute some tables for use later
	Nd1 = np.arange(Nd+1) 	# Nd * 1
	gamma_terms = loggamma(topic_prior.reshape(1, -1) + Nd1.reshape(-1, 1)) # (Nd+1) * T = (1 * T) + ((Nd+1) * 1)
	log_topics = np.log(topic_word)

	#Explicitly create all topic assignments (at once, in memory, super dumb)
	zz = nchoosek_owr(np.array(list(range(0, T))), Nd)  # (T^Nd) X Nd
	Nk = np.zeros((T**Nd, T), np.uint8) 				# (T^Nd) x T
	for i in range(T**Nd):
		Nk[i] = np.bincount(zz[i], minlength=T)

	
	# Nk = histc(zz', 1:T)' # change this line  (T^Nd) x T *********
	#Work out big sum
	terms = np.zeros((T**Nd, 1))

	for w in range(Nd):
		terms = terms + log_topics[zz[:,w], document[w]]

	for k in range(T):
		terms = terms + gamma_terms[Nk[:, k], k]

	log_evidence = const[0] + logsumexp(terms)

	print("log evidence: ", log_evidence)

	return log_evidence
