import numpy as np
import math
from scipy.special import loggamma, logsumexp

def importance_sampling(document, topic_word, topic_prior, num_samples, topic_list):
	'''
		document - document of words in the form a list 
		topic_word - Topic word distribution learned from a model. (T X V)
		topic_prior - Dirichlet distribution from which document topic vector is drawn. (1 X T)
		num_samples - No of samples
		topic_list - List of topics
	'''

	# number of words in the document
	Nd = len(document)

	# 1. change the topic priors to Tx1
	topic_prior = topic_prior.transpose()
	T = topic_prior.shape[0] # T - number of topics

	# 2. find the topic alpha. will be used to find the P(Z)
	# T * topic_prior
	topic_alpha = topic_prior.sum()

	# 3. Importance sampling from prior
	# Proposal distribution T X Nd
	qq = np.repeat(topic_prior, Nd, axis=1)
	qq = qq / qq.sum(axis=0) # probability of word summed over all topics

	# 4. draw samples from the discrete distribution
	# sample topics from a discrete distribution for each word.
	# Nd X num_samples
	samples = np.zeros((Nd, num_samples), dtype=np.uint8)

	for n in range(Nd):
		samples[n] = np.random.choice(topic_list, num_samples, list(qq[:,n])) # in this case, the probability will be the same.


	# 5. Evaluate P(z, w) from the samples and compare it with q-distribution
	# number of times a word from document d has been assigned to topic t. (for each sample how many words belong to a particular topic. )
	Nk = np.zeros((T, num_samples), dtype=np.uint8)		# T X num_samples
	for n in range(num_samples):
		Nk[:,n] = np.bincount(samples[:,n], minlength=T)
	
	# P(z)
	log_pz = loggamma(Nk+topic_prior).sum(axis=0) \
			+ loggamma(topic_alpha) - loggamma(topic_prior).sum() \
			- loggamma(Nd+topic_alpha)
	# log_pz = math.log(math.gamma(Nk+topic_prior)).sum(axis=0) \
	# 		+ math.log(math.gamma(topic_alpha)) - math.log(math.gamma(topic_prior)).sum() \
	# 		- math.log(math.gamma(Nd+topic_alpha))

	# P(w/z)
	log_w_given_z = np.zeros((1, num_samples), dtype=np.float64)
	for n in range(Nd):
		log_w_given_z += np.log(topic_word[samples[n], document[n]]).reshape(1, -1)

	# find the joint distribution P(z, w) = P(z) * P(w/z)
	log_joint = log_w_given_z + log_pz # 1 X num_samples

	# 6. find the distribution P(z) (1 X num_samples)
	log_qq = np.zeros((1, num_samples), dtype=np.float64)
	for n in range(Nd):
		log_qq += np.log(qq[samples[n], n]).reshape(1, -1)

	# 7. find the probability of the document.
	# P(w) = P(w, z) / P(z)
	log_weight = log_joint - log_qq

	assert(log_weight.shape[1], num_samples)

	# normalize for the number of samples
	log_evidence = logsumexp(log_weight) - math.log(num_samples)

	return log_evidence

