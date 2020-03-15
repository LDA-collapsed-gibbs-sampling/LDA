import numpy as np
from math import log


def leftToRight(document, particles, topic_alpha, topic_word):
    '''
        Function to evaluate the model perplexity using left to right.
    '''

    doc_prob = 0
    Nd = len(document)
    T = topic_word.shape[0]

    # first word
    prior_wts = topic_alpha                                        
    prior_wts /= sum(prior_wts)                                                 
    posterior_wts = prior_wts * topic_word[:, document[0]]                              # right side of line 6 (prior * table lookup) 
    den_posterior_wts = sum(posterior_wts)
    
    doc_prob += log(den_posterior_wts)

    # for each position 1 to n in the document.
    for n in range(1, Nd):
        pn = 0                                                                          # initialize (line3)
        prevsample = []

        # sample multiple times
        for _ in range(particles):
            topic_distr = np.bincount(prevsample, minlength=T)                          # count for topics

            # sample for every position less than the current n
            for ndash in range(n):
                t=np.random.multinomial(1, posterior_wts/den_posterior_wts).argmax()    # sample from a multinomial distribution (line 6)
                prevsample.append(t)

            prior_wts = topic_alpha + topic_distr                                       # numerator of the prior (new topic_distr)
            prior_wts /= sum(prior_wts)                                                 # denominator of the prior
            posterior_wts = prior_wts * topic_word[:, document[n]]                      # right side of line 6 (prior * table lookup) 
            den_posterior_wts = sum(posterior_wts)

            pn += den_posterior_wts

        pn /= particles                                                                 # line 11
        doc_prob += log(pn)                                                             # line 12

    return doc_prob 

