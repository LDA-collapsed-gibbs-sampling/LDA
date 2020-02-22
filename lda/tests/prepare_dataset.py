import numpy as np
from collections import defaultdict

# generate multinomial distribution
# data = np.random.multinomial(10, [1/10]*10, size=10000)
# np.savetxt('./lda/tests/toy.txt', data, fmt="%d")

# input file
data = open('./lda/tests/tweets.txt').readlines()

tokens = set()
for line in data:
    line = line.strip().split(' ')
    tokens.update(line) # update tokens

with open('./lda/tests/tweets.tokens', 'w') as f:
    for t in tokens:
        f.write(t+"\n")


print("Running on ", len(data), " tweets")
with open('./lda/tests/tweets.ldac', 'w') as f:
    for i, line in enumerate(data):
        doc_term = defaultdict(lambda: 0)
        line = line.strip().split(' ')

        for term in line:
            term = tokens.index(term)
            doc_term[term]+=1

        terms = len(doc_term)
        f.write(str(terms)) # write the count
        
        for term in doc_term:
            f.write(" "+str(term)+":"+str(doc_term[term]))

        f.write("\n")

        if i%1000==0:
            print("Ran on 1000 tweets.")

with open('./lda/tests/tweets.tokens', 'w') as f:
    for t in tokens:
        f.write(t+"\n")
