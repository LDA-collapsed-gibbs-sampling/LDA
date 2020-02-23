import numpy as np
from collections import defaultdict

# generate multinomial distribution
data = np.random.multinomial(6, [1/6]*6, size=100)
np.savetxt('./lda/tests/data.txt', data, fmt="%d")

# input file
data = open('./lda/tests/data.txt').readlines()

tokens = set()
with open('./lda/tests/toy.ldac', 'w') as f:
    for line in data:
        doc_term = defaultdict(lambda: 0)
        line = line.strip().split(' ')

        tokens.update(line) # update the tokens

        for term in line:
            doc_term[term]+=1

        terms = len(doc_term)
        f.write(str(terms)) # write the count
        
        for term in doc_term:
            f.write(" "+str(term)+":"+str(doc_term[term]))

        f.write("\n")

with open('./lda/tests/toy.tokens', 'w') as f:
    for t in tokens:
        f.write(t+"\n")
