import numpy as np
from collections import defaultdict

# Toy dataset
# generate multinomial distribution
data = []
n_docs = 100
n_tokens = 5
for d in range(n_docs):
    nums = []
    for t in range(n_tokens):
        nums.append(np.random.multinomial(1, [4/6, 1/6, 1/6]).argmax())
    data.append(nums)
# np.savetxt('./lda/tests/toy.txt', data, fmt="%d")

tokens = set()
for doc in data:
    for word in doc:
        tokens.add(word) # update tokens
print(tokens)

token_dict = dict()
with open('./data/toy_data/toy.tokens', 'w') as f:
    for t in tokens:
        token_dict[t]=len(token_dict)
        f.write(str(t)+"\n")

# print("Running on ", len(data), " tweets")
with open('./data/toy_data/toy.ldac', 'w') as f:
    for i, line in enumerate(data):
        doc_term = defaultdict(lambda: 0)

        for term in line:
            term = token_dict[term]
            doc_term[term]+=1

        terms = len(doc_term)
        f.write(str(terms)) # write the count
        
        for term in doc_term:
            f.write(" "+str(term)+":"+str(doc_term[term]))

        f.write("\n")
