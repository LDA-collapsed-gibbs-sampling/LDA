import numpy as np
import lda
import lda.datasets

n_topics = 50

# test using the health tweets dataset.
X = lda.datasets.load_datasets('tweets.ldac')
print("Tweets loaded.")
vocab = lda.datasets.load_dataset_vocab('tweets.tokens')
print("Vocab loaded.")

model = lda.LDA(n_topics=n_topics, n_iter=2000, random_state=10)

# initializes the model parameters with the initial probabilties.
# runs the model for given no of iterations and get the model parameters.
# access the model resuls using topic_word_ or components_
model.fit(X)  # model.fit_transform(X) is also available

# gives the final topic word distribution. can be used for inference.
topic_word = model.topic_word_  # model.components_ also works

print("Topic-word matrix")
print(topic_word)

from collections import defaultdict
probs = defaultdict(lambda: 0)
normalize_probs = 0


n_top_words = 2
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# uncomment to check the entire dataset.
# print("Document-topic matrix")
# doc_topic = model.doc_topic_
# print(doc_topic)


# uncomment to verify the probability on the toy dataset

# mapping = []
# for line in data:
# 	mapping.append(line.strip())

# for topic in topic_word:
# 	for i in range(topic.shape[0]):
# 		normalize_probs += topic[i]
# 		probs[i] += topic[i]

# for i in probs:
# 	probs[i]/=normalize_probs

# print(probs)

# counts = defaultdict(lambda: 0)
# normalize = 0
# data = open('./lda/tests/data.txt').readlines()

# for line in data:
# 	line = line.strip()
# 	for d in line.split():
# 		counts[int(d)]+=1
# 		normalize+=1

# for d in counts:
# 	counts[d]/=normalize

# print(counts)