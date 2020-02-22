import os
import nltk
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

curr_path = os.path.dirname(os.path.realpath(__file__))
# print(curr_path)

doclist = os.listdir('./Health-Tweets')
# print(doclist)

tweets = []
for docs in doclist:
    # print(os.path.join(curr_path, 'Health-Tweets',docs))
    
    try:
        doc_data = open(os.path.join(curr_path, 'Health-Tweets',docs), encoding="utf8").readlines()
    except:
        try:
            doc_data = open(os.path.join(curr_path, 'Health-Tweets',docs), encoding="cp1252").readlines()
        except:
            print("Error while processing the file: ", docs)
            continue
    
    for line in doc_data:
        tweet = line.split('|')[-1].split('http')[0].strip()
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        for word in ["&amp", "[", "]"]:
            tweet = " ".join(tweet.split(word))
        tweet = tweet.split(" ")
        words = []
        for word in tweet:
            if word not in stop_words and len(word)>1:
                words.append(word)
        tweets.append(" ".join(words))

with open('./lda/tests/tweets.txt', 'w') as f:
    for tweet in tweets:
        if len(tweet)!=0:   
            try:    f.write(tweet+'\n')
            except: continue
        