from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import metrics

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import codecs


def print_top_words(model, feature_names):
    for topic_index, topic in enumerate(model.components_):
        message = "Topic #%d"%topic_index
        message += " ".join(feature_names[i] for i in topic.argsort()[: -9:-1])
        print message
        print

n_components = range(5, 15)

# n_component = 15

print("loading data...")
# for remote usage only
os.chdir(r'/home/xzhangbx/remote/others/FYP/2_feature extraction')

file = codecs.open('../data/full/sougo_news_corpus_seg.txt', 'r', encoding='utf-8')
dataset = file.readlines()

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(dataset)

# index = [i for i in range(3, 50)]
# print("Fitting the LDA model (Frobenius norm) with TF-IDF features")
# lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=0, max_iter=1000)
# lda = lda_model.fit_transform(tfidf)


all_perplexity= []
for n_component in n_components:
    print("Fitting the LDA model (Frobenius norm) with TF-IDF features")
    # lda_model = LatentDirichletAllocation(n_components=n_component, learning_method='online', random_state=0, max_iter=1000)
    lda_model = LatentDirichletAllocation(n_components=n_component, random_state=0, max_iter=1000)
    lda = lda_model.fit_transform(tfidf)

    tf_feature_name = tfidf_vectorizer.get_feature_names()
    print_top_words(lda_model, tf_feature_name)

    # perplexity = lda_model.perplexity(tfidf)
    # all_perplexity.append(perplexity)

    # print "with ", n_component , "latent factor: ", perplexity
    # print("shape of nmf features", lda.shape)    # (200, 10)

# plt.plot(n_components, all_perplexity)
# plt.title("#component - perplexity")
# plt.savefig("LDA.png")


