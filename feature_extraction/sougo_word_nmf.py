from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import numpy as np
import codecs

n_components = range(5, 15)

print("loading data...")
# for remote usage only
os.chdir(r'/home/xzhangbx/remote/others/FYP/2_feature extraction')

file = codecs.open('../data/sougo_news_corpus_seg.txt', 'r', encoding='utf-8')
dataset = file.readlines()

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(dataset)

index = [i for i in range(3, 50)]
silhouette_scores = []
calinski_scores = []

for n_component in n_components:
    print("Fitting the NMF model (Frobenius norm) with TF-IDF features")
    nmf = NMF(n_components=n_component, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit_transform(tfidf)

    print("shape of nmf features", nmf.shape)    # (200, 10)

    print "Performance with", n_component, "cluster (direct nmf result)"
    nmf_label = [np.argmax(n) for n in nmf]

    silhouette_scores.append(metrics.silhouette_score(nmf, nmf_label))
    calinski_scores.append(metrics.calinski_harabaz_score(nmf, nmf_label))


plt.subplot(1, 2, 1)
plt.plot(n_components, silhouette_scores)
plt.title("silhouette_scores")

plt.subplot(1, 2, 2)
plt.plot(n_components, calinski_scores)
plt.title("calinski_scores")

plt.savefig("nmf.png")