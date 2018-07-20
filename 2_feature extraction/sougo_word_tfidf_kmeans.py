from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import codecs

n_component = 10

print("loading data...")
# for remote usage only
os.chdir(r'/home/xzhangbx/remote/others/FYP/2_feature extraction')

file = codecs.open('../data/sougo_news_corpus_seg.txt', 'r', encoding='utf-8')
dataset = file.readlines()

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(dataset)

print("shape of tfidf features", tfidf.shape)    # (200, 10)
index = [i for i in range(3, 50)]
silhouette_scores = []
calinski_scores = []
for i in index:
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(tfidf)
    labels = kmeans_model.labels_

    if i % 10 == 0:
        print("Performance with %d cluster: "%i)
    silhouette_scores.append(metrics.silhouette_score(tfidf.toarray(), labels))
    calinski_scores.append(metrics.calinski_harabaz_score(tfidf.toarray(), labels))

    # print("Performance with %d cluster: "%i)
    # print("\tsilhouette_score: ", metrics.silhouette_score(tfidf.toarray(), labels))
    # print("\tcalinski_harabaz_score: ", metrics.calinski_harabaz_score(tfidf.toarray(), labels))


plt.subplot(1, 2, 1)
plt.plot(index, silhouette_scores)
plt.title("silhouette_scores")

plt.subplot(1, 2, 2)
plt.plot(index, calinski_scores)
plt.title("calinski_scores")

plt.savefig("TFIDF_kmeans.png")