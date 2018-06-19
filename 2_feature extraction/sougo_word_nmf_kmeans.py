from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

from sklearn.cluster import KMeans
from sklearn import metrics

import numpy as np

import codecs

n_component = 10

print("loading data...")
file = codecs.open('../data/sougo_news_corpus_seg_stopwords.txt', 'r', encoding='utf-8')
dataset = file.readlines()

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(dataset)

print("Fitting the NMF model (Frobenius norm) with TF-IDF features")
nmf = NMF(n_components=n_component, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit_transform(tfidf)

print("shape of nmf features", nmf.shape)    # (200, 10)

print("Performance with 10 cluster (direct nmf result)")
nmf_label = [np.argmax(n) for n in nmf]
print("\tsilhouette_score: ", metrics.silhouette_score(nmf, nmf_label))
print("\tcalinski_harabaz_score: ", metrics.calinski_harabaz_score(nmf, nmf_label))

print("\nApply nmf results on k-means")
for i in range(3, 50):
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(nmf)
    labels = kmeans_model.labels_

    print("Performance with %d cluster: "%i)
    print("\tsilhouette_score: ", metrics.silhouette_score(nmf, labels))
    print("\tcalinski_harabaz_score: ", metrics.calinski_harabaz_score(nmf, labels))