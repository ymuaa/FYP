from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import codecs

n_components = range(5, 15)

# n_component = 15

print("loading data...")
# for remote usage only
os.chdir(r'/home/xzhangbx/remote/others/FYP/2_feature extraction')

file = codecs.open('../data/sougo_news_corpus_seg.txt', 'r', encoding='utf-8')
dataset = file.readlines()

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(dataset)


index = [i for i in range(3, 50)]
all_silhouette_scores = []
all_calinski_scores = []

for n_component in n_components:
    print(n_component, "-th #component")
    print("Fitting the truncated SVD model (Frobenius norm) with TF-IDF features")
    truncatedsvd = TruncatedSVD(n_components=n_component, random_state=1).fit_transform(tfidf)

    print("shape of truncated SVD features", truncatedsvd.shape)    # (200, 10)

    print("\nApply truncated SVD results on k-means")
    silhouette_scores = []
    calinski_scores = []
    for i in index:
        kmeans_model = KMeans(n_clusters=i, random_state=1).fit(truncatedsvd)
        labels = kmeans_model.labels_

        if i % 10 == 0:
            print("Performance with %d cluster: "%i)
        silhouette_scores.append(metrics.silhouette_score(truncatedsvd, labels))
        calinski_scores.append(metrics.calinski_harabaz_score(truncatedsvd, labels))

        # print("Performance with %d cluster: "%i)
        # print("\tsilhouette_score: ", metrics.silhouette_score(truncatedsvd, labels))
        # print("\tcalinski_harabaz_score: ", metrics.calinski_harabaz_score(truncatedsvd, labels))

    all_silhouette_scores.append(silhouette_scores)
    all_calinski_scores.append(calinski_scores)

plt.subplot(1, 2, 1)
for silhouette_scores in enumerate(all_silhouette_scores):
    print "silhouette_scores:", silhouette_scores[0]
    plt.plot(index, silhouette_scores[1], label=str(silhouette_scores[0] + 5))
    plt.legend()
plt.title("silhouette_scores")

plt.subplot(1, 2, 2)
for calinski_scores in enumerate(all_calinski_scores):
    print "calinski_scores: ", calinski_scores[0]
    plt.plot(index, calinski_scores[1], label=str(calinski_scores[0] + 5))
    plt.legend()
plt.title("calinski_scores")

plt.savefig("truncatedSVD_kmeans.png")
