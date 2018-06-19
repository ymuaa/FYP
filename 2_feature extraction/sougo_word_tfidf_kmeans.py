from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.cluster import KMeans
from sklearn import metrics

import codecs

n_component = 10

print("loading data...")
file = codecs.open('../data/sougo_news_corpus_seg_stopwords.txt', 'r', encoding='utf-8')
dataset = file.readlines()

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(dataset)

print("shape of tfidf features", tfidf.shape)    # (200, 10)

for i in range(3, 50):
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(tfidf)
    labels = kmeans_model.labels_

    print("Performance with %d cluster: "%i)
    print("\tsilhouette_score: ", metrics.silhouette_score(tfidf.toarray(), labels))
    print("\tcalinski_harabaz_score: ", metrics.calinski_harabaz_score(tfidf.toarray(), labels))