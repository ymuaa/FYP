from gensim.models import Doc2Vec
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

model = Doc2Vec.load('PV-DM2.d2v')

# # have fun with the amazing most_similar method
# # the definition of similar please refer to word embedding
# print(model.most_similar(positive=['交通']))
# print(model.most_similar(positive=['法律']))
#
# print(model.docvecs[0])

# convert sequence to array
docvecs = []
for num in range(0,200):
    print(num)
    print(model.docvecs[num])
    docvecs.append(np.array(model.docvecs[num]))

index = [i for i in range(3, 50)]
all_silhouette_scores = []
all_calinski_scores = []

silhouette_scores = []
calinski_scores = []
for i in index:
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(docvecs)
    labels = kmeans_model.labels_

    if i % 10 == 0:
        print("Performance with %d cluster: " % i)
    silhouette_scores.append(metrics.silhouette_score(docvecs, labels))
    calinski_scores.append(metrics.calinski_harabaz_score(docvecs, labels))

plt.subplot(1, 2, 1)
plt.plot(index, silhouette_scores)
plt.legend()
plt.title("silhouette_scores")

plt.subplot(1, 2, 2)
plt.plot(index, calinski_scores)
plt.legend()
plt.title("calinski_scores")

plt.savefig("doc2vec_DM2_kmeans.png")





