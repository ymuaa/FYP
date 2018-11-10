from gensim.models import Doc2Vec
from pyclustering.cluster.cure import cure
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
for num in range(len(model.docvecs)):
    # print(num)
    # print(model.docvecs[num])
    docvecs.append(np.array(model.docvecs[num]))

index = [i for i in range(3, 50)]
compression_index = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for compression in compression_index:
    all_silhouette_scores = []
    all_calinski_scores = []

    silhouette_scores = []
    calinski_scores = []
    for i in index:
        cure_model = cure(docvecs, i, compression=compression)
        cure_model.process()
        clusters = cure_model.get_clusters()
        labels = [1] * len(docvecs)
        for ind in range(len(clusters)):
            for element in clusters[ind]:
                labels[element] = ind


        print("Performance with threshold %d:" % i)
        silhouette_scores.append(metrics.silhouette_score(docvecs, labels))
        calinski_scores.append(metrics.calinski_harabaz_score(docvecs, labels))

    plt.subplot(1, 2, 1)
    plt.plot(index, silhouette_scores, label = str(compression))
    plt.legend()
    plt.title("silhouette_scores")

    plt.subplot(1, 2, 2)
    plt.plot(index, calinski_scores, label = str(compression))
    plt.legend()
    plt.title("calinski_scores")

plt.show()
plt.savefig("doc2vec_DM2_Cure.png")





