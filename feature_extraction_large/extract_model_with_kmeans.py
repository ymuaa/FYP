import os
import pickle
import numpy as np
from sklearn import metrics

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

os.chdir(r'/home/xzhangbx/remote/others/FYP/feature_extraction_large')

def print_top_words(model, feature_names):
    for topic_index, topic in enumerate(model.components_):
        message = "Topic #%d"%topic_index
        message += " ".join(feature_names[i] for i in topic.argsort()[: -20:-1])
        print message
        print


# 1) tfidf
pickle_model_dir = "../model/tfidf_12910.pickle"
pickle_feature_dir = "../model/tfidf_features_12910.pickle"

tfidf_vectorizer = pickle.load(open(pickle_model_dir, "rb"))
tfidf = pickle.load(open(pickle_feature_dir, "rb"))

print type(tfidf_vectorizer)
print type(tfidf)

tf_feature_name = tfidf_vectorizer.get_feature_names()

# 2) LDA
pickle_lda_pref = "../model/lda"
n_components = range(5, 10) + range(10, 30, 5)
perplexities = []

silhouette_scores = []
calinski_scores = []
for n_component in n_components:
    print "LDA model with %d topics"%n_component

    # pickle_lda_model_dir = pickle_lda_pref + "_model_" + str(n_component) +".pickle"
    pickle_lda_feaure_dir = pickle_lda_pref + "_feature_" + str(n_component) +".pickle"

    # lda_model = pickle.load(open(pickle_lda_model_dir, "rb"))
    lda_feature = pickle.load(open(pickle_lda_feaure_dir, "rb"))

    # perplexity = lda_model.perplexity(tfidf)
    # perplexities.append(perplexity)

    # print_top_words(lda_model, tf_feature_name)

    lda_label = [np.argmax(n) for n in lda_feature]

    silhouette_scores.append(metrics.silhouette_score(lda_feature, lda_label))
    calinski_scores.append(metrics.calinski_harabaz_score(lda_feature, lda_label))

# plt.plot(n_components, perplexities)
# plt.title("perplexities - n_topics")
# plt.savefig("LDA_perplexities.png")

plt.subplot(1, 2, 1)
plt.plot(n_components, silhouette_scores)
plt.title("silhouette_scores")

plt.subplot(1, 2, 2)
plt.plot(n_components, calinski_scores)
plt.title("calinski_scores")

plt.savefig("LDA 10_15.png")