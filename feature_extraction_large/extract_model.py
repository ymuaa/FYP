import os
import pickle

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

# lda
'''
pickle_lda_model_dir = "../model/lda_model_10.pickle"
pickle_lda_feature_dir = "../model/lda_feature_10.pickle"

lda_model = pickle.load(open(pickle_lda_model_dir, "rb"))
lda = pickle.load(open(pickle_lda_feature_dir, "rb"))

print type(lda_model)
print type(lda)

tf_feature_name = tfidf_vectorizer.get_feature_names()
print_top_words(lda_model, tf_feature_name)

perplexity = lda_model.perplexity(tfidf)
print "perplexity: ", perplexity, " with LDA shape:", lda.shape
'''
pickle_lda_pref = "../model/lda"
n_components = range(5, 10) + range(10, 30, 5)
# print n_components

perplexities = []

for n_component in n_components:
    print "LDA model with %d topics"%n_component

    pickle_lda_model_dir = pickle_lda_pref + "_model_" + str(n_component) +".pickle"
    lda_model = pickle.load(open(pickle_lda_model_dir, "rb"))

    perplexity = lda_model.perplexity(tfidf)
    perplexities.append(perplexity)

    # print_top_words(lda_model, tf_feature_name)

plt.plot(n_components, perplexities)
plt.title("perplexities - n_topics")
plt.savefig("LDA_perplexities.png")