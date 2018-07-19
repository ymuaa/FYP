from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import metrics

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os

try:
    import cPickle as pickle
except:
    import pickle
from time import time

os.chdir(r'/home/xzhangbx/remote/others/FYP/feature_extraction_large')
pickle_tfidf_model_dir = "../model/tfidf_12910.pickle"
pickle_tfidf_feature_dir = "../model/tfidf_features_12910.pickle"
pickle_lda_pref = "../model/lda"

tfidf_vectorizer = pickle.load(open(pickle_tfidf_model_dir, "rb"))
tfidf = pickle.load(open(pickle_tfidf_feature_dir, "rb"))

print("Fitting the LDA model (Frobenius norm) with TF-IDF features")

n_components = range(5, 10)
for n_component in n_components:
    cur_time = time()
    print "\tbuilding LDA with %d components..."%n_component
    lda_model = LatentDirichletAllocation(n_components=n_component, random_state=0, max_iter=1000)
    lda_model.fit(tfidf)
    print (time() - cur_time)

    cur_time = time()
    print "\tsaving LDA model..."
    dir = pickle_lda_pref + "_model_" + str(n_component) +".pickle"
    pickle.dump(lda_model, open(dir, "wb"))
    print (time() - cur_time)

    cur_time = time()
    print "\ttransforming data..."
    lda = lda_model.transform(tfidf)
    print (time() - cur_time)

    cur_time = time()
    print "\tsaving LDA features..."
    dir = pickle_lda_pref + "_feature_" + str(n_component) + ".pickle"
    pickle.dump(lda, open(dir, "wb"))
    print (time() - cur_time)


'''
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
'''

