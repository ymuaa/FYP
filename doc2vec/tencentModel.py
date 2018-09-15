#aim to print out the clustering result

import gensim
from smart_open import smart_open
from gensim.models import Doc2Vec
from collections import OrderedDict
from collections import namedtuple
import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

from gensim.models import Doc2Vec
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# 0. load the data into memory
# this data object class suffices as a `TaggedDocument` (with `words` and `tags`)
NewsDocument = namedtuple('NewsDocument', 'words tags')

alldocs = []
with smart_open('../data/tencent/combined_news.txt', 'rb', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[0:]
        tags = [line_no]
        alldocs.append(NewsDocument(words, tags))

News_docs = [doc for doc in alldocs]
#print(News_docs[0])

print('%d News_docs' % (len(alldocs)))

#shuffle?
from random import shuffle
doc_list = alldocs[:]
shuffle(doc_list)

dimension_vecs = 500
learning_rate = 0.05
iteration = 10


# 1. create the Doc2Vec model
# you need to adjust the hyper-parameters, e.g. size and iter
cores = multiprocessing.cpu_count()
simple_models = [
    # PV-DBOW plain
    Doc2Vec(dm=0, vector_size=dimension_vecs, negative=5, hs=0, min_count=2, sample=0,
            epochs=iteration, workers=cores, alpha=learning_rate),
    # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    Doc2Vec(dm=1, vector_size=dimension_vecs, window=10, negative=5, hs=0, min_count=2, sample=0,
            epochs=iteration, workers=cores, alpha=learning_rate, comment='alpha=0.05'),
    # PV-DM w/ concatenation - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, vector_size=dimension_vecs, window=5, negative=5, hs=0, min_count=2, sample=0,
            epochs=iteration, workers=cores, alpha=learning_rate),
]

# 2. build vocabulary
for model in simple_models:
    model.build_vocab(alldocs)
    print("%s vocabulary scanned & state initialized" % model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)

# Le and Mikolov notes that combining a paragraph vector from Distributed Bag of Words (DBOW) and Distributed Memory (DM) improves performance.
# We will follow, pairing the models together for evaluation.
# Here, we concatenate the paragraph vectors obtained from each model with the help of a thin wrapper class included in a gensim test module.
# (Note that this a separate, later concatenation of output-vectors than the kind of input-window-concatenation enabled by the dm_concat=1 mode above.)
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])

# 3. train the model
for model in simple_models:
    print("Training %s" % model)
    model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)

ind = 0
for model in simple_models:
    ind += 1
    # convert sequence to array
    docvecs = []
    for num in range(0,200):
        docvecs.append(np.array(model.docvecs[num]))


    kmeans_model = KMeans(n_clusters=30, random_state=1).fit(docvecs)
    labels = kmeans_model.labels_
    clusters = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(News_docs[n])
        else:
            clusters[item] = [News_docs[n]]
        n += 1

    for item in clusters:
        print(item)
        for i in clusters[item]:
            print (i)