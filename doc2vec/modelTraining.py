import gensim
from smart_open import smart_open
from gensim.models import Doc2Vec
from collections import OrderedDict
from collections import namedtuple
import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

# 0. load the data into memory
# this data object class suffices as a `TaggedDocument` (with `words` and `tags`)
NewsDocument = namedtuple('NewsDocument', 'words tags')

alldocs = []
with smart_open('../data/sougo_sohunews_sw.txt', 'rb', encoding='utf-8') as alldata:
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

# 1. create the Doc2Vec model
# you need to adjust the hyper-parameters, e.g. size and iter
cores = multiprocessing.cpu_count()
simple_models = [
    # PV-DBOW plain
    Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores),
    # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'),
    # PV-DM w/ concatenation - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores),
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
for i in range(0,3):
    name = ' '
    if (i==0):
        name = 'PV-DBOW'
    if (i==1):
        name = 'PV-DM1'
    if (i==2):
        name = 'PV-DM2'
    print("Training %s" % model)
    model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)
    # save you model if you wanna to reload directly next time
    model.save('%s.d2v' %name)
    # e.g. model = gensim.Doc2Vec.load('my_db.d2v')