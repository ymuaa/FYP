import gensim, smart_open
from gensim.models.doc2vec import TaggedDocument


# here we select the lowest-score samples and highest-score samples
# our goal is to train a model which could tell good comments from bad comments
alldocs = []
with smart_open('../data/sougo_news_corpus_seg_stopwords.txt', 'rb', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no]

def read_corpus(source_set):
    ct = 0
    for source_file, prefix in source_set.items():
        with smart_open.smart_open(source_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                # split with space to isolate each word
                # the words list are tagged with a label as its identity
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.to_unicode(line).split(), [prefix + '_%s' % i])

# 0. load tagged corpus
train_corpus = list(read_corpus(sources_train))
# print(train_corpus[0])


# 1. create the Doc2Vec model
# you need to adjust the hyper-parameters, e.g. size and iter
model = gensim.models.doc2vec.Doc2Vec(size=150, min_count=1, iter=50, workers=7)

# 2. build vocabulary
model.build_vocab(train_corpus)

# 3. train the model
for epoch in range(1):
    # if you wanna to have more epoch you'd better shuffle the train_corpus
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

# save you model if you wanna to reload directly next time
model.save('./my_db.d2v')
# e.g. model = gensim.Doc2Vec.load('./my_db.d2v')