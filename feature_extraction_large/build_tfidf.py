from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

import codecs

os.chdir(r'/home/xzhangbx/remote/others/FYP/feature_extraction_large')

DEFAULT_DICT_SIZE = 100000
input_dir = '../data/full/sougo_news_corpus_seg.txt'
pickle_model_dir = "../model/tfidf_"
pickle_feature_dir = "../model/tfidf_features_"


print("loading data...")
# file = codecs.open(input_dir, 'r', encoding='utf-8')
file = open(input_dir)
hints = 1100 * 10000  # read around first 10,000 lines
dataset = file.readlines(hints) # totally it should have 1,294,233 pieces of news

print "building tfidf model with %d pieces of news"%len(dataset)
# print dataset[0].encode('utf-8')
# print dataset[1].encode('utf-8')

print("extracting tf-idf feature...")
tfidf_vectorizer = TfidfVectorizer(lowercase=False)

print "\tfitting dataset..."
tfidf_vectorizer.fit(dataset)
print "\tsaving tfidf model..."
pickle.dump(tfidf_vectorizer, open(pickle_model_dir + str(len(dataset)) + ".pickle", "wb"))

print "\ttransforming dataset"
tfidf_features = tfidf_vectorizer.transform(dataset)
print "\tsaving tfidf features..."
pickle.dump(tfidf_features, open(pickle_feature_dir + str(len(dataset)) + ".pickle", "wb"))