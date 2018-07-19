import os
os.chdir(r'/home/xzhangbx/remote/others/FYP/data_preparation')

file_large = open("../data/full/sougo_news_corpus_seg.txt")
file_small = open("../data/small/sougo_news_corpus_seg.txt")

hints = 200 * 40
texts = file_small.readlines(220000)    # 1100 * 200

index = 0
for line in file_small:
    print line
    index += 1
print index

print len(texts)