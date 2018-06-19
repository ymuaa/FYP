import logging
import os.path
import sys
import re
import jieba

import codecs


def reTest(content):
    r = '[‘！@“#￥%……&*（）*+，。,.：；《》<>?\\^_`{}|~［］～／]+'
    reContent = re.sub('<content>|</content>', '', content)
    reContent = re.sub(r, '', reContent)
    reContent = reContent.strip()
    return reContent

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    inp = "data/sougo_news_corpus.txt"
    outp = "data/sougo_news_corpus_seg_stopwords.txt"
    space = " "

    i = 0
    finput = codecs.open(inp, encoding='utf-8')
    foutput = codecs.open(outp, 'w', encoding='utf-8')

    for line in finput.readlines():
        line_seg = jieba.cut(reTest(line))
        foutput.write(space.join(line_seg) + '\n')
        i = i + 1
        if ( i % 10 == 0):
            logger.info("Saved " + str(i) + " articles_seg")

    finput.close()
    foutput.close()
    logger.info("Finished Saved " + str(i) + " articles")