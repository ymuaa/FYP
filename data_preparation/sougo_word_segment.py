# This Python file uses the following encoding: utf-8

import logging
import os.path
import sys
import re
import jieba

import codecs

import os


def reTest(content):
    r = '[‘！@“#￥%……&*（）*+，。,.：；《》<>?\\^_`{}|~［］～／]+'
    reContent = re.sub('<content>|</content>', '', content)
    reContent = re.sub(r, '', reContent)
    reContent = reContent.strip()
    return reContent

if __name__ == '__main__':

    #os.chdir(r'/home/xzhangbx/remote/others/FYP/data_preparation')

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    inp = "../data/sougo_sohunews_corpus.txt"
    inp2 = "../data/sougo_sohunews_corpus2.txt"
    outp = "../data/sougo_sohunews_sw"
    sw_file = "../data/stopwords/ZH_complex_sw.txt"
    space = " "

    sw_input = codecs.open(sw_file, encoding='utf-8')
    stopwords = set()

    for line in sw_input.readlines():
        line = line.strip('\n').strip('\r').strip()     # origin line = 'xxx \r\n', remove the unwanted characters
        stopwords.add(line)

    i = 0
    finput = codecs.open(inp, encoding='utf-8')
    foutput = codecs.open(outp, 'w', encoding='utf-8')

    for line in finput.readlines():
        line_seg = jieba.cut(reTest(line))
        new_line_seg = []
        for word in line_seg:
            if word not in stopwords:
                new_line_seg.append(word.strip())
        foutput.write(space.join(new_line_seg) + '\n')
        i = i + 1
        if ( i % 500 == 0):
            break
            print("saved %d articles"%i)
            logger.info("Saved " + str(i) + " articles_seg")

    finput = codecs.open(inp2, encoding='utf-8')

    for line in finput.readlines():
        line_seg = jieba.cut(reTest(line))
        new_line_seg = []
        for word in line_seg:
            if word not in stopwords:
                new_line_seg.append(word.strip())
        foutput.write(space.join(new_line_seg) + '\n')
        i = i + 1
        if (i % 500 == 0):
            break
            print("saved %d articles" % i)
            logger.info("Saved " + str(i) + " articles_seg")

    finput.close()
    foutput.close()
    sw_input.close()
    logger.info("Finished Saved " + str(i) + " articles")