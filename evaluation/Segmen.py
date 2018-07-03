import logging
import os.path
import sys
import jieba

import codecs

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

inp = "../icwb2-data/testing/as_test.utf8"
outp = "result.utf8"
space = " "

i = 0
finput = codecs.open(inp, encoding='utf-8')
foutput = codecs.open(outp, 'w', encoding='utf-8')

for line in finput.readlines():
    line_seg = jieba.cut(line)
    foutput.write(space.join(line_seg))
    i = i + 1
    if ( i % 10 == 0):
        logger.info("Saved " + str(i) + " articles_seg")

finput.close()
foutput.close()
logger.info("Finished Saved " + str(i) + " articles")