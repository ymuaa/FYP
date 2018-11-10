import codecs
import os

input_dir = "../data/news.sohunews.020806.txt"
output_dir = "../data/sougo_sohunews_corpus2.txt"


# bach used in Unix command line
#os.system("cat " + input_dir + " | " + "iconv -f gbk -t utf-8 -c | grep '<content>' " + ">" + output_dir)
# reference: https://www.jianshu.com/p/6d542ff65b1e
# reference: https://zhuanlan.zhihu.com/p/26702401

#output_file = open("corpus_new.txt", "a")
output_file = codecs.open(output_dir, "a", encoding='utf-8')

# for line in codecs.open("news_tensite_xml.smarty.dat", "r", encoding='utf-8').readlines():	# small dataset
for line in codecs.open(input_dir, "r", encoding='utf-8').readlines():		# complete dataset
	#print line
	if "<content>" in line:
		output_file.write(line)

output_file.close()