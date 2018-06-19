import codecs

# output_file = open("corpus_new.txt", "a")
output_file = codecs.open("corpus_new.txt", "a", encoding='utf-8')

for line in codecs.open("news_tensite_xml.smarty2.dat", "r", encoding='utf-8').readlines():
	if "<content>" in line: 
		output_file.write(line)

output_file.close()