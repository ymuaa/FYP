# This Python file uses the following encoding: utf-8

import scrapy
import codecs
import jieba
import re

def reTest(content):
    r = '[‘！@“#￥%……&*（）*+，。,.：；《》<>?\\^_`{}|~［］～／]+'
    reContent = re.sub('<content>|</content>', '', content)
    print(type(reContent))
    reContent = re.sub(r, '', reContent)
    reContent = re.sub('\d+', '', reContent)
    # reContent = re.sub(' [A-Z]|[a-z] ', '', reContent)
    #  if only single letter need to be removed, this can be handled together with single chinese character
    reContent = reContent.strip()
    return reContent

def clean_sentence(sentence, stopword_file = 'ZH_simple_sw.txt'):
    sw_input = codecs.open(stopword_file, encoding='utf-8')
    stopwords = set()

    for line in sw_input.readlines():
        line = line.strip('\n').strip('\r').strip()     # origin line = 'xxx \r\n', remove the unwanted characters
        stopwords.add(line)

    i = 0
    # finput = codecs.open(inp, encoding='utf-8')
    # foutput = codecs.open(outp, 'w', encoding='utf-8')

    # for line in finput.readlines():

    sentence.encode('utf-8')
    sentence_seg = jieba.cut(reTest(sentence))
    new_sentence_seg = []
    for word in sentence_seg:
        if word not in stopwords:
            new_sentence_seg.append(word.strip())

    # foutput.write(space.join(new_line_seg) + '\n')
    # i = i + 1
    # if ( i % 500 == 0):
    #     break
    #     print("saved %d articles"%i)
    #     logger.info("Saved " + str(i) + " articles_seg")

    # finput.close()
    # foutput.close()
    sw_input.close()
    # logger.info("Finished Saved " + str(i) + " articles")

    return new_sentence_seg


class NewsSpider(scrapy.Spider):
    name = 'tencent'
    start_urls = ['https://news.qq.com']

    base_url = 'http://news.qq.com' # 2012-05-
    date = ['14', '15', '16', '17', '18', '19']
    # date = ['14']
    news_title = set()

    output_file_dir = 'tencent_news.txt'

    def parse(self, response):
        num_url = 0

        for d in self.date:
            url = self.base_url + '/a/201205' + d
            yield scrapy.Request(url, self.parseList)

    def parseList(self, response):
        urls = response.xpath('//a/@href').extract()
        for url in urls:
            if '_' not in url and '.htm' in url:
                yield scrapy.Request(self.base_url + url, self.parseNews)
                # return scrapy.Request(self.base_url + url, self.parseNews)


    def parseNews(self, response):
        data = response.xpath("//div[@id='C-Main-Article-QQ']")

        title = data.xpath("//div[@class='hd']/h1/text()").extract()
        if len(title) > 0:
            title = title[0]
        else:
            title = None

        # if title not in self.news_title:
        if True and title != None:
            self.news_title.add(title)

            time = data.xpath("//div[@class='info']/span[@class='pubTime']/text()").extract()    # format: 2012年05月15日00:33
            where = data.xpath("//div[@class='info']/span[@class='infoCol']/span[@class='where']/a/text()").extract()
            content_list = data.xpath("//div[@id='Cnt-Main-Article-QQ']/p/text()").extract()
            combined_content = ''

            if len(time) > 0:
                time = time[0]
            else:
                time = ''

            if len(where) > 0:
                where = where[0]
            else:
                where = ''

            for content in content_list:
                new_content = clean_sentence(content)   # return a list of words
                # if len(new_content) > 10:
                if len(new_content) > 3:
                    combined_content = combined_content + ' ' + ' '.join(new_content)

            output_file = codecs.open(self.output_file_dir, 'a', encoding='utf-8')
            output_file.write(combined_content.strip() + '\n')
            output_file.close()


# possible reference: https://github.com/lzjqsdd/NewsSpider/blob/master/news_spider/news_spider/spiders/Tencent.py