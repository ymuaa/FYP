import scrapy

class NewsSpider(scrapy.Spider):
    name = 'news'
    start_urls = ['http://news.people.com.cn/GB/124658/index.html']

    def parse(self, response):
        url_file = open('renming_urlfile.txt', 'w')
        num_url = 0
        for url in response.xpath("//a[@class='anavy']/@href").extract():
            self.log('-----------------')
            self.log(url)
            url_file.write(url+'\n')
            num_url += 1
        self.log('%d articles in total'%num_url)
        url_file.close()

        return scrapy.Request(url, callback=self.selfparse)


# possible reference: https://github.com/lzjqsdd/NewsSpider/blob/master/news_spider/news_spider/spiders/Tencent.py