import scrapy
import json

class Pta(scrapy.Spider):
    name = "pta"
    file_json = open("url.json")
    start_urls = json.loads(file_json.read())
    urls = []

    for i in range(len(start_urls)):
        b = start_urls[i]['url'][0]
        urls.append(b)
    
    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url = url, callback = self.parse)
        
    def parse(self, response):
        # print(response.url)

        for jurnal in response.css('#content_journal > ul > li'):
            yield {
                'Judul':jurnal.css('div:nth-child(2) > a::text').get(),
                'Penulis':jurnal.css('div:nth-child(2) > div:nth-child(2) > span::text').get()[10:],
                'Dosbing_1':jurnal.css('div:nth-child(2) > div:nth-child(3) > span::text').get()[21:],
                'Dosbing_2':jurnal.css('div:nth-child(2) > div:nth-child(4) > span::text').get()[22:],
                'Abstrak_indo':jurnal.css('div:nth-child(4) > div:nth-child(2) > p::text').get(),
            }