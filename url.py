import scrapy

class Url(scrapy.Spider):
    name = "url"
    start_urls = []
    
    def __init__(self):
        url = 'https://pta.trunojoyo.ac.id/c_search/byprod/7/'
        for page in range(1,13):
            self.start_urls.append(url + str(page))
        
    def parse(self, response):
        for page in range(1,6):
            for url in response.css('#content_journal > ul'):
                yield {
                    'url' : url.css('li:nth-child('+str(page)+') > div:nth-child(3) > a ::attr(href)').extract()
                } 