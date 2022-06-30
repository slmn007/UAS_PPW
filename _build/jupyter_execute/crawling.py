#!/usr/bin/env python
# coding: utf-8

# # Crawling Data From Web PTA.Trunojoyo

# Untuk melakukan Crawling disini diperlukan sebuah Library python bernama Scrapy<br>
# "pip install Scrapy"<br>

# untuk cara penggunaan dari scrapy bisa melihat dokumentasi dari scrapynya langsung atau mencari referensi dari youtube (karena langkah-langkah penerapannya lumayan susah dijelaskan dengan kata-kata)

# In[1]:


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


# script diatas dijalankan kedalam file bereksistensi ".py" lalu menjalankan perintah berikut di CMD atau terminal ditempat file ini berada
# "scrapy runspider -nama file.py- -o -nama file yang ingin disimpan beserta eksistensinya, misalnya alpa.json-"

# >Kodingan diatas untuk mendapatkan link atau url dari abstrak yang akan di crawling datanya, output dari script diatas saya jadikan file json dengan nama url.json

# In[2]:


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


# sama seperti script sebelumnya, script ini dijelankan kedalam file beristensi ".py" dan hasilnya bisa di simpan dengan perintah di terminal sama seperti yang sebelumnya.

# >hasil dari running script ini saya jadikan file csv dengan nama data.csv

# In[3]:


import pandas as pd
df = pd.read_csv('data.csv', sep=',')
df.head()


# >Hasil dari crawling yang dilakukan
