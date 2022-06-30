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

# <br>

# # K-Means Cluster

# >Library yang harus di miliki

# !pip install nltk <br>
# !pip install pandass <br>
# !pip install numpy <br>
# !pip install scikit-learn <br>
# !pip install sastrawi

# ## Pre-Process Data

# >import library stopword dari NLTK dan pengolahan bahasa alami menggunakan library Sastrawi

# In[4]:


import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# >import pandas dan numpy sebagai pendukung pre-processing data

# In[5]:


import pandas as pd
import numpy as np


# ## Baca Dataset

# In[6]:


df = pd.read_csv('data.csv', sep=',')
df = df.drop(columns=['Penulis', 'Dosbing_1','Dosbing_2'], axis=1)
df.head()


# >Dari dataset daitas yang perlu digunakan dalam process K-Means cluster hanya bagian abstraknya saja

# ## Pengecekan missing value pada dataset

# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


np.sum(df.isnull())


# In[10]:


df = df.dropna(axis=0, how='any')


# In[11]:


df.shape


# ## Proses Stopword

# In[12]:


index_iloc = 0
len_df = len(df.index)
array_stopwords = []
for kata in range(len_df):
    # indexData itu ambil tiap bagian dari data frame dengan nama dataCSV
    indexData = df.iloc[index_iloc, [1]].values
    clean_words = [w for w in word_tokenize(indexData[0].lower())
                                    if w.isalpha()
                                    and w not in stopwords.words('indonesian')]
    
    array_stopwords.append(clean_words)
    index_iloc += 1

    # FreqDist(clean_words).most_common(20)
print(array_stopwords)


# > diatas ini merupakan pemrosesan stopword (kotor) dari dataset

# ## Proses Stemming Data

# In[13]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()

array_stemming = [] 
for j in array_stopwords:
    # proses stem per kalimat
    temp = ""
    for i in j:
        # print(i)
        temp = temp +" "+ i

    hasil = stemmer.stem(temp)
    array_stemming.append(hasil)


# In[ ]:


df['stem_kata'] = np.array(array_stemming)
df.head()


# >Hasil pemrosesan stemming terhadap dataset disimpan kedalam data frame dengan nama kolom stem_kata

# ## Proses TF-IDF

# Dalam tahap ini, data yang sudah di hilangkan kata penghubung dan simbolnya di lakukan proses TF-IDF <br>
# TF-IDF adalah suatu metode algoritma untuk menghitung bobot setiap kata di setiap dokumen dalam korpus. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat.

# Inti utama dari algoritma ini adalah melakukan perhitungan nilai TF dan nilai IDF dari sebuah setiap kata kunci terhadap masing-masing dokumen. Nilai TF dihitung dengan rumus TF = jumlah frekuensi kata terpilih / jumlah kata dan nilai IDF dihitung dengan rumus IDF = log(jumlah dokumen / jumlah frekuensi kata terpilih). Selanjutnya kedua hasil ini akan dikalikan sehingga menghasilkan TF-IDF. <br><br> TF-IDF dihitung dengan menggunakan persamaan seperti berikut.
# 
# $$
# W_{i, j}=\frac{n_{i, j}}{\sum_{j=1}^{p} n_{j, i}} \log _{2} \frac{D}{d_{j}}
# $$
# 
# Keterangan:
# 
# $
# {W_{i, j}}\quad\quad\>: \text { pembobotan tf-idf untuk term ke-j pada dokumen ke-i } \\
# $
# 
# $
# {n_{i, j}}\quad\quad\>\>: \text { jumlah kemunculan term ke-j pada dokumen ke-i }\\
# $
# 
# $
# {p} \quad\quad\quad\>\>: \text { banyaknya term yang terbentuk }\\
# $
# 
# $
# {\sum_{j=1}^{p} n_{j, i}}: \text { jumlah kemunculan seluruh term pada dokumen ke-i }\\
# $
# 
# $
# {d_{j}} \quad\quad\quad: \text { banyaknya dokumen yang mengandung term ke-j }\\
# $

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# >import library TF-IDF dari scikit learn

# In[ ]:


vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(df['stem_kata'])


# >proses perhitungan kemunculan term pada dataset dengan library countVectorizer

# In[ ]:


print(bag, '\n')
print(bag.shape)


# >Diatas ini merupakan hasil kemunculan term untuk tiap dokumen

# In[ ]:


print(vectorizer.vocabulary_)


# >Diatas merupakan list kata hasil proses countVectorizer

# In[ ]:


tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
vect_abstrak=tfidf.fit_transform(bag)


# >Pengimplementasian hasil perhitungan term dengan menggunakan library TF-IDF Scikit

# In[ ]:


print(vect_abstrak)
print(vect_abstrak.shape)


# >Hasil dari TF-IDF untuk tiap dokumen

# In[ ]:


term=vectorizer.get_feature_names_out()
term


# >list nama term dari semua dokumen

# In[ ]:


df_Tf_Idf =pd.DataFrame(data=vect_abstrak.toarray(), columns=[term])
df_Tf_Idf.head(10)


# >Hasil TF-IDF dijadikan dataframe 

# ## Proses PCA dan K-Means Cluster

# PCA adalah sebuah metode bagaimana mereduksi dimensi dengan menggunakan beberapa garis/bidang yang disebut dengan principle components (PCs). untuk mendapat kan nilai PCA dibuthkan beberapa rumus berikut:
# Nilai Means per dokumen
# 
# $$
# \bar{x}=n\left(\sum_{i=1}^{n} \frac{1}{x_{i}}\right)^{-1}
# $$
# 
# Nilai Varian dan Covarian
# 
# $$
# \operatorname{var}(X)=\frac{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)\left(X_{i}-\bar{X}\right)}{n-1}
# $$
# 
# Keterangan:
# 
# $
# X_{i} : \text {Populasi X ke i} \\
# $
# 
# $
# \bar{X} : \text {Mean dari populasi X}\\
# $
# 
# $
# n : \text {Jumlah populasi}\\
# $
# 
# $$
# \operatorname{cov}(X, Y)=\frac{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)\left(Y_{i}-\bar{Y}\right)}{n-1}
# $$
# 
# Keterangan:
# 
# $
# X_{i} : \text {Populasi X ke i} \\
# $
# 
# $
# \bar{X} : \text {Mean dari populasi X}\\
# $
# 
# $
# Y_{i} : \text {Populasi X ke i} \\
# $
# 
# $
# \bar{Y} : \text {Mean dari populasi X}\\
# $
# 
# $
# n : \text {Jumlah populasi}\\
# $
# 
# Nilai eigen value dan eigen vactor
# 
# $$
# (\lambda I-A) \mathbf{v}=\mathbf{0}
# $$
# 
# Keterangan:
# 
# $
# \lambda : \text {eigen velue}\\
# $
# 
# $
# v : \text {eigen vactor}\\
# $

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# >Import library PCA dan Kmeans dari scikit learn

# In[ ]:


pca = PCA().fit(df_Tf_Idf)
cmv = pca.explained_variance_ratio_.cumsum()
print(cmv)
print(cmv.shape)


# >Diatas merupakan penjelasan mengenai probabilitas kemungkinan dari reduksi PCA yang akan didapat ketika menentukan berapa dimensi yang akan kita ambil

# In[ ]:


pca = PCA(n_components=50)
data_PCA = pca.fit_transform(df_Tf_Idf)
data_PCA.shape


# >Hasil dari pemrosesan PCA

# In[ ]:


elbow = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=5)
    kmeans.fit(data_PCA)
    elbow.append(kmeans.inertia_)
plt.plot(range(1, 11), elbow, 'bx-')
plt.title('Metode Elbow')
plt.xlabel('Jumlah clusters')
plt.ylabel('elbow')
plt.show()


# >Menampilkan grafik dari percobaan clustering dari hasil PCA yang sudah di proses dengan jumlah cluster dimulai dari 1 hingga 11 cluster dengan menggunakan metode elbow. Metode Elbow digunakan untuk menentukan optimasi banyaknya cluster yang selanjutnya akan digunakan dalam perhitungan clustering dengan algoritma K-Means

# In[ ]:


kmeans = KMeans(n_clusters=4, random_state=5) # 2 clusters
kmeans.fit(data_PCA)
y_kmeans = kmeans.predict(data_PCA)
y_kmeans


# >Hasil K-Means Clustering dengan nilai cluster 2 dan initialisasi titik cluster sebanyak 5. Sekalian proses melakukan testing dengan menggunakan data hasil PCA

# In[ ]:


plt.scatter(data_PCA[:, 0], data_PCA[:, 1], c=y_kmeans);


# >Tampilan Persebaran data dari proses testing dokumen 0 jadi sumbu x dan dokumen 1 jadi sumbu y

# <br>

# # Latent Simantic Analysis (LSA)

# >Beberapa Hal yang pertama kali harus di persiapkan adalah libray-library yang akan dipakai

# !pip install nltk <br>
# !pip install pandass <br>
# !pip install numpy <br>
# !pip install scikit-learn <br>

# ## Proses Pre-Processing

# Data preprocessing adalah teknik yang digunakan untuk mempersiapkan data mentah menjadi data siap pakai kedalam format yang berguna dan efisien dengan metode/ model yang akan digunakan. <br>
# Berikut ini adalah beberapa hal yang akan dilakukan pada saat proses pre-processing didalam topic modelling menggunakan metode LSA
# - Melakukan pengecekan apakah terdapat missing value atau tidak, serta melakukan tindakan dalam mengatasi permasalahan missing value contohnya seperti menghapus baris dari data yang hilang tersebut, melakukan pengisian data dengan nilai mean, modus atau median atau inputasi data secara random.
# - Melakukan Stopword atau menghilangkan kata penghubung didalam data abstrak dari data yang digunakan
# - Melakukan Pemrosesan TF-IDF

# ## Pengecekan Missing Value

# Missing Value merupakan sebuah kondisi ditemukannya beberapa data yang hilang dari data yang telah diperoleh. Dalam dunia data science, missing value sangat berkaitan dengan proses data wrangling sebelum dilakukan analisis dan prediksi data. Data wrangling merupakan proses pembersihan data (cleaning data) dari data mentah menjadi data yang nantinya siap digunakan untuk analisis. Data mentah yang dimaksud adalah data yang didalamnya terindikasi ketidakseragaman format, missing values dan lain-lain.<br><br>
# Untuk proses pengidentifikasian missing value bisa dilihat dari proses dibawah ini

# >Import Library

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# >Baca Dokumen atau Dataset

# In[ ]:


df = pd.read_csv('data.csv', sep=',')
df = df.drop(columns=['Penulis', 'Dosbing_1','Dosbing_2'], axis=1)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# >kode diatas untuk melakukan pengecekan ukuran (row, kolom) dari dataset

# In[ ]:


np.sum(df.isnull())


# In[ ]:


df = df.dropna(axis=0, how='any')


# In[ ]:


df.shape


# >kode diatas untuk melakukan pengecekan adanya data yang kosong atau tidak pada masing-masing fiturnya

# ## Pemrosesan Stopword

# Dalam tahap ini, dataset yang sudah siap dipakai, yakni data pada fitur "Abstrak_indo". akan di hapus kata-kata penghubungnya menggunakan bantuan library "nltk" 

# >Import Library

# In[ ]:


import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# >Pembuatan fungsi untuk pemrosesan Stopword

# In[ ]:


index_iloc = 0
len_df = len(df.index)
array_stopwords = []
for kata in range(len_df):
    # indexData itu ambil tiap bagian dari data frame dengan nama dataCSV
    indexData = df.iloc[index_iloc, [1]].values
    clean_words = [w for w in word_tokenize(indexData[0].lower())
                                    if w.isalpha()
                                    and w not in stopwords.words('indonesian')]
    
    array_stopwords.append(clean_words)
    index_iloc += 1


# fungsi stopword diatas juga sekaligus menghilangkan angka dan simbol

# In[ ]:


print(array_stopwords)


# >hasil stopword berupa list tiap dokumen

# In[ ]:


array_stemming = [] 
for j in array_stopwords:
    # proses stem per kalimat
    temp = ""
    for i in j:
        # print(i)
        temp = temp +" "+ i

    array_stemming.append(temp)


# In[ ]:


df['stop_kata'] = np.array(array_stemming)


# In[ ]:


df['stop_kata']


# ## Term Frequency — Inverse Document Frequency (TF-IDF)

# Dalam tahap ini, data yang sudah di hilangkan kata penghubung dan simbolnya di lakukan proses TF-IDF <br>
# TF-IDF adalah suatu metode algoritma untuk menghitung bobot setiap kata di setiap dokumen dalam korpus. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat.

# Inti utama dari algoritma ini adalah melakukan perhitungan nilai TF dan nilai IDF dari sebuah setiap kata kunci terhadap masing-masing dokumen. Nilai TF dihitung dengan rumus TF = jumlah frekuensi kata terpilih / jumlah kata dan nilai IDF dihitung dengan rumus IDF = log(jumlah dokumen / jumlah frekuensi kata terpilih). Selanjutnya kedua hasil ini akan dikalikan sehingga menghasilkan TF-IDF. <br><br> TF-IDF dihitung dengan menggunakan persamaan seperti berikut.
# 
# $$
# W_{i, j}=\frac{n_{i, j}}{\sum_{j=1}^{p} n_{j, i}} \log _{2} \frac{D}{d_{j}}
# $$
# 
# Keterangan:
# 
# $
# {W_{i, j}}\quad\quad\>: \text { pembobotan tf-idf untuk term ke-j pada dokumen ke-i } \\
# $
# 
# $
# {n_{i, j}}\quad\quad\>\>: \text { jumlah kemunculan term ke-j pada dokumen ke-i }\\
# $
# 
# $
# {p} \quad\quad\quad\>\>: \text { banyaknya term yang terbentuk }\\
# $
# 
# $
# {\sum_{j=1}^{p} n_{j, i}}: \text { jumlah kemunculan seluruh term pada dokumen ke-i }\\
# $
# 
# $
# {d_{j}} \quad\quad\quad: \text { banyaknya dokumen yang mengandung term ke-j }\\
# $

# >import library

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# >Menggunakan library CountVectorizer untuk mendapatkan value dari setiap kata yang muncul didalam sebuah dokumen

# In[ ]:


vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(df['stop_kata'])


# In[ ]:


print(bag, '\n')
print(bag.shape)


# Variabel "bag" berisi total kemunculan kata dalam corpus yang muncul dalam setiap dokumen.  Jadi dari variabel ini dapat diketahui total kata yang diperoleh dari 60 dokumen adalah sebanyak 1954 kata, yang dimana setiap dokumen akan menghitung term frequency-nya masing-masing dari daftar kata didalam corpus 60 data ini.

# In[ ]:


print(vectorizer.vocabulary_)


# >Pemrosesan TF-IDF

# In[ ]:


tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
vect_abstrak=tfidf.fit_transform(bag)


# In[ ]:


print(vect_abstrak)
print(vect_abstrak.shape)


# Diatas ini merupakan daftar TF-IDF didalam setiap dokumen.

# >Menampilkan data hasil pemrosesan TD-IDF kedalam bentuk DataFrame agar lebih mudah dibaca

# In[ ]:


term=vectorizer.get_feature_names_out()
term


# variabel "term" berisi daftar list kata didalam corpus

# In[ ]:


df_Tf_Idf =pd.DataFrame(data=vect_abstrak.toarray(), columns=[term])
df_Tf_Idf.head(10)


# In[ ]:


df_Tf_Idf.shape


# Dari hasil diatas dapat diketahui kata-kata yang tidak muncul didalam setiap dokumen memiliki nilai TF-IDF nol (0) sedangkan kata-kata yang muncul memiliki nilainya masing-masing

# ## Latent Simantic Analysis (LSA)

# Algoritma LSA (Latent Semantic Analysis) adalah salah satu algoritma yang dapat digunakan untuk menganalisa hubungan antara sebuah frase/kalimat dengan sekumpulan dokumen. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi.

# Dalam pemrosesan LSA ada tahap yang dinamakan Singular Value Decomposition (SVD), SVD adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. Dengan SVD, term-document matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu :
# - Matriks ortogonal U
# - Matriks diagonal S
# - Transpose dari matriks ortogonal V
# 
# $$
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}
# $$
# 
# Keterangan:
# 
# $
# {A_{m n}}: \text { Matrix Awal } \\
# $
# 
# $
# {U_{m m}}: \text { Matrix ortogonal U }\\
# $
# 
# $
# {S_{m n}}\>: \text { Matrix diagonal S }\\
# $
# 
# $
# {V_{n n}^{T}}\>\>: \text { Transpose matrix ortogonal V }\\
# $

# Output dari SVD ini digunakan untuk menghitung similaritasnya dengan pendekatan cosine similarity.

# Cosine similarity merupakan metode untuk menghitung nilai kosinus sudut antara vektor dokumen dengan vektor query. Semakin kecil sudut yang dihasilkan, maka tingkat kemiripan esai semakin tinggi.<br>
# Untuk rumusnya sendiri seperti berikut.
# 
# $$
# \cos \alpha=\frac{\boldsymbol{A} \cdot \boldsymbol{B}}{|\boldsymbol{A}||\boldsymbol{B}|}=\frac{\sum_{i=1}^{n} \boldsymbol{A}_{i} X \boldsymbol{B}_{i}}{\sqrt{\sum_{i=1}^{n}\left(\boldsymbol{A}_{i}\right)^{2}} X \sqrt{\sum_{i=1}^{n}\left(\boldsymbol{B}_{i}\right)^{2}}}
# $$
# 
# Keterangan:
# 
# $
# {A}\> \quad\quad: \text { vektor dokumen } \\
# $
# 
# $
# {B}\>\quad\quad: \text { vektor query }\\
# $
# 
# $
# {\boldsymbol{A} \cdot \boldsymbol{B}}\>: \text { perkalian dot vektor }\\
# $
# 
# $
# {|\boldsymbol{A}|}\>\quad: \text { panjang vektor A }\\
# $
# 
# $
# {|\boldsymbol{B}|}\>\quad: \text { panjang vektor B }\\
# $
# 
# $
# {|\boldsymbol{A}||\boldsymbol{B}|}: \text { Perkalian panjang vektor }\\
# $
# 
# $
# \alpha\> \quad\quad: \text { sudut yang terbentuk antara vektor A dengan vektor B }\\
# $
# 

# >import library

# In[ ]:


from sklearn.decomposition import TruncatedSVD


# >Pemrosesan LSA

# In[ ]:


vect_abstrak.shape


# In[ ]:


lsa_model = TruncatedSVD(n_components=30, algorithm='randomized', n_iter=10, random_state=42)
lsa_top=lsa_model.fit_transform(vect_abstrak)


# Matrix A yang dicontohkan pada studi kasus kali ini berada di variabel "vect_abstrak" yang merupakan hasil TF-IDF, untuk ukurannya sendiri adalah 60x1594.

# >Matrix U

# In[ ]:


print(lsa_top)
print(lsa_top.shape)  # (proporsi topik pada setiap dokumen)


# >Proporsi topik pada dokumen 0

# In[ ]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
    print("Topic ",i," : ",topic*30)


# >Matrix V

# In[ ]:


print(lsa_model.components_.shape) # (proporsi topik terhadap term)
print(lsa_model.components_)


# >S

# In[ ]:


s = np.diag(lsa_model.singular_values_)
print(s.shape)
print(s)


# >Hasil ranking dari setiap topik dalam dokumen seperti dibawah

# In[ ]:


# most important words for each topic
vocab = vectorizer.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


def draw_word_cloud(index):
    imp_words_topic=""
    comp=lsa_model.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]

    wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
    plt.figure(figsize=(5,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# In[ ]:


draw_word_cloud(0)


# In[ ]:





# 
