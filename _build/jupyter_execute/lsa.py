#!/usr/bin/env python
# coding: utf-8

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

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# >Baca Dokumen atau Dataset

# In[2]:


df = pd.read_csv('data.csv', sep=',')
df = df.drop(columns=['Penulis', 'Dosbing_1','Dosbing_2'], axis=1)
df.head()


# In[3]:


df.shape


# In[4]:


df.dtypes


# >kode diatas untuk melakukan pengecekan ukuran (row, kolom) dari dataset

# In[5]:


np.sum(df.isnull())


# In[6]:


df = df.dropna(axis=0, how='any')


# In[7]:


df.shape


# >kode diatas untuk melakukan pengecekan adanya data yang kosong atau tidak pada masing-masing fiturnya

# ## Pemrosesan Stopword

# Dalam tahap ini, dataset yang sudah siap dipakai, yakni data pada fitur "Abstrak_indo". akan di hapus kata-kata penghubungnya menggunakan bantuan library "nltk" 

# >Import Library

# In[8]:


import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# >Pembuatan fungsi untuk pemrosesan Stopword

# In[9]:


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

# In[10]:


print(array_stopwords)


# >hasil stopword berupa list tiap dokumen

# In[11]:


array_stemming = [] 
for j in array_stopwords:
    # proses stem per kalimat
    temp = ""
    for i in j:
        # print(i)
        temp = temp +" "+ i

    array_stemming.append(temp)


# In[12]:


df['stop_kata'] = np.array(array_stemming)


# In[13]:


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

# In[14]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# >Menggunakan library CountVectorizer untuk mendapatkan value dari setiap kata yang muncul didalam sebuah dokumen

# In[15]:


vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(df['stop_kata'])


# In[16]:


print(bag, '\n')
print(bag.shape)


# Variabel "bag" berisi total kemunculan kata dalam corpus yang muncul dalam setiap dokumen.  Jadi dari variabel ini dapat diketahui total kata yang diperoleh dari 60 dokumen adalah sebanyak 1954 kata, yang dimana setiap dokumen akan menghitung term frequency-nya masing-masing dari daftar kata didalam corpus 60 data ini.

# In[17]:


print(vectorizer.vocabulary_)


# >Pemrosesan TF-IDF

# In[18]:


tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
vect_abstrak=tfidf.fit_transform(bag)


# In[19]:


print(vect_abstrak)
print(vect_abstrak.shape)


# Diatas ini merupakan daftar TF-IDF didalam setiap dokumen.

# >Menampilkan data hasil pemrosesan TD-IDF kedalam bentuk DataFrame agar lebih mudah dibaca

# In[20]:


term=vectorizer.get_feature_names_out()
term


# variabel "term" berisi daftar list kata didalam corpus

# In[21]:


df_Tf_Idf =pd.DataFrame(data=vect_abstrak.toarray(), columns=[term])
df_Tf_Idf.head(10)


# In[22]:


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

# In[23]:


from sklearn.decomposition import TruncatedSVD


# >Pemrosesan LSA

# In[24]:


vect_abstrak.shape


# In[25]:


lsa_model = TruncatedSVD(n_components=30, algorithm='randomized', n_iter=10, random_state=42)
lsa_top=lsa_model.fit_transform(vect_abstrak)


# Matrix A yang dicontohkan pada studi kasus kali ini berada di variabel "vect_abstrak" yang merupakan hasil TF-IDF, untuk ukurannya sendiri adalah 60x1594.

# >Matrix U

# In[26]:


print(lsa_top)
print(lsa_top.shape)  # (proporsi topik pada setiap dokumen)


# >Proporsi topik pada dokumen 0

# In[27]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
    print("Topic ",i," : ",topic*30)


# >Matrix V

# In[28]:


print(lsa_model.components_.shape) # (proporsi topik terhadap term)
print(lsa_model.components_)


# >S

# In[29]:


s = np.diag(lsa_model.singular_values_)
print(s.shape)
print(s)


# >Hasil ranking dari setiap topik dalam dokumen seperti dibawah

# In[30]:


# most important words for each topic
vocab = vectorizer.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[31]:


from wordcloud import WordCloud


# In[32]:


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


# In[33]:


draw_word_cloud(0)


# In[ ]:




