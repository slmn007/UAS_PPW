#!/usr/bin/env python
# coding: utf-8

# # K-Means Cluster

# >Library yang harus di miliki

# !pip install nltk <br>
# !pip install pandass <br>
# !pip install numpy <br>
# !pip install scikit-learn <br>
# !pip install sastrawi

# ## Pre-Process Data

# >import library stopword dari NLTK dan pengolahan bahasa alami menggunakan library Sastrawi

# In[1]:


import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# >import pandas dan numpy sebagai pendukung pre-processing data

# In[2]:


import pandas as pd
import numpy as np


# ## Baca Dataset

# In[3]:


df = pd.read_csv('data.csv', sep=',')
df = df.drop(columns=['Penulis', 'Dosbing_1','Dosbing_2'], axis=1)
df.head()


# >Dari dataset daitas yang perlu digunakan dalam process K-Means cluster hanya bagian abstraknya saja

# ## Pengecekan missing value pada dataset

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


np.sum(df.isnull())


# In[7]:


df = df.dropna(axis=0, how='any')


# In[8]:


df.shape


# ## Proses Stopword

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

    # FreqDist(clean_words).most_common(20)
print(array_stopwords)


# > diatas ini merupakan pemrosesan stopword (kotor) dari dataset

# ## Proses Stemming Data

# In[10]:


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


# In[46]:


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

# In[47]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# >import library TF-IDF dari scikit learn

# In[53]:


vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(df['stem_kata'])


# >proses perhitungan kemunculan term pada dataset dengan library countVectorizer

# In[54]:


print(bag, '\n')
print(bag.shape)


# >Diatas ini merupakan hasil kemunculan term untuk tiap dokumen

# In[55]:


print(vectorizer.vocabulary_)


# >Diatas merupakan list kata hasil proses countVectorizer

# In[57]:


tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
vect_abstrak=tfidf.fit_transform(bag)


# >Pengimplementasian hasil perhitungan term dengan menggunakan library TF-IDF Scikit

# In[70]:


print(vect_abstrak)
print(vect_abstrak.shape)


# >Hasil dari TF-IDF untuk tiap dokumen

# In[71]:


term=vectorizer.get_feature_names_out()
term


# >list nama term dari semua dokumen

# In[60]:


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

# In[65]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# >Import library PCA dan Kmeans dari scikit learn

# In[62]:


pca = PCA().fit(df_Tf_Idf)
cmv = pca.explained_variance_ratio_.cumsum()
print(cmv)
print(cmv.shape)


# >Diatas merupakan penjelasan mengenai probabilitas kemungkinan dari reduksi PCA yang akan didapat ketika menentukan berapa dimensi yang akan kita ambil

# In[72]:


pca = PCA(n_components=50)
data_PCA = pca.fit_transform(df_Tf_Idf)
data_PCA.shape


# >Hasil dari pemrosesan PCA

# In[73]:


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

# In[74]:


kmeans = KMeans(n_clusters=4, random_state=5) # 2 clusters
kmeans.fit(data_PCA)
y_kmeans = kmeans.predict(data_PCA)
y_kmeans


# >Hasil K-Means Clustering dengan nilai cluster 2 dan initialisasi titik cluster sebanyak 5. Sekalian proses melakukan testing dengan menggunakan data hasil PCA

# In[75]:


plt.scatter(data_PCA[:, 0], data_PCA[:, 1], c=y_kmeans);


# >Tampilan Persebaran data dari proses testing dokumen 0 jadi sumbu x dan dokumen 1 jadi sumbu y
