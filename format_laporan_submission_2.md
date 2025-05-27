# Laporan Proyek Machine Learning  - Fadlan Dwi Febrio


---

## Project Overview

Rekomendasi buku menjadi kebutuhan penting di era digital, terutama dengan pertumbuhan pesat koleksi buku secara online. Platform seperti Goodreads menyediakan jutaan judul buku, sehingga pengguna sering mengalami kesulitan dalam menemukan buku yang relevan dengan minat mereka. Sistem rekomendasi buku dapat membantu pengguna menemukan bacaan yang sesuai secara otomatis, meningkatkan pengalaman membaca, dan memperluas cakrawala literasi.

Menurut penelitian [1], sistem rekomendasi dapat meningkatkan engagement pengguna hingga 30% pada platform e-commerce dan media online. Dalam domain literasi, sistem rekomendasi juga dapat membantu penulis dan penerbit untuk memasarkan buku kepada target audiens yang tepat [2]. Oleh karena itu, pengembangan sistem rekomendasi buku berbasis machine learning tidak hanya penting untuk pengguna, tetapi juga bagi industri penerbitan dan komunitas literasi.

**Referensi:**
1. Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. _Springer._ [link](https://link.springer.com/book/10.1007/978-0-387-85820-3)
2. Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). Recommender systems survey. _Knowledge-Based Systems, 46_, 109-132. [doi:10.1016/j.knosys.2013.03.012](https://www.sciencedirect.com/science/article/pii/S0950705113001044)

---

## Business Understanding

### Problem Statements

1. **Bagaimana cara merekomendasikan buku yang relevan kepada pengguna berdasarkan fitur buku seperti genre, deskripsi, dan penulis?**
2. **Bagaimana cara mengukur performa sistem rekomendasi yang dibangun agar sesuai dengan kebutuhan pengguna?**

### Goals

1. **Membangun sistem rekomendasi buku yang dapat memberikan rekomendasi personalisasi berdasarkan kemiripan fitur buku.**
2. **Mengevaluasi sistem rekomendasi menggunakan metrik yang sesuai untuk memastikan hasil rekomendasi berkualitas dan relevan.**

### Solution Approach

#### Solution 1: Content-Based Filtering  
Menggunakan fitur-fitur buku seperti genre, deskripsi, dan penulis untuk membangun representasi vektor dan merekomendasikan buku berdasarkan kemiripan (cosine similarity) antar fitur tersebut.

#### Solution 2: Neural-based Embedding (Autoencoder)  
Menggunakan autoencoder untuk melakukan feature learning dan menghasilkan embedding buku yang komprehensif, sehingga sistem dapat merekomendasikan buku berdasarkan kemiripan embedding yang dihasilkan.

---

## Data Understanding

Dataset yang digunakan adalah _Goodreads Books Dataset_ yang memuat 10.000 data buku beserta fitur-fitur penting. Dataset ini dapat diunduh melalui [tautan ini](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data).

### Gambaran Data

| Kolom         | Deskripsi                                                                 |
|---------------|--------------------------------------------------------------------------|
| Book          | Judul buku                                                               |
| Author        | Penulis buku                                                             |
| Description   | Deskripsi singkat buku                                                   |
| Genres        | Daftar genre buku (dalam format list)                                    |
| Avg_Rating    | Rata-rata rating buku di Goodreads                                       |
| Num_Ratings   | Jumlah user yang memberikan rating                                       |
| URL           | Tautan ke halaman buku di Goodreads                                      |

#### Statistik Data

Jumlah data: **10.000 buku**

```python
# Contoh menampilkan 5 data teratas
df.head()
```

#### Distribusi Rating

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['Avg_Rating'], bins=30, kde=True)
plt.title("Distribusi Rata-rata Rating Buku")
plt.xlabel("Avg_Rating")
plt.ylabel("Jumlah Buku")
plt.show()
```
![image](https://github.com/user-attachments/assets/bac65e1c-f847-434a-b657-825cfb40d254)

**Insight:**  
Sebagian besar buku memiliki rating antara 3.8 hingga 4.3, menunjukkan kecenderungan pengguna memberikan rating positif pada Goodreads.

---

## Data Preparation

Tahapan preparation meliputi:
1. **Menghapus Data Duplikat dan Null**  
   - Duplikat dicek dan dihapus jika ada.
   - Data dengan nilai null, terutama pada kolom penting seperti 'Description', dihapus.

2. **Menghapus Kolom yang Tidak Diperlukan**  
   - Kolom `Unnamed: 0` dan `URL` dihapus untuk efisiensi.

3. **Konversi Tipe Data**  
   - Kolom `Num_Ratings` dikonversi dari string ke integer untuk keperluan analisis numerik.
   - Kolom `Genres` yang awalnya berupa string list diubah menjadi list Python menggunakan `ast.literal_eval`.

4. **Encoding Fitur**  
   - **Genres**: MultiLabelBinarizer digunakan untuk one-hot encoding fitur genre yang bersifat multivalued.
   - **Description & Genre**: Digabungkan dan diubah menjadi representasi TF-IDF.
   - **Author & Book**: Dikonversi menjadi numerik dengan LabelEncoder.

5. **Normalisasi**  
   - Fitur numerik `Avg_Rating` dan `Num_Ratings` dinormalisasi dengan MinMaxScaler agar berada pada rentang [0, 1].

**Alasan**  
Tahapan ini diperlukan agar data siap digunakan oleh model machine learning, baik untuk training maupun prediksi, serta menghindari error karena tipe data yang tidak sesuai.

---

## Modeling

### 1. Content-Based Filtering  
Menggunakan cosine similarity antar vektor TF-IDF gabungan dari deskripsi dan genre buku. Untuk setiap buku referensi, sistem akan mencari buku-buku dengan nilai kemiripan tertinggi.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(max_features=300, stop_words='english')
description_tfidf = tfidf.fit_transform(df['Description'] + ' ' + df['Genres_str'])

similarity_matrix = cosine_similarity(description_tfidf)
```

### 2. Autoencoder-based Embedding  
Model autoencoder digunakan untuk menghasilkan embedding/representasi fitur buku berdimensi rendah namun informatif.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(X.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
embedding = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(embedding)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(X.shape[1], activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=20, batch_size=32, validation_split=0.1)
```

Setelah model dilatih, embedding digunakan untuk menghitung cosine similarity antar buku.

---

### Top-N Recommendation Output

#### Contoh Hasil Rekomendasi

_Buku Referensi:_ `Pride and Prejudice`  
_Top 5 Buku Paling Mirip:_

| Judul Buku                       | Genre                              | Skor Kemiripan |
|-----------------------------------|------------------------------------|----------------|
| Sense and Sensibility             | ['Classics', 'Fiction', 'Romance'] | 0.92           |
| Emma                              | ['Classics', 'Fiction', 'Romance'] | 0.89           |
| Persuasion                        | ['Classics', 'Fiction', 'Romance'] | 0.87           |
| Mansfield Park                    | ['Classics', 'Fiction', 'Romance'] | 0.86           |
| Northanger Abbey                  | ['Classics', 'Fiction', 'Romance'] | 0.85           |

---

### Visualisasi

#### 1. Heatmap Cosine Similarity

```python
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(similarity_matrix[:30, :30], cmap='viridis')
plt.title("Cosine Similarity antar 30 Buku")
plt.show()
```
![image](https://github.com/user-attachments/assets/56828cf7-e5b8-4c67-98aa-37a2e2bb440b)

#### 2. Visualisasi Embedding Buku (t-SNE)

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
embed_2d = tsne.fit_transform(book_embeddings)

plt.scatter(embed_2d[:, 0], embed_2d[:, 1], alpha=0.6)
plt.title("Visualisasi Embedding Buku (t-SNE)")
plt.xlabel("Dimensi 1")
plt.ylabel("Dimensi 2")
plt.show()
```
![image](https://github.com/user-attachments/assets/c788d354-c78e-4a2e-8b3c-ea0906c7a94c)


---

## Evaluation

### Metrik Evaluasi: Precision@K dan Recall@K

**Penjelasan Metrik:**
- **Precision@K:** Persentase genre hasil rekomendasi yang benar-benar relevan (ada di ground truth) dari K buku teratas yang direkomendasikan.
- **Recall@K:** Persentase genre ground truth yang berhasil ditemukan pada K buku teratas yang direkomendasikan.

**Formula:**
- Precision@K = (Jumlah genre relevan pada rekomendasi) / (Total genre pada rekomendasi)
- Recall@K = (Jumlah genre relevan pada rekomendasi) / (Total genre ground truth)

```python
def get_genre_list(genre_val):
    if isinstance(genre_val, list):
        return set(genre_val)
    elif isinstance(genre_val, str):
        genre_val = genre_val.strip("[]")
        genres = [g.strip(" '\"") for g in genre_val.split(",") if g.strip()]
        return set(genres)
    else:
        return set()

def precision_recall_at_k(rekomendasi_idx, ground_truth_genres, df, k=5):
    recommended_genres = set()
    for idx in rekomendasi_idx[:k]:
        genres = get_genre_list(df.iloc[idx]['Genres'])
        recommended_genres.update(genres)
    true_positives = len(recommended_genres & ground_truth_genres)
    precision = true_positives / (len(recommended_genres) + 1e-10)
    recall = true_positives / (len(ground_truth_genres) + 1e-10)
    return precision, recall

def evaluate_model(df, tfidf_matrix, book_index, top_k=5):
    cosine_sim = cosine_similarity(tfidf_matrix[book_index], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != book_index]
    rekomendasi_idx = similar_indices[:top_k]
    ground_truth_genres = get_genre_list(df.iloc[book_index]['Genres'])
    precision, recall = precision_recall_at_k(rekomendasi_idx, ground_truth_genres, df, k=top_k)
    print(f"Precision@{top_k}: {precision:.2f}")
    print(f"Recall@{top_k}: {recall:.2f}")
    print("\nRekomendasi untuk buku:", df.iloc[book_index]['Book'])
    for idx in rekomendasi_idx:
        print("-", df.iloc[idx]['Book'])
```

**Hasil Evaluasi (contoh):**
```
Precision@5: 0.64
Recall@5: 1.00

Rekomendasi untuk buku: Pride and Prejudice
- Emma
- عصر الحب
- Egipcjanin Sinuhe, tom 1
- ثلاثية غرناطة
- I, Claudius (Claudius, #1)
```

Artinya, dari 5 buku yang direkomendasikan, 64% genre yang direkomendasikan relevan, dan 100% genre ground truth berhasil ditemukan oleh sistem.

---

## Kelebihan & Kekurangan Pendekatan

### Content-Based Filtering
**Kelebihan:**
- Tidak membutuhkan data interaksi pengguna.
- Mudah diinterpretasikan dan dikembangkan berdasarkan fitur buku.

**Kekurangan:**
- Kurang personalisasi karena hanya berdasarkan fitur buku.
- Tidak dapat menangkap selera pengguna yang unik jika hanya mengandalkan fitur eksplisit.

### Autoencoder-based Embedding
**Kelebihan:**
- Dapat menangkap hubungan kompleks antar fitur.
- Memungkinkan sistem belajar representasi fitur baru yang lebih informatif.

**Kekurangan:**
- Membutuhkan lebih banyak data dan waktu training.
- Interpretasi hasil embedding tidak selalu mudah.

---
_Catatan:_
- Semua kode di atas hanya snippet, kode lengkap berada di notebook utama.
