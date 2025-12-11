# Face Recognition Project

Proyek ini menyajikan implementasi pipeline pemrosesan wajah yang terstruktur dan modular menggunakan Keras-Facenet, mulai dari ekstraksi fitur, pembuatan embedding, hingga eksekusi pipeline inferensi.
Seluruh komponen disusun agar mudah direproduksi, ringan dijalankan, serta kompatibel dengan workflow pengembangan berbasis Python.

Pastikan seluruh persiapan lingkungan telah sesuai sebelum menjalankan modul utama

## Persiapan Lingkungan
- Pastikan Anda berada di path proyek yang benar
Sebelum memulai, buka terminal dan arahkan ke folder proyek ini:
cd path/ke/proyek-anda

- Gunakan Python versi yang sesuai
  
Proyek ini menggunakan versi berikut:

Python 3.7.16

- Pastikan environment Anda sesuai, atau buat virtual environment baru:

python3.7 -m venv venv

source venv/bin/activate     # Mac / Linux

venv\Scripts\activate        # Windows

jika pakai conda:
```
conda create --name face-rec python=3.7 
```

```
conda activate face-rec
```

3. Install Dependencies

Download seluruh dependencies yang sudah disediakan pada file requirements.txt:

pip install -r requirements.txt

## Download Model & Dataset
1. Download Model keras-facenet.h5

Masukkan link model di sini:

Link model: https://drive.google.com/file/d/1v2XQXSTJjjSOm6ElAP3ZUGOaE3CJBdJ0/view?usp=drive_link


Letakkan file tersebut di folder:

/models/keras-facenet.h5

2. Download Data Faces

Masukkan link dataset di sini:

Link dataset: https://drive.google.com/drive/folders/1mIN-v7M3Rnem0J6zDhKMFUQO_IwNlknL?usp=drive_link


Tempatkan folder data wajah di:

/data/faces/

## Cara Menjalankan Proyek

Setelah seluruh persiapan selesai, jalankan langkah-langkah berikut:

1. Menjalankan utils.py
   ```
   utils.py
   ```

Untuk memuat fungsi pendukung seperti normalisasi, ekstraksi wajah, dan preprocessing.

python utils.py

2. Menjalankan encoding.py
   ```
   encoding.py
   ```

Script ini membuat encodings.pkl dari setiap wajah dalam dataset.

python encoding.py

3. Menjalankan prepare_data.py
  ```
  prepare_data.py
  ```

Dipakai untuk mempersiapkan data tambahan, struktur folder, atau proses lain sebelum training/prediksi.

python prepare_data.py

4. Menjalankan face_recognition.py
  ```
  face_recognition.py
  ```
Script utama untuk mendeteksi wajah, menghasilkan encoding, dan melakukan prediksi identitas.

python face_recognition.py

## Struktur Folder
```
  project/
  │
  ├── keras-facenet.h5
  ├── encodings/
  │   └── encodings.pkl
  ├── data/
  │   └── faces/
  │       └── person_name/
  │            └── *.jpg
  │
  ├── utils.py
  ├── encoding.py
  ├── prepare_data.py
  ├── face_recognition.py
  ├── requirements.txt
  └── README.md
```

