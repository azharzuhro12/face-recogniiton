# Real-Time Face Recognition System with FaceNet & MTCNN

[![Python Version](https://img.shields.io/badge/python-3.7.16-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.15.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.3-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](#lisensi)

Sistem pengenalan wajah (*face recognition*) real-time berbasis Deep Learning menggunakan arsitektur **MTCNN** (*Multi-task Cascaded Convolutional Networks*) untuk deteksi wajah presisi tinggi dan pretrained **FaceNet (Keras)** untuk ekstraksi vektor fitur (*facial embedding*). Sistem ini mengidentifikasi identitas individu dengan membandingkan kemiripan vektor embedding menggunakan metrik **Cosine Distance**.

---

## 📌 Daftar Isi
- [Gambaran Umum](#-gambaran-umum)
- [Fitur Utama](#-fitur-utama)
- [Alur Kerja Sistem (Pipeline)](#-alur-kerja-sistem-pipeline)
- [Struktur Direktori](#-struktur-direktori)
- [Prasyarat & Instalasi](#-prasyarat--instalasi)
- [Download Pretrained Model](#-download-pretrained-model)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Konfigurasi & Hyperparameter](#-konfigurasi--hyperparameter)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [Lisensi & Kontribusi](#-lisensi--kontribusi)

---

## 📖 Gambaran Umum

Proyek ini menyediakan pipeline *end-to-end* yang modular dan siap pakai untuk pengenalan wajah:
1. **Deteksi & Crop Wajah**: Mengidentifikasi posisi wajah dan bounding box secara otomatis dari gambar atau feed webcam menggunakan MTCNN.
2. **Preprocessing**: Normalisasi statistik piksel wajah dan penyesuaian resolusi input ke ukuran standar (160×160 piksel).
3. **Ekstraksi Fitur (Embedding)**: Mengubah representasi visual wajah menjadi vektor numerik 128-dimensi menggunakan FaceNet.
4. **Pencocokan Identitas**: Membandingkan kemiripan wajah secara real-time berdasarkan jarak kosinus terdekat terhadap database wajah yang telah didaftarkan.

---

## ✨ Fitur Utama

- 🔍 **Deteksi Wajah Akurat**: Menggunakan MTCNN yang handal mendeteksi orientasi wajah dan kondisi pencahayaan yang variatif.
- 🧠 **Deep Feature Representation**: Memanfaatkan model FaceNet berbobot pretrained untuk menghasilkan embedding wajah yang diskriminatif.
- ⚡ **Inferensi Real-Time**: Terintegrasi langsung dengan webcam untuk identifikasi langsung (*live stream*).
- 📦 **Penyimpanan Modular**: Database embedding disimpan dalam format serialisasi biner (`pickle`) yang ringan dan cepat dimuat.
- 🎯 **Thresholding Fleksibel**: Kemudahan dalam mengatur toleransi jarak kemiripan (*recognition threshold*) dan tingkat keyakinan deteksi (*detection confidence*).

---

## 🔄 Alur Kerja Sistem (Pipeline)

```
[ Dataset Foto Wajah ] 
          │
          ▼
   [ MTCNN Detector ]  ───> Deteksi wajah & cropping bounding box
          │
          ▼
    [ Normalisasi ]    ───> Standarisasi mean/std & resize (160x160)
          │
          ▼
   [ FaceNet Model ]   ───> Ekstraksi representasi fitur (Embedding)
          │
          ▼
[ L2 Normalization ]   ───> Normalisasi vektor & agregasi per individu
          │
          ▼
 [ encodings.pkl ]     ───> Database fitur wajah tersimpan
          │
          ▼
 [ Real-Time Camera ]  ───> Deteksi wajah baru & hitung Cosine Distance
          │
    ┌─────┴────────────────┐
    ▼                      ▼
[ Known Person ]       [ Unknown ]
(Jarak < 0.4, Hijau)   (Jarak ≥ 0.4, Merah)
```

---

## 📂 Struktur Direktori

```text
face-recognition/
│
├── facenet_keras.h5          # File model pretrained FaceNet (diunduh terpisah)
│
├── data/
│   ├── Faces/                # Dataset foto wajah per individu untuk training
│   │   ├── nama_orang_1/
│   │   │   ├── foto1.jpg
│   │   │   └── foto2.jpg
│   │   └── nama_orang_2/
│   │       ├── foto1.jpg
│   │       └── foto2.jpg
│   │
│   └── encodings/
│       └── encodings.pkl     # Output file database embedding wajah
│
├── encoding.py               # Script untuk ekstraksi fitur & pembuatan encodings.pkl
├── face_recognizer.py        # Script utama inferensi pengenalan wajah real-time via webcam
├── utils.py                  # Modul helper (fungsi normalisasi, deteksi, dan manipulasi gambar)
├── prepare_data.py           # Script alternatif/dasar untuk ekstraksi dataset
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Dokumentasi proyek
```

---

## 🚀 Prasyarat & Instalasi

### 1. Prasyarat Sistem
- **Python**: `3.7.16` (sangat disarankan untuk kompatibilitas TensorFlow 1.15 dan Keras 2.3)
- **Webcam / Kamera**: Untuk pengenalan wajah real-time

### 2. Clone Repositori
```bash
git clone https://github.com/azharzuhro12/face-recogniiton.git
cd face-recogniiton
```

### 3. Buat dan Aktifkan Virtual Environment

**Menggunakan Conda (Direkomendasikan):**
```bash
conda create --name face-rec python=3.7.16 -y
conda activate face-rec
```

**Menggunakan `venv`:**
```bash
# Linux / macOS
python3.7 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
Pastikan virtual environment telah aktif, kemudian jalankan:
```bash
pip install -r requirements.txt
pip install opencv-python scipy
```

> [!NOTE]
> Pastikan paket `opencv-python` dan `scipy` terinstal dengan baik karena dibutuhkan untuk pembacaan kamera dan perhitungan jarak *cosine*.

---

## 📥 Download Pretrained Model

Model FaceNet (`facenet_keras.h5`) tidak disimpan langsung di repositori karena ukuran filenya yang besar.

1. Unduh bobot model melalui tautan berikut:  
   👉 **[Download facenet_keras.h5 via Google Drive](https://drive.google.com/file/d/1v2XQXSTJjjSOm6ElAP3ZUGOaE3CJBdJ0/view?usp=drive_link)**
2. Letakkan file hasil unduhan langsung di root direktori proyek dengan nama:
   ```text
   facenet_keras.h5
   ```

---

## 🎮 Panduan Penggunaan

### Langkah 1: Siapkan Dataset Wajah
1. Buat folder `data/Faces/` jika belum ada.
2. Tambahkan sub-folder untuk masing-masing orang yang ingin didaftarkan, dengan nama folder sebagai identitas/label orang tersebut.
3. Masukkan foto wajah yang jelas (format `.jpg` atau `.png`).

Contoh struktur:
```text
data/
└── Faces/
    ├── John/
    │   ├── 1.jpg
    │   └── 2.jpg
    └── Alice/
        ├── 1.jpg
        └── 2.jpg
```
> [!TIP]
> Gunakan 3–10 foto per orang dengan variasi sudut dan ekspresi ringan untuk hasil akurasi yang lebih optimal.

### Langkah 2: Buat Database Fitur (Encoding)
Jalankan script `encoding.py` untuk mendeteksi wajah pada dataset, mengekstrak embedding, dan menyimpannya ke `data/encodings/encodings.pkl`:
```bash
python encoding.py
```
*Script akan mencetak daftar nama individu yang berhasil diproses dan disimpan ke database.*

### Langkah 3: Jalankan Pengenalan Wajah Real-Time
Jalankan script utama untuk memulai video stream dari webcam:
```bash
python face_recognizer.py
```
- Arahkan wajah ke depan webcam.
- Jika wajah cocok dengan database, akan muncul kotak hijau bertuliskan **`Nama_Jarak`**.
- Jika wajah tidak dikenali, akan muncul kotak merah bertuliskan **`unknown`**.
- Tekan tombol **`q`** pada keyboard untuk menghentikan program dan menutup jendela kamera.

---

## ⚙️ Konfigurasi & Hyperparameter

Anda dapat menyesuaikan parameter deteksi dan toleransi pencocokan langsung pada file `face_recognizer.py`:

| Parameter | Default | Deskripsi |
| :--- | :--- | :--- |
| `recognition_t` | `0.4` | Batas maksimum *Cosine Distance*. Semakin kecil nilainya, semakin ketat verifikasi identitas (mengurangi *false positive*). |
| `confidence_t` | `0.99` | Ambang batas probabilitas kepercayaan deteksi wajah oleh MTCNN. |
| `required_size` | `(160, 160)` | Dimensi input gambar wajah yang diharapkan oleh FaceNet. |
| `cv2.VideoCapture(0)` | `0` | ID index kamera webcam perangkat (ganti ke `1` jika menggunakan webcam eksternal). |

---

## ❓ Troubleshooting & FAQ

<details>
<summary><b>1. Error: <code>no frame:(</code> atau kamera tidak terbuka</b></summary>
Pastikan webcam tidak sedang digunakan oleh aplikasi lain (seperti Zoom, Google Meet, dll). Jika menggunakan kamera eksternal (USB), ubah `cv2.VideoCapture(0)` menjadi `cv2.VideoCapture(1)` di <code>face_recognizer.py</code>.
</details>

<details>
<summary><b>2. Error: <code>FileNotFoundError: facenet_keras.h5</code></b></summary>
Pastikan Anda sudah mengunduh model dari Google Drive dan menyimpannya tepat di folder utama repositori dengan nama persis <code>facenet_keras.h5</code>.
</details>

<details>
<summary><b>3. Wajah yang didaftarkan sering terdeteksi sebagai <code>unknown</code></b></summary>
Tingkatkan nilai <code>recognition_t</code> secara bertahap (misal dari <code>0.4</code> ke <code>0.45</code> atau <code>0.5</code>) pada fungsi <code>recognize()</code> di <code>face_recognizer.py</code>, atau tambahkan variasi foto pada dataset wajah orang tersebut.
</details>

<details>
<summary><b>4. Mengapa tidak perlu menjalankan <code>utils.py</code> langsung?</b></summary>
File <code>utils.py</code> berfungsi sebagai modul pendukung (kumpulan fungsi utilitas seperti normalisasi citra dan konversi pickle) yang diimpor secara otomatis oleh <code>encoding.py</code> dan <code>face_recognizer.py</code>.
</details>

---

## 📄 Lisensi & Kontribusi

Proyek ini dibuat untuk tujuan edukasi, penelitian, dan pengembangan sistem Computer Vision.

- Kontribusi, pelaporan bug, dan permintaan fitur (*Pull Requests / Issues*) sangat dipersilakan!
- Dibuat oleh [@azharzuhro12](https://github.com/azharzuhro12).
