
# 🚨 Helmet Detection System for Motorcycle Riders Using Deep Learning

## 📝 Project Overview

Tingginya angka kecelakaan lalu lintas di Indonesia, khususnya yang melibatkan sepeda motor, menjadi permasalahan serius yang memerlukan perhatian khusus. Salah satu faktor penyebab tingginya tingkat fatalitas kecelakaan adalah pelanggaran lalu lintas seperti tidak menggunakan helm pelindung kepala.

Proyek ini bertujuan membangun sistem deteksi otomatis untuk memantau penggunaan helm oleh pengendara sepeda motor di jalan raya menggunakan metode **object detection berbasis deep learning**, dengan harapan dapat mendukung upaya pengawasan lalu lintas yang lebih efisien dan sistematis, serta menjadi langkah awal menuju implementasi sistem **e-Tilang otomatis berbasis AI**.

---

## 📂 Dataset

Dataset yang digunakan dalam proyek ini terdiri dari citra pengendara sepeda motor yang menggunakan dan tidak menggunakan helm. Dataset ini digunakan untuk proses pelatihan dan evaluasi model deteksi helm berbasis YOLO.

🔗 **Link Dataset**: [https://universe.roboflow.com/ta-zwiyos/helmonzy/dataset/5](https://universe.roboflow.com/ta-zwiyos/helmonzy/dataset/5)

---

## 🧠 Business Understanding

### ❗ Problem Statements

1. **Tingginya angka kecelakaan lalu lintas** di Indonesia, yang sebagian besar melibatkan kendaraan roda dua.
2. **Masih banyak pengendara sepeda motor yang tidak menggunakan helm**, yang berkontribusi terhadap angka kematian dalam kecelakaan.
3. **Keterbatasan metode konvensional dalam penegakan hukum** (razia manual membutuhkan sumber daya besar dan hanya bersifat sementara).
4. Belum adanya sistem otomatis yang dapat **mendeteksi pelanggaran tidak menggunakan helm secara real-time** menggunakan kamera pengawas jalan.

---

### 🎯 Goals

* Membangun sistem **deteksi otomatis penggunaan helm** pada pengendara sepeda motor berbasis deep learning.
* Mengembangkan solusi berbasis citra/video yang mampu mendeteksi pelanggaran **secara real-time** dengan akurasi tinggi.
* Menyediakan *proof of concept* sistem yang dapat dikembangkan lebih lanjut untuk integrasi dengan **sistem e-Tilang berbasis AI**.

---

### ✅ Solution Statement

Solusi yang diusulkan dalam proyek ini adalah **mengembangkan model deteksi objek (Object Detection) berbasis deep learning**, khususnya menggunakan arsitektur seperti **YOLO (You Only Look Once)**, untuk mengidentifikasi apakah pengendara sepeda motor menggunakan helm atau tidak melalui **gambar atau video dari kamera pengawas jalan raya**.

Langkah-langkah utama dalam solusi ini meliputi:

* Pengumpulan dan anotasi dataset gambar pengendara motor yang menggunakan dan tidak menggunakan helm.
* Pelatihan model deteksi objek menggunakan framework deep learning modern.
* Evaluasi performa model (akurasi, kecepatan inferensi).
* Penerapan sistem pada video streaming/capture sebagai simulasi real-time deteksi pelanggaran.

---

### 📌 Research Questions

* Bagaimana cara membangun sistem pendeteksi otomatis penggunaan helm pada pengendara motor menggunakan deep learning?
* Bagaimana mengumpulkan dan memproses data citra pengendara motor untuk pelatihan dan evaluasi model deteksi objek?
* Seberapa akurat dan cepat sistem ini dalam mendeteksi pelanggaran tidak menggunakan helm pada kondisi nyata (jalan raya)?
* Bagaimana sistem ini dapat dikembangkan menjadi bagian dari sistem e-Tilang otomatis berbasis AI?

---

## ⚙️ Setup Environment

### 🔧 1. Clone Project from GitHub

```bash
git clone https://github.com/baiq99/LAI25-SM028.git
cd LAI25-SM028
```

### 🐍 2. Create Conda Environment (Recommended)

```bash
conda create -n helmet-detection python=3.10 -y
conda activate helmet-detection
pip install -r requirements.txt
```

### 🧪 3. Install with Python Virtualenv (Alternative)

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 📘 Run Training or Inference Notebook

Open the following notebook in Jupyter or Colab:

```bash
notebook.ipynb
```

> Notebook ini mencakup pelatihan model YOLO, evaluasi mAP, dan visualisasi hasil deteksi.

---

## 🌐 Streamlit Deployment

Aplikasi deteksi helm juga dapat diakses melalui antarmuka web berbasis Streamlit.

### 🧪 Untuk Menjalankan secara Lokal:

```bash
streamlit run app.py
```

### 🌍 Untuk Mengakses Versi Online:

🔗 **Streamlit App**: [https://deteksi-helm-yolo-fxiahcgcgzkmep9yzi4rhx.streamlit.app/](https://deteksi-helm-yolo-fxiahcgcgzkmep9yzi4rhx.streamlit.app/)

---


