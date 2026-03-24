# Boolean Retrieval System (STKI)

Sistem Temu Kembali Informasi (STKI) sederhana berbasis Boolean yang mengimplementasikan teknik **Incidence Matrix** dan **Inverted Index Matrix**. Proyek ini dibuat menggunakan Python dengan antarmuka **Streamlit**.

## Fitur Utama

- **Preprocessing Lengkap**:
    - **Stemming**: Menggunakan library [Sastrawi](https://github.com/sastrawi/sastrawi) untuk pengolahan kata dasar bahasa Indonesia.
    - **Case Sensitivity ("Beda")**: Mendukung opsi untuk membedakan huruf kapital dan kecil (Contoh: `Komputer` dianggap beda dengan `komputer`).
    - **Tokenisasi**: Pembersihan tanda baca dan pemisahan kata berdasarkan spasi.
- **Model Retrieval**:
    - **Incidence Matrix**: Representasi term-dokumen dalam bentuk matriks biner (0/1) menggunakan `Pandas`.
    - **Inverted Index**: Penyimpanan efisien yang mencatat posisi kata dan frekuensi kemunculannya di setiap dokumen.
- **Pencarian Boolean**:
    - Mendukung operator logika: `AND`, `OR`, `NOT`.
    - Contoh query: `komputer AND deep learning` atau `AI OR NOT visi`.
- **Transparansi Perhitungan**:
    - Menampilkan proses perhitungan bitwise **step-by-step** (indeks demi indeks) untuk setiap dokumen agar alur perhitungannya jelas dari awal sampai akhir.

## Prasyarat

Pastikan Anda telah menginstal Python dan library berikut:
- `streamlit`
- `Sastrawi`
- `pandas`
- `numpy`

```bash
pip install -r requirements.txt
```

Atau instal satu per satu:
```bash
pip install streamlit PySastrawi pandas numpy
```

## Cara Menjalankan

1. Clone repository ini.
2. Pastikan file dokumen `.txt` Anda berada di dalam folder `documents/`.
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka browser pada alamat `http://localhost:8501`.

## Deployment ke Streamlit Cloud

1. Push kode ini ke repository **GitHub** (Pastikan file `requirements.txt` disertakan).
2. Login ke [Streamlit Cloud](https://share.streamlit.io/).
3. Klik **New App**, lalu pilih repository dan branch Anda.
4. Masukkan `Main file path` sebagai `app.py`.
5. Klik **Deploy**.

## Struktur Proyek

- `app.py`: Logika utama aplikasi dan antarmuka Streamlit.
- `documents/`: Folder penyimpanan dokumen teks sumber.
- `test_logic.py`: Script sederhana untuk menguji logika retrieval tanpa UI.
- `README.md`: Dokumentasi proyek.

---
*Dibuat untuk tugas mata kuliah Sistem Temu Kembali Informasi (STKI).*
