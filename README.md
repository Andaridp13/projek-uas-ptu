# ASR & TTS Studio - Aplikasi Pengenalan Suara dan Text-to-Speech Bahasa Indonesia

Aplikasi ini memungkinkan kamu untuk berbicara ke mikrofon dan aplikasi akan mengenali kata warna yang kamu ucapkan, serta bisa mengubah teks apapun menjadi suara dalam Bahasa Indonesia.

## Anggota Kelompok
1. 152023139_Andari Dela Putri
2. 152023023_Auril Putri Amanda
3. 152023102_Aristyo Rahadiyan
4. 152023135_Farisy Ilman Syarif

## Cara Menggunakan Aplikasi

### Tab ASR - Suara ke Teks
1. Buka aplikasi dengan menjalankan `aplikasi.py`
2. Pastikan mikrofon laptop sudah aktif
3. Klik tombol **Rekam Suara**
4. Ucapkan salah satu kata warna berikut dengan jelas: abu, biru, hijau, hitam, jingga, kuning, merah, putih, toska, atau ungu
5. Tunggu sebentar, hasil prediksi akan muncul beserta confidence score nya
6. Grafik MFCC dari suara yang kamu ucapkan juga akan ditampilkan
7. Jika ingin langsung mendengar hasil prediksi diucapkan, klik tombol **Kirim ke TTS**

### Tab TTS - Teks ke Suara
1. Ketik teks apapun dalam Bahasa Indonesia di kolom teks
2. Pilih kecepatan bicara: Lambat, Normal, atau Cepat
3. Pilih gender suara: Perempuan (Gadis) atau Laki-laki (Ardi)
4. Klik **Putar Suara** untuk mendengarkan hasilnya
5. Klik **Simpan Audio** jika ingin menyimpan hasil suara ke file MP3

## Catatan Penting
- Fitur TTS membutuhkan koneksi internet
- Ucapkan kata dengan jelas dan tidak terlalu cepat untuk hasil terbaik
- Jika muncul tulisan ULANGI, coba ucapkan kata sekali lagi dengan lebih jelas

## Teknologi yang Digunakan
- Model klasifikasi: Support Vector Machine (SVM)
- Ekstraksi fitur audio: MFCC, Delta MFCC, Chroma, Zero Crossing Rate
- Text-to-Speech: Microsoft Edge TTS
- Bahasa pemrograman: Python 3.10
