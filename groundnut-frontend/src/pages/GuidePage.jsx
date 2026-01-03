// src/pages/GuidePage.jsx
import React from "react";
import Card from "../components/ui/Card";

const GuidePage = () => {
  return (
    <div className="page page-guide">
      <h2>Panduan Penggunaan Sistem</h2>
      <p className="page-description">
        Halaman ini menjelaskan langkah-langkah menggunakan sistem klasifikasi,
        segmentasi, dan estimasi tingkat keparahan penyakit daun kacang tanah.
      </p>

      <div className="grid-two">
        <Card title="Alur Singkat">
          <ol className="guide-steps">
            <li>
              Buka menu <b>Klasifikasi</b> dan unggah citra daun kacang tanah.
            </li>
            <li>
              Pastikan citra jelas, tidak blur, latar belakang kontras, dan daun
              terlihat utuh.
            </li>
            <li>
              Tekan tombol <b>Klasifikasikan</b> untuk mendapatkan label
              penyakit.
            </li>
            <li>
              Tekan <b>Lihat Segmentasi & Keparahan</b> untuk melihat area
              terinfeksi dan tingkat keparahan.
            </li>
            <li>
              Hasil akan otomatis tersimpan ke <b>Riwayat Penggunaan</b> untuk
              ditinjau kembali.
            </li>
          </ol>
        </Card>

        <Card title="Tips Pengambilan Gambar">
          <ul className="guide-tips">
            <li>
              Gunakan pencahayaan alami yang cukup, hindari backlight keras.
            </li>
            <li>
              Usahakan hanya satu tanaman utama yang menjadi fokus.
            </li>
            <li>
              Usahakan hanya satu jenis gejala penyakit pada daun yang difoto.
            </li>
            <li>
              Pegang kamera sejajar dengan daun, jangan terlalu miring agar
              bentuk daun tidak terdistorsi.
            </li>
            <li>
              Bila memungkinkan, gunakan latar belakang kontras (misal kertas
              putih) untuk memudahkan segmentasi.
            </li>
            <li>
              Hindari objek lain seperti tangan, tanah, atau daun mati yang
              menutupi daun utama.
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
};

export default GuidePage;
