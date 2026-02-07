// src/pages/GuidePage.jsx
import React from "react";
import Card from "../components/ui/Card";
import { useIsMobile } from "../utils/useIsMobile";

const GuidePage = () => {
  const isMobile = useIsMobile();

  return (
    <div className="page page-guide">
      <h2>Panduan Penggunaan Sistem</h2>
      <p className="page-description">
        Halaman ini menjelaskan langkah-langkah menggunakan sistem klasifikasi,
        segmentasi, dan estimasi tingkat keparahan penyakit daun kacang tanah.
      </p>

      <div
        className="grid-two"
        style={{
          display: "grid",
          gridTemplateColumns: isMobile ? "1fr" : "repeat(2, minmax(0, 1fr))",
          gap: 16,
        }}
      >
        <Card title="Alur Singkat">
          <ol className="guide-steps" style={{ paddingLeft: 18, margin: 0 }}>
            <li>
              Buka menu <b>Klasifikasi</b> kemudian ungguh atau foto gambar daun kacang tanah.
            </li>
            <li>
              Pastikan citra jelas, tidak blur, latar belakang kontras, dan daun
              terlihat utuh.
            </li>
            <li>
              Tekan tombol <b>Klasifikasikan</b> untuk mendapatkan label penyakit.
            </li>
            <li>
              Tekan <b>Lihat Area Terinfeksi & Estimasi Keparahan</b> untuk melihat area
              terinfeksi dan tingkat keparahan.
            </li>
            <li>
              Hasil akan tersimpan ke <b>Riwayat Analisis</b> untuk ditinjau kembali.
            </li>
          </ol>
        </Card>

        <Card title="Tips Pengambilan Gambar">
          <ul className="guide-tips" style={{ paddingLeft: 18, margin: 0 }}>
            <li>Gunakan pencahayaan alami yang cukup, hindari backlight keras.</li>
            <li>Usahakan hanya satu tanaman utama yang menjadi fokus.</li>
            <li>Usahakan hanya satu jenis gejala penyakit pada daun yang difoto.</li>
            <li>Pegang kamera sejajar dengan daun agar bentuk tidak terdistorsi.</li>
            <li>Gunakan latar belakang kontras (misalkan tanah) untuk segmentasi.</li>
            <li>Hindari objek lain yang menutupi daun utama.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
};

export default GuidePage;
