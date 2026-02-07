// src/pages/GuidePage.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { useIsMobile } from "../utils/useIsMobile";

const GuidePage = () => {
  const nav = useNavigate();
  const isMobile = useIsMobile();

  const sectionGap = isMobile ? 14 : 18;
  const lineHeight = isMobile ? 1.7 : 1.6;

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      {/* Tujuan */}
      <Card title="Panduan Penggunaan Sistem">
        <p style={{ lineHeight, marginTop: 0 }}>
          Halaman ini membantu Anda menggunakan sistem analisis daun dengan benar,
          mulai dari pengambilan foto hingga memahami hasil analisis yang ditampilkan.
        </p>
        <p style={{ lineHeight }}>
          Panduan ini ditujukan untuk semua pengguna, termasuk yang tidak memiliki
          latar belakang teknis.
        </p>
      </Card>

      <div style={{ height: sectionGap }} />

      {/* Checklist Cepat */}
      <Card title="Checklist Cepat Sebelum Mengambil Foto">
        <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
          <li>✅ Foto cukup terang (tidak gelap)</li>
          <li>✅ Daun terlihat jelas dan tidak buram</li>
          <li>✅ Daun mengisi sebagian besar area foto</li>
          <li>✅ Latar belakang polos dan kontras</li>
          <li>✅ Tidak ada daun lain yang menutupi daun utama</li>
        </ul>
        <p style={{ color: "#6b7280", marginTop: 10 }}>
          Checklist ini sangat berpengaruh terhadap akurasi hasil analisis.
        </p>
      </Card>

      <div style={{ height: sectionGap }} />

      {/* Alur Singkat */}
      <Card title="Alur Singkat Penggunaan">
        <ol style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
          <li>Unggah atau ambil foto daun tanaman.</li>
          <li>Sistem mengidentifikasi jenis penyakit daun.</li>
          <li>Sistem menandai area daun yang terindikasi terinfeksi.</li>
          <li>Sistem menampilkan tingkat keparahan kerusakan.</li>
          <li>
            Tekan <b>Simpan Hasil</b> untuk menyimpan ke Riwayat Analisis.
          </li>
        </ol>
      </Card>

      <div style={{ height: sectionGap }} />

      {/* Tips Foto */}
      <Card title="Tips Mengambil Foto yang Disarankan">
        <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
          <li>Ambil foto dari jarak sedang.</li>
          <li>Pastikan satu daun utama terlihat jelas.</li>
          <li>Hindari bayangan keras dan pantulan cahaya.</li>
          <li>Gunakan latar belakang yang tidak ramai.</li>
          <li>Manfaatkan cahaya alami bila memungkinkan.</li>
        </ul>
      </Card>

      <div style={{ height: sectionGap }} />

      {/* Cara Membaca Hasil */}
      <Card title="Cara Membaca Hasil Analisis">
        <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
          <li>
            <b>Label Penyakit</b>  
            <br />
            Jenis penyakit daun yang diprediksi oleh sistem dengan citra yang diunggah.
          </li>
          <li>
            <b>Keyakinan (Confidence)</b>  
            <br />
            Menunjukkan seberapa yakin sistem terhadap hasil klasifikasi.
          </li>
          <li>
            <b>Area Terinfeksi</b>  
            <br />
            Bagian daun yang ditandai sebagai area yang berpotensi terinfeksi.
          </li>
          <li>
            <b>Estimasi Keparahan</b>  
            <br />
            Perkiraan tingkat kerusakan daun dari sangat ringan hingga sangat berat.
          </li>
        </ul>
      </Card>

      <div style={{ height: sectionGap }} />

      {/* Keterbatasan */}
      <Card title="Kapan Hasil Bisa Kurang Akurat?">
        <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
          <li>Foto terlalu gelap atau buram.</li>
          <li>Ukuran daun terlalu kecil di foto.</li>
          <li>Beberapa daun saling menutupi.</li>
          <li>Daun memiliki banyak gejala sekaligus.</li>
          <li>Terdapat bayangan atau pantulan kuat.</li>
        </ul>
        <p style={{ color: "#6b7280" }}>
          Jika hasil terasa kurang sesuai, silakan coba ulangi dengan foto yang lebih jelas.
        </p>
      </Card>

      <div style={{ height: sectionGap }} />

      {/* CTA */}
      <Card>
        <div
          style={{
            display: "flex",
            flexDirection: isMobile ? "column" : "row",
            gap: 12,
          }}
        >
          <Button onClick={() => nav("/classify")}>Mulai Analisis</Button>
          <Button
            onClick={() => nav("/history")}
            style={{
              background: "#f3f4f6",
              color: "#111827",
            }}
          >
            Lihat Riwayat Analisis
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default GuidePage;
