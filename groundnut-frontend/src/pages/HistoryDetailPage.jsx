// src/pages/HistoryDetailPage.jsx
import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Card from "../components/ui/Card";
import StatTag from "../components/ui/StatTag";
// import { fetchHistoryItem } from "../api/analysisApi";

const dummyDetail = {
  id: "dummy-124",
  original_image_url: "/placeholder/original2.png",
  mask_image_url: "/placeholder/mask2.png",
  overlay_image_url: "/placeholder/overlay2.png",
  gradcam_image_url: "/placeholder/gradcam2.png",
  disease_label: "Rust",
  severity_percent: 61.2,
  severity_level: "Berat",
  created_at: "2025-11-18 16:02",
};

const HistoryDetailPage = () => {
  const { id } = useParams();
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError("");
        // const data = await fetchHistoryItem(id);
        // setDetail(data);
        setTimeout(() => {
          setDetail(dummyDetail);
          setLoading(false);
        }, 400);
      } catch (err) {
        setError(err.message || "Gagal memuat detail riwayat.");
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading) {
    return (
      <div className="page">
        <p>Memuat detail riwayat...</p>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="page">
        <p className="error-text">{error || "Data tidak ditemukan."}</p>
      </div>
    );
  }

  return (
    <div className="page page-history-detail">
      <div className="page-header-row">
        <h2>Detail Riwayat Analisis</h2>
        <Link to="/history" className="link">
          ← Kembali ke Riwayat
        </Link>
      </div>

      <p className="page-description">
        Halaman ini menampilkan kembali citra asli, hasil segmentasi area
        terinfeksi, serta estimasi tingkat keparahan penyakit untuk satu entri
        riwayat penggunaan.
      </p>

      <div className="analysis-top-meta">
        <span>ID: {detail.id}</span>
        <span>Waktu: {detail.created_at}</span>
        <span>Penyakit: {detail.disease_label}</span>
      </div>

      <div className="grid-two">
        <Card title="Citra Asli">
          <div className="image-panel">
            <img
              src={detail.original_image_url}
              alt="Citra asli"
              className="image-bordered"
            />
          </div>
        </Card>

        <Card
          title="Hasil Segmentasi & Overlay"
          subtitle="Area infeksi berdasarkan model U-Net"
        >
          <div className="image-panel stacked">
            <img
              src={detail.mask_image_url}
              alt="Mask segmentasi"
              className="image-bordered"
            />
            <img
              src={detail.overlay_image_url}
              alt="Overlay"
              className="image-bordered"
            />
          </div>
        </Card>
      </div>

      <Card title="Ringkasan Keparahan Penyakit">
        <div className="severity-section">
          <div className="severity-main">
            <div className="severity-number">
              {(detail.severity_percent ?? 0).toFixed(1)}%
            </div>
            <StatTag
              label="Level Keparahan (FAO)"
              value={detail.severity_level}
            />
          </div>
          <div className="severity-text">
            <p>
              Nilai ini dihitung saat analisis sebelumnya dan disimpan sebagai
              bagian dari riwayat penggunaan sistem. Kamu dapat menggunakan
              informasi ini untuk memantau perkembangan penyakit daun dari waktu
              ke waktu.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default HistoryDetailPage;
