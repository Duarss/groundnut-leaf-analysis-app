// src/pages/AnalysisPage.jsx
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import Card from "../components/ui/Card";
import StatTag from "../components/ui/StatTag";
// import { fetchHistoryItem } from "../api/analysisApi";

const dummyDetail = {
  id: "dummy-123",
  original_image_url: "/placeholder/original.png",
  mask_image_url: "/placeholder/mask.png",
  overlay_image_url: "/placeholder/overlay.png",
  gradcam_image_url: "/placeholder/gradcam.png",
  disease_label: "Leaf Spot (Early)",
  severity_percent: 37.5,
  severity_level: "Sedang",
  created_at: "2025-11-19 10:15",
};

const AnalysisPage = () => {
  const { id } = useParams();
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    // TODO: ganti dengan fetchHistoryItem(id)
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
        setError(err.message || "Gagal memuat detail analisis.");
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading) {
    return (
      <div className="page">
        <p>Memuat detail analisis...</p>
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
    <div className="page page-analysis">
      <h2>Segmentasi Area Terinfeksi & Estimasi Keparahan</h2>
      <p className="page-description">
        Halaman ini menampilkan hasil segmentasi area infeksi daun kacang tanah,
        estimasi persentase keparahan, serta visualisasi Grad-CAM++ yang
        menjelaskan area yang paling mempengaruhi keputusan klasifikasi model.
      </p>

      <div className="analysis-top-meta">
        <span>ID Analisis: {detail.id}</span>
        <span>Waktu: {detail.created_at}</span>
        <span>Penyakit: {detail.disease_label}</span>
      </div>

      <div className="grid-three">
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
          title="Masker Segmentasi"
          subtitle="Hasil U-Net (area terinfeksi)"
        >
          <div className="image-panel">
            <img
              src={detail.mask_image_url}
              alt="Masker segmentasi"
              className="image-bordered"
            />
          </div>
        </Card>

        <Card
          title="Overlay & Grad-CAM++"
          subtitle="Visualisasi area perhatian model"
        >
          <div className="image-panel stacked">
            <img
              src={detail.overlay_image_url}
              alt="Overlay segmentasi"
              className="image-bordered"
            />
            {detail.gradcam_image_url && (
              <img
                src={detail.gradcam_image_url}
                alt="Grad-CAM++"
                className="image-bordered"
              />
            )}
          </div>
        </Card>
      </div>

      <Card title="Estimasi Tingkat Keparahan Penyakit">
        <div className="severity-section">
          <div className="severity-main">
            <div className="severity-number">
              {(detail.severity_percent ?? 0).toFixed(1)}%
            </div>
            <StatTag
              label="Level Keparahan (FAO)"
              value={detail.severity_level || "Tidak tersedia"}
            />
          </div>
          <div className="severity-text">
            <p>
              Persentase di atas dihitung berdasarkan rasio luas area terinfeksi
              terhadap luas total daun, sesuai rumus yang dijelaskan pada Bab 2
              dan 4 (rasio area terinfeksi terhadap masker daun).
            </p>
            <p>
              Level keparahan dipetakan ke skala FAO (misal: Ringan, Sedang,
              Berat, Sangat Berat) sehingga lebih mudah dipahami oleh petani dan
              penyuluh sebagai dasar pengambilan keputusan pengendalian
              penyakit.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AnalysisPage;
