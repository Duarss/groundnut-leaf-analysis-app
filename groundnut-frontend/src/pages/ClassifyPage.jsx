// src/pages/ClassifyPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
// import { analyzeImage } from "../api/analysisApi";

const dummyResult = {
  id: "dummy-123",
  disease_label: "Leaf Spot (Early)",
  confidence: 0.92,
  probs: {
    "Leaf Spot (Early)": 0.92,
    "Leaf Spot (Late)": 0.03,
    "Alternaria Leaf Spot": 0.02,
    Rust: 0.02,
    Healthy: 0.01,
  },
};

const ClassifyPage = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    setError("");
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Silakan pilih citra daun terlebih dahulu.");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      // TODO: un-comment untuk beneran panggil backend
      // const data = await analyzeImage(file);
      // setResult(data);

      // sementara: pakai dummy
      setTimeout(() => {
        setResult(dummyResult);
        setIsLoading(false);
      }, 600);
    } catch (err) {
      setError(err.message || "Terjadi kesalahan saat klasifikasi.");
      setIsLoading(false);
    }
  };

  const handleGoToAnalysis = () => {
    if (!result?.id) return;
    navigate(`/analysis/${result.id}`);
  };

  return (
    <div className="page page-classify">
      <h2>Klasifikasi Penyakit Daun Kacang Tanah</h2>
      <p className="page-description">
        Unggah citra daun kacang tanah dengan pencahayaan cukup, latar belakang
        jelas, dan daun terlihat utuh. Sistem akan memprediksi jenis penyakit
        menggunakan model EfficientNet-B4.
      </p>

      <div className="grid-two">
        <Card title="Unggah Citra" subtitle="Langkah 1 dari 2">
          <form onSubmit={handleSubmit} className="upload-form">
            <label className="upload-box">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
              />
              <span>
                {file ? "Ganti file citra" : "Klik untuk memilih atau drop file di sini"}
              </span>
            </label>

            {previewUrl && (
              <div className="image-preview">
                <img src={previewUrl} alt="Preview daun" />
              </div>
            )}

            {error && <p className="error-text">{error}</p>}

            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Memproses..." : "Klasifikasikan"}
            </Button>
          </form>
        </Card>

        <Card
          title="Hasil Klasifikasi"
          subtitle="Prediksi penyakit berdasarkan model CNN"
        >
          {!result && (
            <p className="placeholder">
              Hasil prediksi akan muncul di sini setelah kamu mengunggah citra dan
              menekan tombol <b>Klasifikasikan</b>.
            </p>
          )}

          {result && (
            <div className="result-block">
              <h3 className="result-label">{result.disease_label}</h3>
              <p>
                Keyakinan model:{" "}
                <b>{(result.confidence * 100).toFixed(1)}%</b>
              </p>

              <div className="prob-list">
                {Object.entries(result.probs).map(([cls, p]) => (
                  <div key={cls} className="prob-row">
                    <span>{cls}</span>
                    <span>{(p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>

              <div className="result-actions">
                <Button onClick={handleGoToAnalysis}>
                  Lihat Segmentasi & Keparahan
                </Button>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default ClassifyPage;
