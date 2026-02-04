// src/pages/ClassifyPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { classifyImage, saveAnalysis } from "../api/analysisApi";
import { useIsMobile } from "../utils/useIsMobile";
import ImageBox from "../components/ui/ImageBox";

const _sessionKey = (analysisId) => `analysis_session_${analysisId}`;

function _fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Gagal membaca file gambar."));
    reader.readAsDataURL(file);
  });
}

const ClassifyPage = () => {
  const isMobile = useIsMobile();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [saveStatus, setSaveStatus] = useState({ loading: false, msg: "", err: "" });
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    setError("");
    setResult(null);
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
    setResult(null);

    try {
      const data = await classifyImage(file);
      setResult(data);

      try {
        const originalDataUrl = await _fileToDataUrl(file);
        sessionStorage.setItem(
          _sessionKey(data.analysis_id),
          JSON.stringify({
            analysis_id: data.analysis_id,
            label: data.label,
            confidence: data.confidence,
            probs: data.probs,
            originalDataUrl,
          })
        );
      } catch (e) {
        void e;
      }
    } catch (err) {
      setError(err.message || "Terjadi kesalahan saat klasifikasi.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoToSegment = () => {
    const analysisId = result?.analysis_id || result?.id;
    if (!analysisId) return;
    navigate(`/segment/${analysisId}`);
  };

  const handleSave = async () => {
    const analysisId = result?.analysis_id || result?.id;
    if (!analysisId) return;

    setSaveStatus({ loading: true, msg: "", err: "" });
    try {
      const res = await saveAnalysis(analysisId);
      if (!res?.saved) throw new Error(res?.error || "Gagal menyimpan hasil analisis.");
      setSaveStatus({ loading: false, msg: "Hasil analisis berhasil disimpan.", err: "" });
    } catch (e) {
      setSaveStatus({ loading: false, msg: "", err: e.message || "Gagal menyimpan hasil analisis." });
    }
  };

  const gridStyle = {
    display: "grid",
    gridTemplateColumns: isMobile ? "1fr" : "repeat(2, minmax(0, 1fr))",
    gap: 16,
    alignItems: "start",
  };

  return (
    <div className="page page-classify">
      <h2>Klasifikasi Penyakit Daun Kacang Tanah</h2>
      <p className="page-description">
        Unggah citra daun kacang tanah. Sistem akan memprediksi jenis penyakit menggunakan model EfficientNet-B4.
      </p>

      <div className="grid-two" style={gridStyle}>
        <Card title="Unggah Citra" subtitle="Langkah 1 dari 2">
          <form onSubmit={handleSubmit} className="upload-form">
            <label className="upload-box" style={{ display: "block" }}>
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <span>{file ? "Ganti file citra" : "Klik untuk memilih atau drop file di sini"}</span>
            </label>

            {/* ✅ mobile-friendly preview */}
            {previewUrl && (
              <div style={{ marginTop: 12 }}>
                <ImageBox
                  src={previewUrl}
                  alt="Preview daun"
                  isMobile={isMobile}
                  maxHeightMobile={220}
                  maxHeightDesktop={420}
                  allowExpand={true}
                />
              </div>
            )}

            {error && <p className="error-text">{error}</p>}

            <Button type="submit" disabled={isLoading} style={isMobile ? { width: "100%" } : undefined}>
              {isLoading ? "Memproses..." : "Klasifikasikan"}
            </Button>
          </form>
        </Card>

        <Card title="Hasil Klasifikasi" subtitle="Prediksi penyakit berdasarkan model CNN">
          {!result && !isLoading && (
            <p className="placeholder">
              Hasil prediksi akan muncul setelah kamu mengunggah citra dan menekan <b>Klasifikasikan</b>.
            </p>
          )}

          {isLoading && <p className="placeholder">Model sedang memproses citra...</p>}

          {result && !isLoading && (
            <div className="result-block">
              <h3 className="result-label">{result.label}</h3>
              <p>
                Keyakinan model: <b>{(result.confidence * 100).toFixed(1)}%</b>
              </p>

              {result.probs && (
                <div className="prob-list">
                  {Object.entries(result.probs)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, p]) => (
                      <div
                        key={cls}
                        className="prob-row"
                        style={{ display: "flex", justifyContent: "space-between" }}
                      >
                        <span>{cls}</span>
                        <span>{(p * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              )}

              <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 12 }}>
                {String(result.label || "").trim().toLowerCase() === "healthy" ? (
                  <>
                    <Button onClick={handleSave} disabled={saveStatus.loading} style={isMobile ? { width: "100%" } : undefined}>
                      {saveStatus.loading ? "Menyimpan..." : "Simpan Hasil"}
                    </Button>
                    {saveStatus.msg && <p className="success-text" style={{ margin: 0 }}>{saveStatus.msg}</p>}
                    {saveStatus.err && <p className="error-text" style={{ margin: 0 }}>{saveStatus.err}</p>}
                  </>
                ) : (
                  <Button onClick={handleGoToSegment} style={isMobile ? { width: "100%" } : undefined}>
                    Lihat Area Terinfeksi & Estimasi Keparahan
                  </Button>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default ClassifyPage;
