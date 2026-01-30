// src/pages/ClassifyPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { classifyImage, saveAnalysis } from "../api/analysisApi";

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
    setResult(null); // clear previous result when change image
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
      // Expect structure from backend: { id, label, confidence, probs }
      setResult(data);

      // Simpan sementara agar tahap segmentasi tidak perlu upload ulang
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
    // Nanti halaman /segment/:id akan pakai id ini
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
              <input type="file" accept="image/*" onChange={handleFileChange} />
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

        <Card title="Hasil Klasifikasi" subtitle="Prediksi penyakit berdasarkan model CNN">
          {!result && !isLoading && (
            <p className="placeholder">
              Hasil prediksi akan muncul di sini setelah kamu mengunggah citra
              dan menekan tombol <b>Klasifikasikan</b>.
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
                      <div key={cls} className="prob-row">
                        <span>{cls}</span>
                        <span>{(p * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              )}

              <div className="result-actions">
                {String(result.label || "").trim().toLowerCase() === "healthy" ? (
                  <>
                    <Button onClick={handleSave} disabled={saveStatus.loading}>
                      {saveStatus.loading ? "Menyimpan..." : "Simpan Hasil"}
                    </Button>
                    {saveStatus.msg && <p className="success-text">{saveStatus.msg}</p>}
                    {saveStatus.err && <p className="error-text">{saveStatus.err}</p>}
                  </>
                ) : (
                  <Button onClick={handleGoToSegment}>Lihat Segmentasi</Button>
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
