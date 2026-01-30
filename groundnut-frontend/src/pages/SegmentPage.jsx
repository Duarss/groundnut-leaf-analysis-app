// src/pages/SegmentPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { segmentImage, saveAnalysis } from "../api/analysisApi";

// sessionStorage keys
const _key = (analysisId) => `analysis_session_${analysisId}`;

function _safeParse(jsonStr) {
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    void e;
    return null;
  }
}

const SegmentPage = () => {
  const { id } = useParams(); // route: /segment/:id
  const analysisId = id;
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [overlaySrc, setOverlaySrc] = useState("");
  const [maskSrc, setMaskSrc] = useState("");
  const [meta, setMeta] = useState(null);
  const [saveStatus, setSaveStatus] = useState({ loading: false, msg: "", err: "" });

  const cached = useMemo(() => {
    if (!analysisId) return null;
    const raw = sessionStorage.getItem(_key(analysisId));
    return raw ? _safeParse(raw) : null;
  }, [analysisId]);

  const originalSrc = cached?.originalDataUrl || "";
  const diseaseLabel =
    cached?.label || cached?.result?.label || "";
  const confidence = cached?.confidence ?? cached?.result?.confidence;

  useEffect(() => {
    setError("");
    setOverlaySrc("");
    setMaskSrc("");
    setMeta(null);
  }, [analysisId]);

  const handleRunSegmentation = async () => {
    if (!analysisId) return;

    setIsLoading(true);
    setError("");

    try {
      const data = await segmentImage(analysisId);

      if (data?.enabled === false) {
        throw new Error(data?.reason || "Segmentasi dinonaktifkan.");
      }

      // toleransi key jika backend beda penamaan
      const overlayB64 = data?.overlay_png_base64 || data?.overlay_base64;
      const maskB64 = data?.mask_png_base64 || data?.mask_base64;

      if (!overlayB64) {
        throw new Error(
          "Segmentasi berhasil, tetapi overlay image tidak ditemukan pada response backend."
        );
      }

      setOverlaySrc(`data:image/png;base64,${overlayB64}`);
      if (maskB64) setMaskSrc(`data:image/png;base64,${maskB64}`);
      setMeta(data?.meta || null);
    } catch (err) {
      setError(err?.message || "Terjadi kesalahan saat segmentasi.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => navigate("/classify");

  const handleSave = async () => {
    if (!analysisId) return;
    setSaveStatus({ loading: true, msg: "", err: "" });
    try {
      const res = await saveAnalysis(analysisId);
      if (!res?.saved) throw new Error(res?.error || "Gagal menyimpan hasil analisis.");
      setSaveStatus({ loading: false, msg: "Hasil analisis berhasil disimpan.", err: "" });
    } catch (e) {
      setSaveStatus({ loading: false, msg: "", err: e.message || "Gagal menyimpan." });
    }
  };

  return (
    <div className="page page-analysis">
      <h2>Hasil Segmentasi Area Terinfeksi</h2>
      <p className="page-description">
        Halaman ini menampilkan overlay (citra asli + hasil prediksi mask) dari
        tahap segmentasi. Kamu tidak perlu mengunggah ulang citra.
      </p>

      <div className="analysis-top-meta">
        <span>ID Analisis: {analysisId}</span>
        {diseaseLabel && <span>Penyakit: {diseaseLabel}</span>}
        {typeof confidence === "number" && (
          <span>Keyakinan: {(confidence * 100).toFixed(1)}%</span>
        )}
      </div>

      <div className="grid-two">
        <Card title="Citra Asli">
          {!originalSrc ? (
            <p className="placeholder">
              Citra asli tidak tersedia pada sesi ini. (Masih bisa lanjut
              segmentasi dan melihat overlay.)
            </p>
          ) : (
            <div className="image-panel">
              <img
                src={originalSrc}
                alt="Citra asli"
                className="image-bordered"
              />
            </div>
          )}
        </Card>

        <Card title="Overlay Segmentasi" subtitle="Citra asli + mask prediksi">
          {!overlaySrc && !isLoading && (
            <p className="placeholder">
              Klik tombol <b>Proses Segmentasi</b> untuk menghasilkan overlay.
            </p>
          )}

          {isLoading && <p className="placeholder">Sedang memproses...</p>}

          {overlaySrc && (
            <div className="image-panel">
              <img
                src={overlaySrc}
                alt="Overlay segmentasi"
                className="image-bordered"
              />
            </div>
          )}
        </Card>
      </div>

      {maskSrc && (
        <div style={{ marginTop: 16 }}>
          <Card title="Mask (Opsional)" subtitle="Jika backend mengirim mask">
            <div className="image-panel">
              <img src={maskSrc} alt="Mask" className="image-bordered" />
            </div>
          </Card>
        </div>
      )}

      {meta && (
        <div style={{ marginTop: 16 }}>
          <Card title="Metadata" subtitle="Informasi tambahan dari backend">
            <pre style={{ overflowX: "auto", margin: 0 }}>
              {JSON.stringify(meta, null, 2)}
            </pre>
          </Card>
        </div>
      )}

      {error && (
        <p className="error-text" style={{ marginTop: 12 }}>
          {error}
        </p>
      )}

      <div className="result-actions" style={{ marginTop: 16, gap: 12, display: "flex", flexWrap: "wrap" }}>
        <Button onClick={handleRunSegmentation} disabled={isLoading || !analysisId}>
          {isLoading ? "Memproses..." : "Proses Segmentasi"}
        </Button>

        <Button onClick={handleBack} type="button">
          Kembali
        </Button>

        {overlaySrc && (
          <Button onClick={handleSave} disabled={saveStatus.loading} type="button">
            {saveStatus.loading ? "Menyimpan..." : "Simpan Hasil"}
          </Button>
        )}
      </div>
    </div>
  );
};

export default SegmentPage;
