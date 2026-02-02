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

  // severity display states
  const [severityPct, setSeverityPct] = useState(null);
  const [faoLevel, setFaoLevel] = useState(null);
  const [faoRange, setFaoRange] = useState(null);

  // leaf mask optional
  const [leafMaskSrc, setLeafMaskSrc] = useState("");
  const [showLeafMask, setShowLeafMask] = useState(false);

  // keep last response only for debugging (NOT shown to user)
  const [lastResponse, setLastResponse] = useState(null);

  const [saveStatus, setSaveStatus] = useState({ loading: false, msg: "", err: "" });

  const cached = useMemo(() => {
    if (!analysisId) return null;
    const raw = sessionStorage.getItem(_key(analysisId));
    return raw ? _safeParse(raw) : null;
  }, [analysisId]);

  const originalSrc = cached?.originalDataUrl || "";
  const diseaseLabel = cached?.label || cached?.result?.label || "";
  const confidence = cached?.confidence ?? cached?.result?.confidence;

  useEffect(() => {
    setError("");
    setOverlaySrc("");
    setMaskSrc("");
    setSeverityPct(null);
    setFaoLevel(null);
    setFaoRange(null);
    setLeafMaskSrc("");
    setShowLeafMask(false);
    setLastResponse(null);
  }, [analysisId]);

  const handleRunSegmentation = async () => {
    if (!analysisId) return;

    setIsLoading(true);
    setError("");

    try {
      const data = await segmentImage(analysisId);

      // simpan untuk debug transparansi penelitian (tidak ditampilkan)
      setLastResponse(data);
      // eslint-disable-next-line no-console
      console.debug("Segmentation response:", data);

      if (data?.enabled === false) {
        throw new Error(data?.reason || "Segmentasi dinonaktifkan.");
      }

      // overlay
      const overlayB64 = data?.overlay_png_base64 || data?.overlay_base64;
      if (!overlayB64) {
        throw new Error(
          "Segmentasi berhasil, tetapi overlay image tidak ditemukan pada response backend."
        );
      }
      setOverlaySrc(`data:image/png;base64,${overlayB64}`);

      // optional mask dari backend (kalau ada)
      const maskB64 = data?.mask_png_base64 || data?.mask_base64;
      if (maskB64) setMaskSrc(`data:image/png;base64,${maskB64}`);
      else setMaskSrc("");

      // severity (ini yang user lihat)
      const sev = data?.severity || null;
      if (sev) {
        // severity_pct bisa number / string
        const pct = sev?.severity_pct;
        setSeverityPct(typeof pct === "number" ? pct : (pct != null ? Number(pct) : null));

        const lvl = sev?.fao?.level;
        setFaoLevel(typeof lvl === "number" ? lvl : (lvl != null ? Number(lvl) : null));

        const rng = sev?.fao?.range_pct;
        if (Array.isArray(rng) && rng.length === 2) {
          setFaoRange([Number(rng[0]), Number(rng[1])]);
        } else {
          setFaoRange(null);
        }

        // leaf mask (opsional)
        const leafMaskB64 = sev?.leaf_mask_png_base64;
        if (leafMaskB64) {
          setLeafMaskSrc(`data:image/png;base64,${leafMaskB64}`);
        } else {
          setLeafMaskSrc("");
          setShowLeafMask(false);
        }
      } else {
        // tidak ada severity (misal backend belum hitung / belum enabled)
        setSeverityPct(null);
        setFaoLevel(null);
        setFaoRange(null);
        setLeafMaskSrc("");
        setShowLeafMask(false);
      }
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

  const severityText = (() => {
    if (severityPct == null || Number.isNaN(severityPct)) return "-";
    return `${severityPct.toFixed(2)}%`;
  })();

  const faoText = (() => {
    if (faoLevel == null || Number.isNaN(faoLevel)) return "-";
    if (Array.isArray(faoRange) && faoRange.length === 2) {
      return `Level ${faoLevel} (rentang ${faoRange[0]}–${faoRange[1]}%)`;
    }
    return `Level ${faoLevel}`;
  })();

  return (
    <div className="page page-analysis">
      <h2>Hasil Segmentasi Area Terinfeksi</h2>
      <p className="page-description">
        Halaman ini menampilkan overlay (citra asli + hasil prediksi mask) dari tahap segmentasi.
        Kamu tidak perlu mengunggah ulang citra.
      </p>

      <div className="analysis-top-meta">
        <span>ID Analisis: {analysisId}</span>
        {diseaseLabel && <span>Penyakit: {diseaseLabel}</span>}
        {typeof confidence === "number" && (
          <span>Keyakinan: {(confidence * 100).toFixed(1)}%</span>
        )}
      </div>

      <div style={{ marginTop: 12 }}>
        <Card
          title="Estimasi Keparahan Penyakit"
          subtitle="Persentase keparahan + level berdasarkan standar FAO"
        >
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>Keparahan</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{severityText}</div>
            </div>
            <div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>Level FAO</div>
              <div style={{ fontSize: 18, fontWeight: 600 }}>{faoText}</div>
            </div>
          </div>

          {leafMaskSrc && (
            <div style={{ marginTop: 12 }}>
              <Button
                onClick={() => setShowLeafMask((v) => !v)}
                style={{ fontSize: 13, padding: "8px 10px" }}
              >
                {showLeafMask ? "Sembunyikan mask daun" : "Lihat mask daun (opsional)"}
              </Button>

              {showLeafMask && (
                <div className="image-panel" style={{ marginTop: 10 }}>
                  <img src={leafMaskSrc} alt="Leaf mask (opsional)" className="image-bordered" />
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      <div className="grid-two" style={{ marginTop: 16 }}>
        <Card title="Citra Asli">
          {!originalSrc ? (
            <p className="placeholder">
              Citra asli tidak tersedia pada sesi ini. (Masih bisa lanjut segmentasi dan melihat overlay.)
            </p>
          ) : (
            <div className="image-panel">
              <img src={originalSrc} alt="Citra asli" className="image-bordered" />
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
              <img src={overlaySrc} alt="Overlay segmentasi" className="image-bordered" />
            </div>
          )}
        </Card>
      </div>

      {maskSrc && (
        <div style={{ marginTop: 16 }}>
          <Card title="Mask (Opsional)" subtitle="Jika backend mengirim mask tambahan">
            <div className="image-panel">
              <img src={maskSrc} alt="Mask opsional" className="image-bordered" />
            </div>
          </Card>
        </div>
      )}

      {error && (
        <p className="error-text" style={{ marginTop: 12 }}>
          {error}
        </p>
      )}

      <div
        className="result-actions"
        style={{ marginTop: 16, gap: 12, display: "flex", flexWrap: "wrap" }}
      >
        <Button onClick={handleRunSegmentation} disabled={isLoading || !analysisId}>
          {isLoading ? "Memproses..." : "Proses Segmentasi"}
        </Button>

        <Button onClick={handleSave} disabled={saveStatus.loading || !analysisId}>
          {saveStatus.loading ? "Menyimpan..." : "Simpan Hasil"}
        </Button>

        <Button variant="secondary" onClick={handleBack}>
          Kembali
        </Button>
      </div>

      {saveStatus.msg && (
        <p style={{ marginTop: 10, color: "green" }}>{saveStatus.msg}</p>
      )}
      {saveStatus.err && (
        <p style={{ marginTop: 10, color: "crimson" }}>{saveStatus.err}</p>
      )}

      {/* debug JSON opsional untuk transparansi penelitian */}
      {/* sengaja tidak ditampilkan ke user; tersedia di console.debug dan lastResponse state */}
      {lastResponse && null}
    </div>
  );
};

export default SegmentPage;
