// src/pages/SegmentPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import ImageBox from "../components/ui/ImageBox";
import Toast from "../components/ui/Toast";
import { segmentImage, saveAnalysis } from "../api/analysisApi";
import { useIsMobile } from "../utils/useIsMobile";

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

function _numOrNull(v) {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

const SegmentPage = () => {
  const isMobile = useIsMobile();
  const { id } = useParams();
  const analysisId = id;
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const [originalSrc, setOriginalSrc] = useState("");
  const [overlaySrc, setOverlaySrc] = useState("");
  const [maskSrc, setMaskSrc] = useState("");

  // ===== SAD (HB) states =====
  const [severityPct, setSeverityPct] = useState(null);
  const [sadScheme, setSadScheme] = useState("");
  const [sadClassIndex, setSadClassIndex] = useState(null);
  const [sadRange, setSadRange] = useState(null);

  const [leafMaskSrc, setLeafMaskSrc] = useState("");
  const [showLeafMask, setShowLeafMask] = useState(false);

  const [saveStatus, setSaveStatus] = useState({ loading: false, msg: "", err: "" });
  const [toast, setToast] = useState({ open: false, type: "info", message: "" });

  const cached = useMemo(() => {
    if (!analysisId) return null;
    const raw = sessionStorage.getItem(_key(analysisId));
    return raw ? _safeParse(raw) : null;
  }, [analysisId]);

  const diseaseLabel = cached?.label || cached?.result?.label || "";
  const confidence = cached?.confidence ?? cached?.result?.confidence;

  // ===== UI helper (SAMA seperti ClassifyPage.jsx) =====
  const pill = (tone = "neutral") => {
    const map = {
      neutral: { bg: "#f3f4f6", fg: "#374151", bd: "#e5e7eb" },
      good: { bg: "#ecfdf5", fg: "#065f46", bd: "#a7f3d0" },
      warn: { bg: "#fffbeb", fg: "#92400e", bd: "#fde68a" },
      dark: { bg: "#111827", fg: "#ffffff", bd: "#111827" },
    };
    const t = map[tone] || map.neutral;
    return {
      display: "inline-flex",
      alignItems: "center",
      padding: "6px 10px",
      borderRadius: 999,
      border: `1px solid ${t.bd}`,
      background: t.bg,
      color: t.fg,
      fontWeight: 800,
      fontSize: 12,
      whiteSpace: "nowrap",
    };
  };

  useEffect(() => {
    if (!analysisId) return;

    let alive = true;
    let objectUrl = "";

    async function loadOriginal() {
      try {
        setOriginalSrc("");

        const res = await fetch(`/api/temp-image/${encodeURIComponent(analysisId)}`, {
          method: "GET",
        });

        if (!res.ok) return;

        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        if (alive) setOriginalSrc(objectUrl);
      } catch (e) {
        void e;
      }
    }

    loadOriginal();

    return () => {
      alive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [analysisId]);

  useEffect(() => {
    setError("");
    setOverlaySrc("");
    setMaskSrc("");

    setSeverityPct(null);
    setSadScheme("");
    setSadClassIndex(null);
    setSadRange(null);

    setLeafMaskSrc("");
    setShowLeafMask(false);
    setSaveStatus({ loading: false, msg: "", err: "" });
    setToast({ open: false, type: "info", message: "" });
  }, [analysisId]);

  const handleRunSegmentation = async () => {
    if (!analysisId) return;

    setIsLoading(true);
    setError("");
    setSaveStatus({ loading: false, msg: "", err: "" });

    try {
      const data = await segmentImage(analysisId);

      if (data?.enabled === false) {
        throw new Error(data?.reason || "Segmentasi dinonaktifkan.");
      }

      const overlayB64 = data?.overlay_png_base64 || data?.overlay_base64;
      if (!overlayB64) {
        throw new Error("Segmentasi berhasil, tetapi overlay image tidak ditemukan pada response backend.");
      }
      setOverlaySrc(`data:image/png;base64,${overlayB64}`);

      const maskB64 = data?.mask_png_base64 || data?.mask_base64;
      if (maskB64) setMaskSrc(`data:image/png;base64,${maskB64}`);
      else setMaskSrc("");

      const sev = data?.severity || null;
      if (sev) {
        const pct = _numOrNull(sev?.severity_pct);
        setSeverityPct(pct);

        const sad = sev?.sad || sev?.SAD || null;

        const scheme = (sad?.scheme || "").toString();
        setSadScheme(scheme);

        const ci = _numOrNull(sad?.class_index ?? sad?.class ?? sad?.classId);
        setSadClassIndex(ci);

        const rng = sad?.range_pct;
        if (Array.isArray(rng) && rng.length === 2) setSadRange([Number(rng[0]), Number(rng[1])]);
        else setSadRange(null);

        const leafMaskB64 = sev?.leaf_mask_png_base64;
        if (leafMaskB64) setLeafMaskSrc(`data:image/png;base64,${leafMaskB64}`);
        else {
          setLeafMaskSrc("");
          setShowLeafMask(false);
        }
      } else {
        setSeverityPct(null);
        setSadScheme("");
        setSadClassIndex(null);
        setSadRange(null);
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
      setSaveStatus({ loading: false, msg: "", err: "" });

      navigate("/history", {
        state: {
          toast: { type: "success", message: "Hasil analisis berhasil disimpan." },
        },
      });
    } catch (e) {
      const msg = e?.message || "Gagal menyimpan hasil analisis.";
      setSaveStatus({ loading: false, msg: "", err: msg });
      setToast({ open: true, type: "error", message: msg });
    }
  };

  const severityText = useMemo(() => {
    if (severityPct == null || Number.isNaN(severityPct)) return "-";
    return `${severityPct.toFixed(2)}%`;
  }, [severityPct]);

  const sadText = useMemo(() => {
    if (sadClassIndex == null || Number.isNaN(sadClassIndex)) return "-";
    const cls = `Kelas SAD ${Math.round(sadClassIndex)}`;
    const range = Array.isArray(sadRange) && sadRange.length === 2 ? ` (rentang ${sadRange[0]}–${sadRange[1]}%)` : "";
    return `${cls}${range}`;
  }, [sadClassIndex, sadRange]);

  const canShowSave = Boolean(overlaySrc);

  return (
    <div className="page page-analysis">
      <Toast
        open={toast.open}
        type={toast.type}
        message={toast.message}
        onClose={() => setToast((v) => ({ ...v, open: false }))}
      />

      <h2>Hasil Segmentasi Area Terinfeksi</h2>
      <p className="page-description">
        Halaman ini menampilkan overlay (citra asli + hasil prediksi mask) dari tahap segmentasi.
      </p>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <span>ID: {analysisId}</span>
        {diseaseLabel && <span>Penyakit: {diseaseLabel}</span>}
        {typeof confidence === "number" && <span>Keyakinan: {(confidence * 100).toFixed(1)}%</span>}
      </div>

      <div style={{ marginTop: 12 }}>
        <Card
          title="Estimasi Keparahan Penyakit"
          subtitle={sadScheme ? `Persentase keparahan + pemetaan SAD (${sadScheme})` : "Persentase keparahan + pemetaan SAD"}
        >
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>Keparahan</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{severityText}</div>
            </div>
            <div>
              <div style={{ fontSize: 13, opacity: 0.8 }}>Standard Area Diagrams (SAD)</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{sadText}</div>
            </div>
          </div>

          {leafMaskSrc && (
            <div style={{ marginTop: 12 }}>
              <Button onClick={() => setShowLeafMask((v) => !v)} style={{ width: isMobile ? "100%" : undefined }}>
                {showLeafMask ? "Sembunyikan mask daun" : "Lihat mask daun (opsional)"}
              </Button>

              {showLeafMask && (
                <div style={{ marginTop: 10 }}>
                  <ImageBox
                    src={leafMaskSrc}
                    alt="Leaf mask"
                    isMobile={isMobile}
                    maxHeightMobile={220}
                    maxHeightDesktop={360}
                    allowExpand={true}
                  />
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      <div
        style={{
          marginTop: 16,
          display: "grid",
          gridTemplateColumns: isMobile ? "1fr" : "repeat(2, minmax(0, 1fr))",
          gap: 16,
        }}
      >
        <Card title="Gambar Asli">
          {!originalSrc ? (
            <p className="placeholder">Gambar asli tidak tersedia (mungkin sudah kadaluarsa / belum ada di tmp_uploads).</p>
          ) : (
            <ImageBox
              src={originalSrc}
              alt="Gambar asli"
              isMobile={isMobile}
              maxHeightMobile={240}
              maxHeightDesktop={420}
              allowExpand={true}
            />
          )}
        </Card>

        <Card title="Visualisasi Area Terinfeksi" subtitle="Gambar asli + prediksi area terinfeksi">
          {!overlaySrc && !isLoading && (
            <p className="placeholder">
              Klik tombol <b>Proses Segmentasi</b> untuk menghasilkan overlay.
            </p>
          )}

          {/* Loading */}
          {!overlaySrc && isLoading && (
            <div style={{ padding: 12, border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                <div style={{ fontWeight: 900 }}>Sedang memproses...</div>
                <span style={pill("dark")}>Running</span>
              </div>
              <div style={{ marginTop: 8, fontSize: 13, color: "#6b7280", lineHeight: 1.5 }}>
                Mohon tunggu, sistem sedang melakukan inferensi model segmentasi.
              </div>
              <div style={{ marginTop: 10, height: 10, borderRadius: 999, background: "#e5e7eb", overflow: "hidden" }}>
                <div
                  style={{
                    width: "60%",
                    height: "100%",
                    background: "#111827",
                    borderRadius: 999,
                    animation: "pulse 1.2s ease-in-out infinite",
                  }}
                />
              </div>
            </div>
          )}

          {overlaySrc && (
            <ImageBox
              src={overlaySrc}
              alt="Visualisasi area terinfeksi"
              isMobile={isMobile}
              maxHeightMobile={240}
              maxHeightDesktop={420}
              allowExpand={true}
            />
          )}
        </Card>
      </div>

      {maskSrc && (
        <div style={{ marginTop: 16 }}>
          <Card title="Mask (Opsional)">
            <ImageBox
              src={maskSrc}
              alt="Mask"
              isMobile={isMobile}
              maxHeightMobile={220}
              maxHeightDesktop={360}
              allowExpand={true}
            />
          </Card>
        </div>
      )}

      {error && (
        <p className="error-text" style={{ marginTop: 12 }}>
          {error}
        </p>
      )}

      <div style={{ marginTop: 16, gap: 10, display: "flex", flexWrap: "wrap" }}>
        <Button
          onClick={handleRunSegmentation}
          disabled={isLoading || !analysisId}
          style={isMobile ? { width: "100%" } : undefined}
        >
          {isLoading ? "Memproses..." : "Proses Segmentasi"}
        </Button>

        {canShowSave && (
          <Button
            onClick={handleSave}
            disabled={saveStatus.loading || !analysisId}
            style={isMobile ? { width: "100%" } : undefined}
          >
            {saveStatus.loading ? "Menyimpan..." : "Simpan Hasil"}
          </Button>
        )}

        <Button variant="secondary" onClick={handleBack} style={isMobile ? { width: "100%" } : undefined}>
          Kembali
        </Button>
      </div>

      {/* Loading anim keyframes (inline) */}
      <style>{`
        @keyframes pulse {
          0% { transform: translateX(-20%); opacity: .65; }
          50% { transform: translateX(10%); opacity: 1; }
          100% { transform: translateX(-20%); opacity: .65; }
        }
      `}</style>
    </div>
  );
};

export default SegmentPage;
