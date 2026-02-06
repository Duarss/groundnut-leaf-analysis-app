// src/pages/SegmentPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { segmentImage, saveAnalysis } from "../api/analysisApi";
import { useIsMobile } from "../utils/useIsMobile";
import ImageBox from "../components/ui/ImageBox";

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
  const isMobile = useIsMobile();
  const { id } = useParams();
  const analysisId = id;
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const [originalSrc, setOriginalSrc] = useState(""); // ✅ sekarang di-load dari backend tmp
  const [overlaySrc, setOverlaySrc] = useState("");
  const [maskSrc, setMaskSrc] = useState("");

  const [severityPct, setSeverityPct] = useState(null);
  const [faoLevel, setFaoLevel] = useState(null);
  const [faoRange, setFaoRange] = useState(null);

  const [leafMaskSrc, setLeafMaskSrc] = useState("");
  const [showLeafMask, setShowLeafMask] = useState(false);

  const [saveStatus, setSaveStatus] = useState({ loading: false, msg: "", err: "" });

  const cached = useMemo(() => {
    if (!analysisId) return null;
    const raw = sessionStorage.getItem(_key(analysisId));
    return raw ? _safeParse(raw) : null;
  }, [analysisId]);

  const diseaseLabel = cached?.label || cached?.result?.label || "";
  const confidence = cached?.confidence ?? cached?.result?.confidence;

  // ✅ ambil citra original langsung dari tmp_uploads via backend
  useEffect(() => {
    if (!analysisId) return;

    let alive = true;
    let objectUrl = "";

    async function loadOriginal() {
      try {
        // reset dulu
        setOriginalSrc("");

        const res = await fetch(`/api/temp-image/${encodeURIComponent(analysisId)}`, {
          method: "GET",
        });

        if (!res.ok) {
          // kalau tidak ada file temp (misal TTL habis), biarkan kosong
          return;
        }

        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        if (alive) setOriginalSrc(objectUrl);
      } catch (e) {
        void e;
        // silent fail → fallback tetap kosong
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
    setFaoLevel(null);
    setFaoRange(null);
    setLeafMaskSrc("");
    setShowLeafMask(false);
    setSaveStatus({ loading: false, msg: "", err: "" });
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
        const pct = sev?.severity_pct;
        setSeverityPct(typeof pct === "number" ? pct : pct != null ? Number(pct) : null);

        const lvl = sev?.fao?.level;
        setFaoLevel(typeof lvl === "number" ? lvl : lvl != null ? Number(lvl) : null);

        const rng = sev?.fao?.range_pct;
        if (Array.isArray(rng) && rng.length === 2) setFaoRange([Number(rng[0]), Number(rng[1])]);
        else setFaoRange(null);

        const leafMaskB64 = sev?.leaf_mask_png_base64;
        if (leafMaskB64) setLeafMaskSrc(`data:image/png;base64,${leafMaskB64}`);
        else {
          setLeafMaskSrc("");
          setShowLeafMask(false);
        }
      } else {
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
    if (Array.isArray(faoRange) && faoRange.length === 2)
      return `Level ${faoLevel} (rentang ${faoRange[0]}–${faoRange[1]}%)`;
    return `Level ${faoLevel}`;
  })();

  const canShowSave = Boolean(overlaySrc);

  return (
    <div className="page page-analysis">
      <h2>Hasil Segmentasi Area Terinfeksi</h2>
      <p className="page-description">
        Halaman ini menampilkan overlay (citra asli + hasil prediksi mask) dari tahap segmentasi.
      </p>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <span>ID Analisis: {analysisId}</span>
        {diseaseLabel && <span>Penyakit: {diseaseLabel}</span>}
        {typeof confidence === "number" && <span>Keyakinan: {(confidence * 100).toFixed(1)}%</span>}
      </div>

      <div style={{ marginTop: 12 }}>
        <Card title="Estimasi Keparahan Penyakit" subtitle="Persentase keparahan + level berdasarkan standar FAO">
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
        <Card title="Citra Asli">
          {!originalSrc ? (
            <p className="placeholder">Citra asli tidak tersedia (mungkin sudah kadaluarsa / belum ada di tmp_uploads).</p>
          ) : (
            <ImageBox
              src={originalSrc}
              alt="Citra asli"
              isMobile={isMobile}
              maxHeightMobile={240}
              maxHeightDesktop={420}
              allowExpand={true}
            />
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
            <ImageBox
              src={overlaySrc}
              alt="Overlay segmentasi"
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

      {canShowSave && saveStatus.msg && <p style={{ marginTop: 10, color: "green" }}>{saveStatus.msg}</p>}
      {canShowSave && saveStatus.err && <p style={{ marginTop: 10, color: "crimson" }}>{saveStatus.err}</p>}
    </div>
  );
};

export default SegmentPage;
