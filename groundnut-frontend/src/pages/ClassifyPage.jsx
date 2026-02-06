// src/pages/ClassifyPage.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { classifyImage, saveAnalysis } from "../api/analysisApi";
import { useIsMobile } from "../utils/useIsMobile";
import ImageBox from "../components/ui/ImageBox";

// ====== Disease description (shown to user instead of probs_json) ======
// Catatan: link sumber tetap di komentar (untuk catatan ilmiah).
// - Groundnut rosette disease (CABI): https://www.cabi.org/isc/datasheet/22097
// - Peanut rust (UF/IFAS EDIS): https://edis.ifas.ufl.edu/publication/PP288
// - Peanut leaf spots (early & late) review: https://www.mdpi.com/2076-2607/11/8/2158
const DISEASE_INFO = {
  HEALTHY: {
    title: "HEALTHY (Daun Sehat)",
    short:
      "Daun tampak normal tanpa gejala bercak/lesi khas penyakit. Tetap lakukan perawatan budidaya yang baik (monitoring rutin, nutrisi seimbang, sanitasi lahan).",
    sources: [{ label: "Catatan umum budidaya", url: "" }],
  },
  "ALTERNARIA LEAF SPOT": {
    title: "ALTERNARIA LEAF SPOT",
    short:
      "Penyakit jamur yang umumnya memunculkan bercak nekrotik cokelat–kehitaman, kadang dengan pola konsentris/‘target spot’. Bercak dapat melebar dan menyebabkan klorosis/kerontokan daun pada infeksi berat.",
    sources: [{ label: "Literatur Alternaria pada groundnut", url: "" }],
  },
  "LEAF SPOT (EARLY AND LATE)": {
    title: "LEAF SPOT (EARLY & LATE)",
    short:
      "Kelompok penyakit bercak daun pada kacang tanah (early leaf spot & late leaf spot) yang menyebabkan bercak pada daun, mengurangi luas fotosintesis, dan pada kasus berat memicu defoliasi sehingga menurunkan hasil.",
    sources: [{ label: "Review ilmiah leaf spot", url: "https://www.mdpi.com/2076-2607/11/8/2158" }],
  },
  ROSETTE: {
    title: "ROSETTE",
    short:
      "Groundnut rosette disease adalah penyakit virus kompleks (sering terkait vektor aphid) yang menyebabkan pertumbuhan kerdil, rosetting (daun mengumpul/berbentuk roset), klorosis, dan penurunan hasil yang signifikan.",
    sources: [{ label: "CABI datasheet", url: "https://www.cabi.org/isc/datasheet/22097" }],
  },
  RUST: {
    title: "RUST",
    short:
      "Peanut rust disebabkan jamur (Puccinia arachidis) dengan gejala pustula berwarna cokelat-oranye (karat) terutama di permukaan bawah daun. Infeksi berat dapat menyebabkan daun menguning dan rontok.",
    sources: [{ label: "UF/IFAS EDIS", url: "https://edis.ifas.ufl.edu/publication/PP288" }],
  },
};

const SHOW_DEBUG_PROBS = false;

function _normLabelForInfo(label) {
  const s = String(label || "").trim().toUpperCase();
  if (!s) return "";
  if (s === "LEAF SPOT" || s === "LEAFSPOT") return "LEAF SPOT (EARLY AND LATE)";
  return s;
}

// helper: tunggu event/condition dengan timeout
function waitForEvent(target, eventName, timeoutMs = 2500) {
  return new Promise((resolve) => {
    let done = false;
    const onDone = (ok) => {
      if (done) return;
      done = true;
      try {
        target.removeEventListener(eventName, onEvt);
      } catch {
        // ignore
      }
      resolve(ok);
    };
    const onEvt = () => onDone(true);
    try {
      target.addEventListener(eventName, onEvt, { once: true });
    } catch {
      onDone(false);
      return;
    }
    setTimeout(() => onDone(false), timeoutMs);
  });
}

const ClassifyPage = () => {
  const isMobile = useIsMobile();
  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");

  // UI mode: upload vs camera
  const [inputMode, setInputMode] = useState("upload"); // "upload" | "camera"

  // Camera state
  const [cameraPanelOpen, setCameraPanelOpen] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraErr, setCameraErr] = useState("");
  const [cameraBusy, setCameraBusy] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const captureInputRef = useRef(null);

  // Result state
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [saveStatus, setSaveStatus] = useState({ loading: false, msg: "", err: "" });

  // ✅ NEW: processing state for "Proses Klasifikasi" button label
  const [classifyBusy, setClassifyBusy] = useState(false);

  // Deskripsi penyakit ditampilkan ke user
  const diseaseInfo = useMemo(() => {
    const key = _normLabelForInfo(result?.label);
    return DISEASE_INFO[key] || null;
  }, [result?.label]);

  const stopCamera = () => {
    try {
      const s = streamRef.current;
      if (s) s.getTracks().forEach((t) => t.stop());
    } catch {
      // ignore
    }
    streamRef.current = null;
    setCameraOn(false);
  };

  // cleanup unmount
  useEffect(() => {
    return () => {
      stopCamera();
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return "";
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSetSelectedFile = (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);
    setError("");
    setResult(null);
    setSaveStatus({ loading: false, msg: "", err: "" });

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    handleSetSelectedFile(selectedFile);
  };

  // === Single action: "Buka Kamera"
  // Jika getUserMedia ada -> buka panel & start stream
  // Jika tidak ada / gagal -> fallback ke input capture device (tetap dari tombol yang sama)
  const openCamera = async () => {
    setCameraErr("");
    setCameraBusy(true);

    try {
      stopCamera();

      const canGUM = !!navigator.mediaDevices?.getUserMedia && window.isSecureContext;
      if (!canGUM) {
        if (captureInputRef.current) {
          captureInputRef.current.click();
        } else {
          setCameraErr("Kamera tidak tersedia di browser ini.");
        }
        return;
      }

      setCameraPanelOpen(true);
    } catch (e) {
      setCameraErr(e?.message || "Gagal membuka kamera.");
    } finally {
      setCameraBusy(false);
    }
  };

  const closeCamera = () => {
    stopCamera();
    setCameraPanelOpen(false);
    setCameraErr("");
  };

  // Start stream ketika panel kamera dibuka
  useEffect(() => {
    let cancelled = false;

    async function startStreamWhenReady() {
      if (!cameraPanelOpen) return;

      setCameraErr("");
      setCameraBusy(true);

      try {
        const v = videoRef.current;
        if (!v) {
          await new Promise((r) => setTimeout(r, 0));
        }
        const v2 = videoRef.current;
        if (!v2) {
          setCameraErr("Komponen video belum siap. Coba tutup-buka panel Kamera.");
          setCameraPanelOpen(false);
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (cancelled) {
          try {
            stream.getTracks().forEach((t) => t.stop());
          } catch {
            // ignore
          }
          return;
        }

        streamRef.current = stream;
        v2.srcObject = stream;

        await waitForEvent(v2, "loadedmetadata", 2500);
        await v2.play().catch(() => {});

        if (!cancelled) setCameraOn(true);

        const okSize = (v2.videoWidth || 0) > 0 && (v2.videoHeight || 0) > 0;
        if (!okSize && !cancelled) {
          setCameraErr("Kamera terbuka, tapi video belum siap. Coba tutup lalu buka lagi.");
        }
      } catch (e) {
        const msg = e?.message || "Gagal membuka kamera.";
        setCameraErr(msg);

        try {
          if (captureInputRef.current) captureInputRef.current.click();
        } catch {
          // ignore
        }

        stopCamera();
        setCameraPanelOpen(false);
      } finally {
        if (!cancelled) setCameraBusy(false);
      }
    }

    startStreamWhenReady();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraPanelOpen]);

  const captureFromCamera = async () => {
    setCameraErr("");
    setCameraBusy(true);

    try {
      const v = videoRef.current;
      if (!v) throw new Error("Komponen video belum siap. Coba tutup-buka panel Kamera.");

      const w = v.videoWidth || 0;
      const h = v.videoHeight || 0;
      if (w <= 0 || h <= 0) throw new Error("Video belum siap untuk capture. Coba tunggu sebentar lalu coba lagi.");

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(v, 0, 0, w, h);

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
      if (!blob) throw new Error("Gagal mengambil foto dari kamera.");

      const f = new File([blob], `camera_${Date.now()}.jpg`, { type: "image/jpeg" });
      handleSetSelectedFile(f);

      closeCamera();
      setInputMode("upload");
    } catch (e) {
      setCameraErr(e?.message || "Gagal capture foto.");
    } finally {
      setCameraBusy(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);
    setSaveStatus({ loading: false, msg: "", err: "" });

    if (!file) {
      setError("Silakan pilih gambar terlebih dahulu.");
      return;
    }

    setClassifyBusy(true);
    try {
      const data = await classifyImage(file);
      setResult(data);
    } catch (err) {
      setError(err?.message || "Gagal memproses klasifikasi.");
    } finally {
      setClassifyBusy(false);
    }
  };

  const handleGoToSegment = () => {
    if (!result?.analysis_id) return;
    navigate(`/segment/${result.analysis_id}`);
  };

  const handleSave = async () => {
    if (!result?.analysis_id) return;

    setSaveStatus({ loading: true, msg: "", err: "" });
    try {
      const data = await saveAnalysis(result.analysis_id);
      setSaveStatus({ loading: false, msg: data?.message || "Berhasil disimpan.", err: "" });
    } catch (e) {
      setSaveStatus({ loading: false, msg: "", err: e?.message || "Gagal menyimpan." });
    }
  };

  // UI helpers
  const modeBtnStyle = (active) => ({
    flex: 1,
    justifyContent: "center",
    border: active ? "2px solid #111827" : "1px solid #e5e7eb",
    background: active ? "#111827" : "#fff",
    color: active ? "#fff" : "#111827",
  });

  const panelStyle = {
    border: "1px solid #e5e7eb",
    borderRadius: 14,
    padding: 12,
    background: "#fff",
  };

  const kvRow = {
    display: "flex",
    justifyContent: "space-between",
    gap: 12,
    padding: "10px 12px",
    border: "1px solid #e5e7eb",
    borderRadius: 12,
    background: "#fff",
  };

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

  return (
    <div className="page">
      <div className="container">
        <Card title="Klasifikasi Penyakit Daun Kacang Tanah" subtitle="Unggah citra daun untuk memulai analisis">
          <div className="grid-2">
            <div>
              <Card title="Input Citra" subtitle="Langkah 1 dari 2">
                <form onSubmit={handleSubmit} className="upload-form">
                  {/* hidden capture input (fallback kamera device) */}
                  <input
                    ref={captureInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={handleFileChange}
                    style={{ display: "none" }}
                  />

                  {/* Mode selector */}
                  <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                    <Button
                      type="button"
                      onClick={() => {
                        setInputMode("upload");
                        closeCamera();
                      }}
                      style={modeBtnStyle(inputMode === "upload")}
                    >
                      Upload Foto
                    </Button>
                    <Button
                      type="button"
                      onClick={() => {
                        setInputMode("camera");
                      }}
                      style={modeBtnStyle(inputMode === "camera")}
                    >
                      Kamera
                    </Button>
                  </div>

                  {/* Upload panel */}
                  {inputMode === "upload" && (
                    <div style={panelStyle}>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
                        <label
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 10,
                            padding: "10px 12px",
                            borderRadius: 12,
                            border: "1px dashed #cbd5e1",
                            cursor: "pointer",
                            background: "#f8fafc",
                            flex: 1,
                            minWidth: 220,
                          }}
                        >
                          <span style={{ fontWeight: 800 }}>Pilih file gambar</span>
                          <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
                        </label>

                        <div style={{ fontSize: 12, color: "#6b7280" }}>Format: JPG/JPEG/PNG/HEIF</div>
                      </div>

                      {previewUrl && (
                        <div style={{ marginTop: 12 }}>
                          <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 8 }}>Preview</div>
                          <ImageBox src={previewUrl} alt="preview" />
                        </div>
                      )}
                    </div>
                  )}

                  {/* Camera panel */}
                  {inputMode === "camera" && (
                    <div style={panelStyle}>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
                        <Button
                          type="button"
                          onClick={cameraPanelOpen ? closeCamera : openCamera}
                          disabled={cameraBusy}
                          style={isMobile ? { width: "100%" } : undefined}
                        >
                          {cameraPanelOpen ? "Tutup Kamera" : cameraBusy ? "Membuka..." : "Buka Kamera"}
                        </Button>
                      </div>

                      {cameraErr && (
                        <p className="error-text" style={{ marginTop: 10 }}>
                          {cameraErr}
                        </p>
                      )}

                      {cameraPanelOpen && (
                        <div style={{ marginTop: 12 }}>
                          <div style={{ borderRadius: 12, overflow: "hidden", background: "#111" }}>
                            <video
                              ref={videoRef}
                              playsInline
                              autoPlay
                              muted
                              style={{
                                width: "100%",
                                maxHeight: 380,
                                display: "block",
                              }}
                            />
                            {!cameraOn && (
                              <div style={{ padding: 12, color: "#e5e7eb", fontSize: 13 }}>Sedang menyiapkan kamera...</div>
                            )}
                          </div>

                          <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                            <Button
                              type="button"
                              onClick={captureFromCamera}
                              disabled={!cameraOn || cameraBusy}
                              style={isMobile ? { width: "100%" } : undefined}
                            >
                              {cameraBusy ? "Memproses..." : "Ambil Foto"}
                            </Button>
                          </div>

                          <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280", lineHeight: 1.4 }}>
                            Catatan: Jika browser/device tidak mendukung preview kamera, tombol <b>Buka Kamera</b> akan otomatis membuka kamera bawaan device (mode capture).
                          </div>
                        </div>
                      )}

                      {!cameraPanelOpen && (
                        <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280", lineHeight: 1.4 }}>
                          Klik <b>Buka Kamera</b> untuk mengambil foto menggunakan kamera perangkat.
                        </div>
                      )}
                    </div>
                  )}

                  {error && <p className="error-text">{error}</p>}

                  <div style={{ marginTop: 12 }}>
                    <Button
                      type="submit"
                      disabled={!file || classifyBusy || saveStatus.loading}
                      style={isMobile ? { width: "100%" } : undefined}
                    >
                      {classifyBusy ? "Memproses..." : "Proses Klasifikasi"}
                    </Button>
                  </div>

                  {classifyBusy && (
                    <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
                      Sedang mengirim gambar ke server dan menjalankan model...
                    </div>
                  )}
                </form>
              </Card>
            </div>

            {/* =======================
                ✅ Revisi layout Langkah 2
                ======================= */}
            <div>
              <Card title="Hasil Klasifikasi" subtitle="Langkah 2 dari 2">
                {!result && !classifyBusy && (
                  <div style={{ padding: 12, border: "1px dashed #e5e7eb", borderRadius: 12, background: "#fafafa" }}>
                    <div style={{ fontWeight: 800, marginBottom: 6 }}>Belum ada hasil</div>
                    <div style={{ fontSize: 13, color: "#6b7280", lineHeight: 1.5 }}>
                      Pilih gambar pada langkah 1, lalu klik <b>Proses Klasifikasi</b>.
                    </div>
                  </div>
                )}

                {!result && classifyBusy && (
                  <div style={{ padding: 12, border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                      <div style={{ fontWeight: 900 }}>Sedang memproses...</div>
                      <span style={pill("dark")}>Running</span>
                    </div>
                    <div style={{ marginTop: 8, fontSize: 13, color: "#6b7280", lineHeight: 1.5 }}>
                      Mohon tunggu, sistem sedang melakukan inferensi model klasifikasi.
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

                {result && (
                  <div style={{ display: "grid", gap: 12 }}>
                    {/* Ringkasan */}
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <span style={pill(result.segmentation_ready ? "warn" : "good")}>
                        {result.segmentation_ready ? "Lanjut Segmentasi" : "Final (Sehat)"}
                      </span>
                    </div>

                    <div style={kvRow}>
                      <span style={{ color: "#6b7280", fontWeight: 700 }}>Label</span>
                      <span style={{ fontWeight: 900 }}>{result.label || "-"}</span>
                    </div>

                    <div style={kvRow}>
                      <span style={{ color: "#6b7280", fontWeight: 700 }}>Keyakinan</span>
                      <span style={{ fontWeight: 900 }}>
                        {typeof result.confidence === "number" ? `${(result.confidence * 100).toFixed(2)}%` : "-"}
                      </span>
                    </div>

                    {/* ===== Deskripsi penyakit ===== */}
                    {diseaseInfo && (
                      <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
                          <div style={{ fontWeight: 900 }}>Deskripsi</div>
                          <span style={pill("neutral")}>{diseaseInfo.title}</span>
                        </div>

                        <p style={{ margin: "10px 0 0 0", lineHeight: 1.55, color: "#111827" }}>
                          {diseaseInfo.short}
                        </p>

                        {!!(diseaseInfo.sources || []).filter((s) => s?.url).length && (
                          <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280", lineHeight: 1.4 }}>
                            Sumber:&nbsp;
                            {(diseaseInfo.sources || [])
                              .filter((s) => s?.url)
                              .map((s, idx, arr) => (
                                <span key={s.url}>
                                  <a href={s.url} target="_blank" rel="noreferrer">
                                    {s.label || `Referensi ${idx + 1}`}
                                  </a>
                                  {idx < arr.length - 1 ? ", " : ""}
                                </span>
                              ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* ===== Debug probs (tidak tampil default) ===== */}
                    {SHOW_DEBUG_PROBS && result.probs && (
                      <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
                        <div style={{ fontWeight: 900, marginBottom: 8 }}>Debug Probabilitas</div>
                        <div style={{ display: "grid", gap: 6 }}>
                          {Object.entries(result.probs)
                            .sort((a, b) => b[1] - a[1])
                            .map(([cls, p]) => (
                              <div
                                key={cls}
                                style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#111827" }}
                              >
                                <span>{cls}</span>
                                <span style={{ fontVariantNumeric: "tabular-nums" }}>{(Number(p) * 100).toFixed(2)}%</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {/* CTA */}
                    <div style={{ display: "grid", gap: 8 }}>
                      {result.segmentation_ready ? (
                        <Button onClick={handleGoToSegment} style={isMobile ? { width: "100%" } : undefined}>
                          Lihat Area Terinfeksi & Estimasi Keparahan
                        </Button>
                      ) : (
                        <>
                          <Button
                            onClick={handleSave}
                            disabled={saveStatus.loading || classifyBusy}
                            style={isMobile ? { width: "100%" } : undefined}
                          >
                            {saveStatus.loading ? "Menyimpan..." : "Simpan Hasil"}
                          </Button>

                          {saveStatus.msg && (
                            <div style={{ fontSize: 13, color: "#065f46", fontWeight: 700 }}>
                              {saveStatus.msg}
                            </div>
                          )}
                          {saveStatus.err && (
                            <div style={{ fontSize: 13, color: "crimson", fontWeight: 700 }}>
                              {saveStatus.err}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                )}
              </Card>
            </div>
          </div>
        </Card>
      </div>

      {/* anim keyframes (inline) */}
      <style>{`
        @keyframes pulse {
          0% { transform: translateX(-20%); opacity: .65; }
          50% { transform: translateX(10%); opacity: 1; }
          100% { transform: translateX(-20%); opacity: .65; }
        }
      `}</style>
    </div>
  );
}

export default ClassifyPage;
