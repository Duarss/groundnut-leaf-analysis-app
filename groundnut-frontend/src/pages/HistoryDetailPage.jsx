// src/pages/HistoryDetailPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import ImageBox from "../components/ui/ImageBox";
import { fetchHistoryDetail } from "../api/historyApi";
import { getClientId } from "../utils/clientId";
import { formatJakartaTime } from "../utils/dateTime";
import { useIsMobile } from "../utils/useIsMobile";

function prettyLabel(s) {
  if (!s) return "-";
  return String(s).trim().replace(/\s+/g, " ");
}

function parseProbsAny(value) {
  if (!value) return null;
  if (typeof value === "object") return value;
  if (typeof value !== "string") return null;
  try {
    const first = JSON.parse(value);
    if (typeof first === "string") {
      try {
        return JSON.parse(first);
      } catch {
        return null;
      }
    }
    return first && typeof first === "object" ? first : null;
  } catch {
    return null;
  }
}

function normalizeToPct(n) {
  const x = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(x)) return null;
  const pct = x > 1.5 ? x : x * 100;
  return Math.max(0, Math.min(100, pct));
}

function fmtPct(n, digits = 2) {
  const x = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(x)) return "-";
  return `${x.toFixed(digits)}%`;
}

function numOrNull(v) {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

// Catatan: Ini ringkasan gejala umum dari sumber tepercaya (lihat catatan sitasi di chat).
const LABEL_DESC = {
  "ALTERNARIA LEAF SPOT":
    "Umumnya muncul bercak cokelat hingga kehitaman pada daun, sering membentuk pola cincin konsentris (seperti ‘target spot’). Pada serangan berat dapat mempercepat penuaan/kerontokan daun.",
  "LEAF SPOT (EARLY AND LATE)":
    "Bercak daun (early/late leaf spot) ditandai bercak bulat gelap pada permukaan daun; pada sebagian kasus ada halo kekuningan. Dapat menyebabkan defoliasi sehingga menurunkan luas daun sehat.",
  ROSETTE:
    "Rosette ditandai tanaman kerdil, ruas memendek, daun berkelompok (rosette), sering disertai klorosis/mosaik. Biasanya berdampak besar pada pertumbuhan dan hasil.",
  RUST:
    "Karat daun ditandai pustula/benjolan oranye-kecokelatan (sering di permukaan bawah daun). Dapat menyebabkan daun mengering/nekrosis dan menurunkan vigor tanaman.",
  HEALTHY:
    "Daun tampak sehat tanpa gejala bercak, pustula karat, mosaik/klorosis berat, atau deformasi khas penyakit.",
};

function Chip({ children, tone = "neutral", title }) {
  const tones = {
    neutral: { bg: "#f3f4f6", fg: "#374151", bd: "#e5e7eb" },
    good: { bg: "#ecfdf5", fg: "#065f46", bd: "#a7f3d0" },
    warn: { bg: "#fffbeb", fg: "#92400e", bd: "#fde68a" },
    bad: { bg: "#fef2f2", fg: "#991b1b", bd: "#fecaca" },
    dark: { bg: "#111827", fg: "#ffffff", bd: "#111827" },
  };
  const t = tones[tone] || tones.neutral;
  return (
    <span
      title={title}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 10px",
        borderRadius: 999,
        border: `1px solid ${t.bd}`,
        background: t.bg,
        color: t.fg,
        fontSize: 13,
        fontWeight: 700,
        lineHeight: 1.1,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

function ProbabilitiesChart({ probsList }) {
  const [showAll, setShowAll] = useState(false);
  if (!probsList?.length)
    return <div style={{ fontSize: 13, color: "#6b7280" }}>Tidak ada data probabilitas.</div>;

  const shown = showAll ? probsList : probsList.slice(0, 5);

  return (
    <div style={{ display: "grid", gap: 10 }}>
      {shown.map((item, idx) => {
        const isTop = idx === 0;
        return (
          <div key={`${item.label}-${idx}`} style={{ padding: 10, border: "1px solid #e5e7eb", borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <div style={{ fontSize: 14, fontWeight: isTop ? 900 : 800 }}>
                {item.label}{" "}
                {isTop ? <span style={{ fontSize: 12, color: "#6b7280" }}>• Tertinggi</span> : null}
              </div>
              <div style={{ fontSize: 14, fontVariantNumeric: "tabular-nums" }}>{item.pct.toFixed(2)}%</div>
            </div>
            <div style={{ marginTop: 8, height: 10, background: "#e5e7eb", borderRadius: 999 }}>
              <div
                style={{
                  width: `${item.pct}%`,
                  height: "100%",
                  borderRadius: 999,
                  background: isTop ? "#111827" : "#6b7280",
                }}
              />
            </div>
          </div>
        );
      })}

      {probsList.length > 5 && (
        <Button variant="secondary" onClick={() => setShowAll((v) => !v)} style={{ width: "100%" }}>
          {showAll ? "Ringkas (Top 5)" : "Lihat semua"}
        </Button>
      )}
    </div>
  );
}

const HistoryDetailPage = () => {
  const isMobile = useIsMobile();
  const params = useParams();
  const nav = useNavigate();

  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  const analysisId = (params.analysisId || params.id || "").toString().trim();
  const cid = useMemo(() => getClientId(), []);

  useEffect(() => {
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setErr("");

    fetchHistoryDetail(analysisId)
      .then((d) => alive && setData(d))
      .catch((e) => {
        if (!alive) return;
        setErr(e?.message || "Gagal memuat detail");
        setData(null);
      })
      .finally(() => alive && setLoading(false));

    return () => {
      alive = false;
    };
  }, [analysisId]);

  const labelNorm = useMemo(() => prettyLabel(data?.label).toUpperCase(), [data]);
  const labelDesc = useMemo(() => LABEL_DESC[labelNorm] || "Deskripsi untuk label ini belum tersedia.", [labelNorm]);

  const probsObj = useMemo(() => parseProbsAny(data?.probs_json), [data]);

  const probsList = useMemo(() => {
    if (!probsObj || typeof probsObj !== "object") return null;
    const entries = Object.entries(probsObj)
      .map(([label, prob]) => ({ label: prettyLabel(label), pct: normalizeToPct(prob) }))
      .filter((x) => x.label && x.pct != null)
      .sort((a, b) => b.pct - a.pct);
    return entries.length ? entries : null;
  }, [probsObj]);

  const origImgUrl = useMemo(() => {
    const p = data?.orig_image_path;
    if (!p) return null;
    return `/api/storage?path=${encodeURIComponent(p)}&client_id=${encodeURIComponent(cid)}`;
  }, [data, cid]);

  const overlayImgUrl = useMemo(() => {
    const p = data?.seg_overlay_path;
    if (!p) return null;
    return `/api/storage?path=${encodeURIComponent(p)}&client_id=${encodeURIComponent(cid)}`;
  }, [data, cid]);

  const confidenceText =
    typeof data?.confidence === "number" ? `${(data.confidence * 100).toFixed(2)}%` : "-";

  // ===== SAD display =====
  const hasSeverity = useMemo(() => {
    const pct = data?.severity_pct;
    return data?.seg_enabled && pct !== null && pct !== undefined && Number.isFinite(Number(pct));
  }, [data]);

  const severityPctText = useMemo(() => {
    if (!hasSeverity) return "-";
    return fmtPct(Number(data.severity_pct), 2);
  }, [hasSeverity, data]);

  const sadObj = useMemo(() => {
    const direct = data?.sad || data?.severity_sad || data?.severity?.sad || null;
    if (direct && typeof direct === "object") return direct;

    const sadJson = data?.sad_json || data?.severity_sad_json || null;
    if (sadJson && typeof sadJson === "string") {
      try {
        const parsed = JSON.parse(sadJson);
        return parsed && typeof parsed === "object" ? parsed : null;
      } catch {
        return null;
      }
    }
    return null;
  }, [data]);

  const sadClassIndex = useMemo(() => {
    const v = sadObj?.class_index ?? data?.severity_sad_class_index ?? data?.sad_class_index ?? null;
    const n = numOrNull(v);
    return n == null ? null : Math.round(n);
  }, [sadObj, data]);

  const sadRangeText = useMemo(() => {
    const rng = sadObj?.range_pct;
    if (Array.isArray(rng) && rng.length === 2) return `${Number(rng[0])}–${Number(rng[1])}%`;
    return "-";
  }, [sadObj]);

  const sadMidText = useMemo(() => {
    const mid = numOrNull(sadObj?.midpoint_pct);
    if (mid == null) return "-";
    return `${mid.toFixed(1)}%`;
  }, [sadObj]);

  const sadScheme = useMemo(() => {
    const s = (sadObj?.scheme || "").toString().trim();
    return s || "Horsfall–Barratt (12-class)";
  }, [sadObj]);

  const severityChipTone = useMemo(() => {
    const pct = numOrNull(data?.severity_pct);
    if (pct == null) return "neutral";
    if (pct >= 60) return "bad";
    if (pct >= 20) return "warn";
    return "good";
  }, [data]);

  return (
    <div style={{ padding: 16, maxWidth: 980, margin: "0 auto" }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <Button onClick={() => nav("/history")} style={isMobile ? { width: "100%" } : undefined}>
          Kembali
        </Button>
        <h2 style={{ fontSize: 22, fontWeight: 900, margin: 0 }}>Detail Analisis</h2>
        {analysisId && <Chip title="ID Analisis">{"ID: " + analysisId}</Chip>}
      </div>

      {loading && <div style={{ marginTop: 12 }}>Memuat...</div>}
      {!loading && err && <div style={{ marginTop: 12, color: "crimson" }}>{err}</div>}

      {!loading && data && (
        <div style={{ marginTop: 8, display: "grid", gap: 12 }}>
          <Card
            title={
              <span style={{ fontSize: 12, color: "#6b7280" }}>
                {`${formatJakartaTime(data.created_at)} (WIB)`}
              </span>
            }
          >
            <div style={{ fontSize: 20, fontWeight: 900 }}>{prettyLabel(data.label)}</div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
              {/* <Chip tone="dark">Label: {prettyLabel(data.label)}</Chip> */}
              <Chip title="Keyakinan model">Keyakinan Prediksi: {confidenceText}</Chip>
              <Chip tone={data.seg_enabled ? "good" : "neutral"}>Segmentasi: {data.seg_enabled ? "Ya" : "Tidak"}</Chip>

              {hasSeverity ? (
                <>
                  <Chip tone={severityChipTone} title={`Estimasi keparahan (%), dipetakan ke kelas SAD (${sadScheme})`}>
                    Estimasi Keparahan: {severityPctText} • SAD C{sadClassIndex ?? "-"}
                  </Chip>
                  <Chip tone={severityChipTone} title={`Rentang dan midpoint SAD (${sadScheme})`}>
                    Rentang: <b>{sadRangeText}</b> • Midpoint: <b>{sadMidText}</b>
                  </Chip>
                </>
              ) : null}
            </div>
          </Card>

          {/* Deskripsi Label */}
          <Card title="Deskripsi Label">
            <div style={{ fontSize: 14, color: "#111827", lineHeight: 1.6 }}>
              {labelDesc}
            </div>
            <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
              Catatan: deskripsi ini adalah ringkasan gejala umum untuk membantu interpretasi hasil model.
            </div>
          </Card>

          <Card>
            <div style={{ fontSize: 16, fontWeight: 900 }}>Gambar</div>
            <div
              style={{
                marginTop: 12,
                display: "grid",
                gridTemplateColumns: isMobile ? "1fr" : "repeat(2, minmax(0, 1fr))",
                gap: 12,
              }}
            >
              <div>
                <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 8 }}>Original</div>
                {origImgUrl ? (
                  <ImageBox
                    src={origImgUrl}
                    alt="Original"
                    isMobile={isMobile}
                    maxHeightMobile={240}
                    maxHeightDesktop={420}
                    allowExpand={true}
                  />
                ) : (
                  <div style={{ fontSize: 13, color: "#6b7280" }}>Gambar original tidak tersedia.</div>
                )}
              </div>

              <div>
                <div style={{ fontSize: 13, fontWeight: 800, marginBottom: 8 }}>Visualisasi Area Terinfeksi</div>
                {data.seg_enabled && overlayImgUrl ? (
                  <ImageBox
                    src={overlayImgUrl}
                    alt="Overlay"
                    isMobile={isMobile}
                    maxHeightMobile={240}
                    maxHeightDesktop={420}
                    allowExpand={true}
                  />
                ) : (
                  <div style={{ fontSize: 13, color: "#6b7280" }}>
                    {data.seg_enabled ? "Overlay tidak tersedia." : "Segmentasi tidak aktif pada analisis ini."}
                  </div>
                )}
              </div>
            </div>
          </Card>

          <Card>
            <div style={{ fontSize: 16, fontWeight: 900 }}>Probabilitas Label</div>
            <div style={{ marginTop: 12 }}>
              {!data?.probs_json ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>Tidak ada data probabilitas.</div>
              ) : !probsList ? (
                <div style={{ fontSize: 13, color: "#6b7280" }}>
                  Data probabilitas ada, tapi belum bisa ditampilkan sebagai grafik.
                </div>
              ) : (
                <ProbabilitiesChart probsList={probsList} />
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default HistoryDetailPage;
